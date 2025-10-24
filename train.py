import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
import re
from functools import partial

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from flax.training import train_state
from flax import struct
from flax.jax_utils import replicate as R

# IMPORTANT: ignore id for cross entropy
from utils.logging_util import log_for_0, GoodLogger
from utils.metric_utils import Timer, MyMetrics
from utils.info_utils import print_params
from utils.ckpt_utils import save_checkpoint, restore_checkpoint
from utils.vis_util import sudoku_to_image
import models.models_all
import input_pipeline
import optimizers
from input_pipeline import IGNORE_LABEL_ID

# JAX globals
PCI = jax.process_index()
PCC = jax.process_count()
LDC = jax.local_device_count()

@struct.dataclass
class MutableTrainState(train_state.TrainState):
    buffers: dict
    consts: dict


# train state creation
def create_train_state(rng, config, total_steps, batch_size):
    model_cls = getattr(models.models_all, config.model.name)
    model = model_cls(**{k: v for k, v in config.model.to_dict().items() if k != "name"}, **{'seq_len': config.dataset.seq_len, 'vocab_size': config.dataset.vocab_size, 'num_puzzle_identifiers': config.dataset.num_puzzle_identifiers, 'batch_size': batch_size})
    fake_batch_shape = {
        'inputs': (2, config.dataset.seq_len),
        'labels': (2, config.dataset.seq_len),
        'puzzle_identifiers': (2,),
        'base_puzzle_indices': (2,),
        'augmentation_indices': (2,),
    }
    fake_batch = {k: jnp.ones(v, dtype=jnp.int32) for k, v in fake_batch_shape.items()}
    
    rng, init_rng = jax.random.split(rng)
    params_init_rng, const_init_rng = jax.random.split(init_rng, 2)
    model_variables = jax.jit(partial(model.init, method=model.init_fn))({'params': params_init_rng, 'const': const_init_rng}, fake_batch)
    
    print_params(model_variables['params'])

    tx, lr_fn = optimizers.build_optimizer(config.training, total_steps)
    
    state = MutableTrainState.create(
        params=model_variables['params'],
        buffers=model_variables['buffer'],
        consts=model_variables['const'],
        tx=tx,
        apply_fn=model.apply,
    )

    return model, state, lr_fn

def s(x, epsilon=1e-6):
    # NOTE{zhh}: jax where is NOT ALWAYS short-circuiting!!!
    return jnp.where(x<0, 1/jnp.clip(1 - x, epsilon), x + 1)

def log_stablemax(x, axis=-1):
    s_x = s(x)
    return jnp.log(s_x/jnp.sum(s_x, axis=axis, keepdims=True))

def stablemax_cross_entropy(logits, labels, ignore_index: int = IGNORE_LABEL_ID):
    logprobs = log_stablemax(logits.astype(jnp.float64), axis=-1)

    valid_mask = labels != ignore_index
    transformed_labels = jnp.where(valid_mask, labels, 0)
    prediction_logprobs = jnp.take_along_axis(logprobs, jnp.expand_dims(transformed_labels.astype(jnp.int32), axis=-1), axis=-1).squeeze(-1)

    return -jnp.where(valid_mask, prediction_logprobs, 0)

def softmax_cross_entropy(logits, labels, ignore_index: int = IGNORE_LABEL_ID):
    # Cast logits to f32
    # Flatten logits
    raise NotImplementedError("I believe there is subtlety in ignore_index here")
    return optax.softmax_cross_entropy(logits.astype(jnp.float32).reshape(-1, logits.shape[-1]), jax.nn.one_hot(labels.astype(jnp.int32).reshape(-1), logits.shape[-1]), axis=-1).reshape(labels.shape) * (labels != ignore_index)

LOSS_FUNCTIONS = {
    'stablemax_cross_entropy': stablemax_cross_entropy,
    'softmax_cross_entropy': softmax_cross_entropy,
}

def valid_mean(validity, arr):
    assert validity.shape == arr.shape, f"validity.shape: {validity.shape}, arr.shape: {arr.shape}"
    arr = arr.astype(jnp.float32)
    return jnp.where(validity, arr, 0*arr).sum(axis=0) / jnp.maximum(jnp.sum(validity, axis=0), 1)

def act_loss_and_metrics(buffer, outputs, loss_fn, train=True, label_info=None):
    # TODO{zhh}: verify correctness, especially data validity
    B, L, D = outputs["logits"].shape
    # labels.shape: [B, SeqLen]

    carry = buffer['carry']
    labels = carry.current_data["labels"] if label_info is None else label_info

    assert labels.shape == (B, L), f"labels.shape: {labels.shape}, B: {B}, L: {L}"

    mask = (labels != IGNORE_LABEL_ID) # [B, SeqLen]

    # acc
    valid_data_num = jnp.sum(mask, axis=-1) # [B,]
    is_correct = mask & (jnp.argmax(outputs["logits"], axis=-1) == labels) # [B, SeqLen]
    seq_is_correct = jnp.sum(is_correct, axis=-1) == valid_data_num # [B,]
    q_halt_acc = (outputs['q_halt_logits'] >= 0) == seq_is_correct # [B,]
    valid_outputs = carry.halted & (valid_data_num > 0) # [B,]
    
    # losses
    ce_loss = loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID)
    assert ce_loss.shape == (B, L), f"ce_loss.shape: {ce_loss.shape}, B: {B}, L: {L}"
    ce_loss = (ce_loss / jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1)).sum(axis=-1).mean(axis=0) # []
    q_halt_loss = optax.sigmoid_binary_cross_entropy(outputs['q_halt_logits'], seq_is_correct).mean(axis=0) # []

    q_continue_loss = q_halt_loss*0
    if train:
        q_continue_loss = optax.sigmoid_binary_cross_entropy(outputs['q_continue_logits'], outputs['target_q_continue']).mean(axis=0)
        
    loss = ce_loss + (q_halt_loss + q_continue_loss) * 0.5

    # Basic rates (means across the local batch/device)
    metrics = {
        "valid_data_rate": (valid_data_num > 0).mean(axis=0),
        "valid_output_rate": valid_outputs.mean(axis=0),
        "acc_per_token": is_correct.mean(axis=1).mean(axis=0), # just monitor ce
        "pass_at_this_round": (valid_outputs & seq_is_correct).mean(axis=0), # this is the final acc

        # We still keep the per-device conditional mean for quick inspection,
        # but also emit numerator/denominator so callers can compute a global
        # weighted value across devices/hosts which is the correct statistic.
    "valid_pass@1": valid_mean(valid_outputs, seq_is_correct),
    "valid_pass@1_n": jnp.where(valid_outputs, seq_is_correct.astype(jnp.float32), 0).sum(axis=0),

    "q_halt_acc": valid_mean(valid_outputs, q_halt_acc),
    "q_halt_acc_n": jnp.where(valid_outputs, q_halt_acc.astype(jnp.float32), 0).sum(axis=0),

    # inference steps: emit sum of steps over valid outputs so the true
    # average steps can be derived globally. We use a single shared
    # denominator `valid_outputs_count` because all three metrics share it.
    "inference_steps": valid_mean(valid_outputs, carry.steps),
    "inference_steps_n": jnp.where(valid_outputs, carry.steps.astype(jnp.float32), 0).sum(axis=0),

    # shared denominator: count of valid outputs per device
    "valid_outputs_count": jnp.sum(valid_outputs, axis=0),

        # loss
        "ce_loss": ce_loss,
        "q_halt_loss": q_halt_loss,
        "q_continue_loss": q_continue_loss,
        "loss": loss,
    }
    
    assert all([v.shape == () for v in metrics.values()]), f"metrics: {dict((k, v.shape) for k, v in metrics.items())}"
    assert all(v.shape == () for v in [loss, ce_loss, q_halt_loss, q_continue_loss]), f"loss: {loss.shape}, ce_loss: {ce_loss.shape}, q_halt_loss: {q_halt_loss.shape}, q_continue_loss: {q_continue_loss.shape}"
    
    # return loss, metrics, outputs # for DEBUG
    return loss, metrics, outputs['logits'].argmax(axis=-1).astype(jnp.int32)

def compute_metrics(dict_losses):
    metrics = jax.lax.all_gather(dict_losses, axis_name="batch")
    metrics = jax.tree_map(lambda x: x.flatten(), metrics)  # (batch_size,)
    return metrics

def train_step(state: MutableTrainState, batch, loss_fn, init_rng, model, lr_fn):
    rng_step = jax.random.fold_in(init_rng, state.step)
    def L(params):
        variables = {'params': params, 'buffer': state.buffers, 'const': state.consts}
        out, new_model_state = state.apply_fn(variables, batch, rng=rng_step, train=True, mutable=['buffer'])
        loss, metrics, vis = act_loss_and_metrics(new_model_state['buffer'], out, loss_fn, train=True)
        return loss, (metrics, vis, new_model_state['buffer'])
    grad_fn = jax.value_and_grad(L, has_aux=True)
    (loss, (metrics, vis, new_buffers)), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_state = state.apply_gradients(grads=grads, buffers=new_buffers)
    metrics['learning_rate'] = lr_fn(state.step)

    return new_state, compute_metrics(metrics), vis

def eval_step(state: MutableTrainState, batch, loss_fn, model):
    variables = {'params': state.params, 'buffer': state.buffers, 'const': state.consts}
    labels = batch['labels']
    batch['labels'] *= 0 # avoid cheating
    out, new_state = model.apply(variables, batch, mutable=['buffer'], method=model.inference)
    batch['labels'] = labels
    _, metrics, vis = act_loss_and_metrics(new_state['buffer'], out, loss_fn, train=False, label_info=labels)
    logits = out['logits']
    predictions = logits.argmax(axis=-1).astype(jnp.int32)
    valid_mask = labels != IGNORE_LABEL_ID
    seq_correct = jnp.all(jnp.where(valid_mask, predictions == labels, True), axis=-1)

    carry = new_state['buffer']['carry']
    halted = carry.halted
    steps = carry.steps.astype(jnp.int32)
    stop_steps = jnp.where(halted, steps, 17)  # shape: (local_device_count, local_batch)
    # Padding rows are introduced when the dataloader needs to round out the last microbatch.
    # `zhh_is_pad` marks those rows with 1s for every token position.
    is_pad = (batch['zhh_is_pad'] > 0).all(axis=-1)  # shape matched to stop_steps

    stats = {
        "stop_steps": jax.lax.all_gather(stop_steps, axis_name='batch'),
        "seq_correct": jax.lax.all_gather(seq_correct, axis_name='batch'),
        "is_pad": jax.lax.all_gather(is_pad, axis_name='batch'),
    }
    stats = {k: v.reshape(-1) for k, v in stats.items()}

    return compute_metrics(metrics), vis, stats

def pure_inference_step(state: MutableTrainState, batch, model):
    variables = {'params': state.params, 'buffer': state.buffers, 'const': state.consts}
    out, new_state = model.apply(variables, batch, mutable=['buffer'], method=model.inference)
    pred = out['logits'].argmax(axis=-1).astype(jnp.int32)
    return jax.lax.all_gather(pred, axis_name='batch'), jax.lax.all_gather(batch['zhh_is_pad'], axis_name='batch')

def wandb_init(config, workdir):
    if jax.process_index() == 0 and config.wandb.on_use:
        import wandb
        wandb.init(project="TRM", dir=workdir, notes=config.wandb.notes, entity='evazhu-massachusetts-institute-of-technology')
        wandb.config.update(config.to_dict())
        try:
            ka = re.search(
                r"kmh-tpuvm-v[23456e]+-(\d+)(-preemptible)?(-spot)?-.*yang-(\d+)", workdir
            ).group()
        except AttributeError:
            ka = '💩' * 10 + '马勒戈壁'
        wandb.config.update({"ka": ka})

def train_and_evaluate(config, workdir):
    log_for_0('\n' + str(config))
    rng = jax.random.key(config.training.seed)
    log_for_0(f"JAX process index: {PCI} started with local devices {jax.local_devices()}")
    
    # load dataset
    dataset_cfg = config.dataset
    global_batch_size = config.training.batch_size
    global_eval_batch_size = config.training.eval_batch_size
    assert global_batch_size == global_eval_batch_size, "Not implemented two different batch sizes"
    device_batch_size = global_batch_size // PCC
    device_eval_batch_size = global_eval_batch_size // PCC
    assert global_batch_size % (PCC) == 0 and device_batch_size % LDC == 0, f"global_batch_size: {global_batch_size}, PCC: {PCC}, LDC: {LDC}, local_batch_size: {device_batch_size}"
    assert global_eval_batch_size % (PCC) == 0 and device_eval_batch_size % LDC == 0, f"global_eval_batch_size: {global_eval_batch_size}, PCC: {PCC}, LDC: {LDC}, local_eval_batch_size: {device_eval_batch_size}"
    eval_split = getattr(config.training, "eval_split", "test")
    train_dl, train_steps_per_epoch, train_metadata, _ = input_pipeline.create_split(
        dataset_cfg, split='train', batch_size=device_batch_size
    )
    eval_dataset_overrides = None
    if eval_split == 'train':
        eval_dataset_overrides = {
            "augmentations_per_puzzle": int(getattr(config.training, "eval_augmentations_per_puzzle", 0))
        }
    eval_dl, eval_steps_per_epoch, eval_metadata, _ = input_pipeline.create_split(
        dataset_cfg,
        split=eval_split,
        batch_size=device_eval_batch_size,
        dataset_overrides=eval_dataset_overrides,
        shuffle=False,
    )
    log_for_0(f"train_steps_per_epoch: {train_steps_per_epoch}, eval_steps_per_epoch ({eval_split}): {eval_steps_per_epoch}")
    total_steps = config.training.epochs * train_steps_per_epoch
    
    # init model
    rng, init_rng = jax.random.split(rng)
    model, state, lr_fn = create_train_state(init_rng, config, total_steps, device_batch_size // LDC)
    del init_rng
    
    # restore checkpoint if any
    epoch_offset = 0
    if config.load_from:
        state = restore_checkpoint(state, config.load_from, ignore_keys=('buffers',))
        epoch_offset = int(state.step) // train_steps_per_epoch
        log_for_0(f"Resuming from checkpoint {config.load_from}, epoch_offset: {epoch_offset}")

    state = R(state)

    # compile train_step, eval_step
    loss_fn = LOSS_FUNCTIONS[config.training.loss_fn]
    rng, train_rng = jax.random.split(rng)
    p_train_step = jax.pmap(
        partial(
            train_step,
            loss_fn=loss_fn,
            init_rng=train_rng,
            model=model,
            lr_fn=lr_fn,
        ),
        axis_name='batch',
        donate_argnums=(0,),
    )
    p_eval_step = jax.pmap(partial(eval_step, loss_fn=loss_fn, model=model), axis_name='batch')
    test_batch = input_pipeline.prepare_batch_data(next(iter(train_dl)), batch_size=device_batch_size, dataset_metdata=train_metadata)
    log_for_0("Compiling p_train_step and p_eval_step ...")
    timer = Timer()
    p_train_step = p_train_step.lower(state, test_batch).compile()
    p_eval_step = p_eval_step.lower(state, test_batch).compile()
    log_for_0(f"train flops: {p_train_step.cost_analysis()[0]['flops'] / 1e12:.2f} TFLOPS")
    log_for_0(f"eval flops: {p_eval_step.cost_analysis()[0]['flops'] / 1e12:.2f} TFLOPS")
    log_for_0(f"p_train_step and p_eval_step compiled in {timer}.")

    # wandb init
    wandb_init(config, workdir)
    logger = GoodLogger(workdir, use_wandb=config.wandb.on_use)
    
    # training loop
    
    # TODO{ZHH}: this is for debug. remove it afterwards
    # epoch_offset = -1
    if config.just_evaluate:
        log_for_0("Got config.just_evaluate=True. Just evaluate the model once and exit...")
        epoch_offset = config.training.epochs - 1
        train_dl = ()
    print(config.training.log_per_epoch)
    print(config.training.eval_interval)

    timer.reset()
    train_metrics = MyMetrics(reduction='avg')
    for epoch in range(epoch_offset, config.training.epochs):
        log_for_0(f"Epoch {epoch} ...")
        step = epoch * train_steps_per_epoch
        ep = epoch
        for n_batch, batch in enumerate(train_dl):
            batch = input_pipeline.prepare_batch_data(batch, batch_size=device_batch_size,
             dataset_metdata=train_metadata)
            #log_for_0(f"Prepared batch {n_batch} for training in epoch {epoch}.")
            state, metrics, vis = p_train_step(state, batch)
            train_metrics.update(metrics)
            step = epoch * train_steps_per_epoch + n_batch
            ep = epoch + n_batch / train_steps_per_epoch
        # Print the length of train rounds
        #print(f"Epoch {epoch} done in {timer}. Length of this round: {step - epoch * train_steps_per_epoch + 1} steps.")
        if (epoch + 1) % config.training.log_per_epoch == 0 and not config.just_evaluate:
            summary = train_metrics.compute_and_reset()
            # Compute globally-weighted conditional metrics if n/d pairs exist.
            # Use the shared denominator if present to compute weighted metrics
            if "valid_outputs_count" in summary:
                d = summary.pop("valid_outputs_count")
                if "valid_pass@1_n" in summary:
                    n = summary.pop("valid_pass@1_n")
                    summary["valid_pass@1"] = (n / d) if d != 0 else float("nan")
                if "q_halt_acc_n" in summary:
                    n = summary.pop("q_halt_acc_n")
                    summary["q_halt_acc"] = (n / d) if d != 0 else float("nan")
                if "inference_steps_n" in summary:
                    n = summary.pop("inference_steps_n")
                    summary["inference_steps"] = (n / d) if d != 0 else float("nan")

            summary = {f"train/{k}": v for k, v in summary.items()}
            summary["epochs_per_second"] = (
                config.training.log_per_epoch / timer.elapse_with_reset()
            )
            summary.update({"ep": ep, "step": step})
            logger.log(step + 1, summary)
            # for k, v in vis.items():
            #     log_for_0(f'vis[{k}]: {(v.max(), v.min(), v.mean(), v.std())}') # DEBUG
            #     if jnp.isnan(v).any():
            #         log_for_0(f'Training diverged at epoch {epoch}, step {step}. Aborted.')
            #         exit(1)
            if jnp.isnan(vis).any():
                log_for_0(f'Training diverged at epoch {epoch}, step {step}. Aborted.')
                exit(1)
            if epoch == -1: # means debug run
                break

        # eval
        timer.reset()
        if config.just_evaluate or (epoch + 1) % config.training.eval_interval == 0:
            log_for_0(f"Evaluating on epoch {epoch} ...")
            eval_metrics = MyMetrics(reduction='avg')
            eval_stop_steps = []
            eval_seq_correct = []
            eval_is_pad = []
            for n_batch, batch in enumerate(eval_dl):
                batch = input_pipeline.prepare_batch_data(batch, batch_size=device_batch_size, dataset_metdata=eval_metadata)
                metrics, vis, stats = p_eval_step(state, batch)
                eval_metrics.update(metrics)
                eval_stop_steps.append(np.asarray(stats["stop_steps"]).astype(np.int32))
                eval_seq_correct.append(np.asarray(stats["seq_correct"]).astype(bool))
                eval_is_pad.append(np.asarray(stats["is_pad"]).astype(bool))
                if epoch == -1: # means debug run
                    break
            eval_time = timer.elapse_with_reset()
            eval_summary = eval_metrics.compute()
            # compute globally-weighted metrics from n/d pairs if present
            if "valid_outputs_count" in eval_summary:
                d = eval_summary.pop("valid_outputs_count")
                if "valid_pass@1_n" in eval_summary:
                    n = eval_summary.pop("valid_pass@1_n")
                    eval_summary["valid_pass@1"] = (n / d) if d != 0 else float("nan")
                if "q_halt_acc_n" in eval_summary:
                    n = eval_summary.pop("q_halt_acc_n")
                    eval_summary["q_halt_acc"] = (n / d) if d != 0 else float("nan")
                if "inference_steps_n" in eval_summary:
                    n = eval_summary.pop("inference_steps_n")
                    eval_summary["inference_steps"] = (n / d) if d != 0 else float("nan")

            log_for_0(f"Epoch {epoch} evaluation done. " + ", ".join([f"{k}: {v:.4f}" for k, v in eval_summary.items()]))

            if eval_stop_steps:
                stop_steps = np.concatenate(eval_stop_steps, axis=0)  # shape: (num_samples,)
                seq_correct = np.concatenate(eval_seq_correct, axis=0)
                is_pad = np.concatenate(eval_is_pad, axis=0)
                valid_mask = ~is_pad  # padded rows are excluded from statistics
                stop_steps = stop_steps[valid_mask]
                seq_correct = seq_correct[valid_mask]
            else:
                stop_steps = np.array([], dtype=np.int32)
                seq_correct = np.array([], dtype=bool)

            # Clamp to [0, 17]; 17 represents sequences that never emitted a halt signal within 16 steps.
            stop_steps = np.clip(stop_steps.astype(np.int32), 0, 17)
            correct_steps = stop_steps[seq_correct]
            incorrect_steps = stop_steps[~seq_correct]

            avg_stop_correct = float(correct_steps.mean()) if correct_steps.size else float("nan")
            avg_stop_incorrect = float(incorrect_steps.mean()) if incorrect_steps.size else float("nan")
            hist_correct = np.bincount(correct_steps, minlength=18) if correct_steps.size else np.zeros(18, dtype=np.int32)
            hist_incorrect = np.bincount(incorrect_steps, minlength=18) if incorrect_steps.size else np.zeros(18, dtype=np.int32)

            log_for_0(
                f"Epoch {epoch} eval time: {eval_time:.2f} s. "
                f"Avg stop steps (correct/incorrect): {avg_stop_correct:.2f}/{avg_stop_incorrect:.2f}"
            )
            log_for_0(
                f"Stop-time histograms -> correct: {hist_correct.tolist()}, incorrect: {hist_incorrect.tolist()}"
            )

            histogram_image = None
            if plt is not None and stop_steps.size:
                fig, ax = plt.subplots(figsize=(8, 4))
                x = np.arange(18)
                width = 0.4
                ax.bar(x - width / 2, hist_correct, width=width, label="Correct")
                ax.bar(x + width / 2, hist_incorrect, width=width, label="Incorrect")
                ax.set_xticks(x)
                ax.set_xlabel("Steps")
                ax.set_ylabel("Count")
                ax.set_title("Stop Time Distribution")
                ax.legend()
                fig.tight_layout()
                fig.canvas.draw()
                histogram_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                histogram_image = histogram_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)

            summary = {f"eval/{k}": v for k, v in eval_summary.items()}
            summary.update({
                "eval/eval_time_seconds": float(eval_time),
                "eval/avg_stop_steps_correct": avg_stop_correct,
                "eval/avg_stop_steps_incorrect": avg_stop_incorrect,
            })
            logger.log(step + 1, summary)

            if histogram_image is not None:
                logger.log_image(step + 1, {"eval_stop_time_distribution": histogram_image})
            
            # TODO: log visualizations
            assert len(batch['labels'][0]) >= config.training.num_vis, f"batch['labels'][0].shape: {batch['labels'][0].shape}, config.training.num_vis: {config.training.num_vis}"
            logger.log_image(step + 1, {f'data_{i}': sudoku_to_image(batch['labels'][0][i], prompt=batch['inputs'][0][i]) for i in range(config.training.num_vis)})
            logger.log_image(step + 1, {f'completion_{i}': sudoku_to_image(vis[0][i], prompt=batch['inputs'][0][i]) for i in range(config.training.num_vis)})
            
            if config.just_evaluate:
                return jax.random.normal(jax.random.key(0), ()).block_until_ready()

        # save checkpoint
        if (epoch + 1) % config.training.checkpoint_interval == 0 or (epoch + 1) == config.training.epochs:
            save_checkpoint(state, workdir, ignore_keys=('buffers',))
            log_for_0(f"Epoch {epoch} checkpoint saved.")
    
    return jax.random.normal(jax.random.key(0), ()).block_until_ready()  # wait for all computations to finish
            
def inference_folder(config, workdir):
    log_for_0('\n' + str(config))
    rng = jax.random.key(config.training.seed)
    log_for_0(f"JAX process index: {PCI} started with local devices {jax.local_devices()}")
    
    global_batch_size = config.training.batch_size
    device_batch_size = global_batch_size // PCC
    assert global_batch_size % (PCC) == 0 and device_batch_size % LDC == 0, f"global_batch_size: {global_batch_size}, PCC: {PCC}, LDC: {LDC}, local_batch_size: {device_batch_size}"
    if PCC > 1:
        log_for_0("[Warning] this function is inefficient when PCC > 1.")
    
    dl, steps_per_epoch, metadata = input_pipeline.create_split_from_folder(config.dataset.dataset_path, batch_size=device_batch_size)
    
    rng, init_rng = jax.random.split(rng)
    model, state, _ = create_train_state(init_rng, config, total_steps=114514, batch_size=device_batch_size // LDC)
    del init_rng
    
    assert config.load_from, "You must specify config.load_from for inference"
    state = restore_checkpoint(state, config.load_from, ignore_keys=('buffers',))
    log_for_0(f"Resuming from checkpoint {config.load_from}")
    
    state = R(state)
    loss_fn = lambda: None
    
    rng, eval_rng = jax.random.split(rng)
    p_inference_step = jax.pmap(partial(pure_inference_step, model=model), axis_name='batch')
    test_batch = input_pipeline.prepare_batch_data(next(iter(dl)), batch_size=device_batch_size, dataset_metdata=metadata)
    log_for_0("Compiling p_inference_step ...")
    timer = Timer()
    p_inference_step = p_inference_step.lower(state, test_batch).compile()
    log_for_0(f"p_inference_step compiled in {timer}.")

    all_samples = []
    for n_batch, batch in enumerate(dl):
        batch = input_pipeline.prepare_batch_data(batch, batch_size=device_batch_size, dataset_metdata=metadata)
        pred, is_pad = p_inference_step(state, batch)
        pred = pred[0].reshape(-1, pred.shape[-1]) # [B, SeqLen]
        is_pad = is_pad[0].reshape(-1, is_pad.shape[-1]) # [B, SeqLen]

        if n_batch % 200 == 0:
            log_for_0(f"Inference batch {n_batch}/{steps_per_epoch} done.")
        if n_batch == steps_per_epoch - 1:
            # remove padding
            num_padded = (is_pad == 1).all(axis=-1).sum()
            num_padded = int(num_padded)
            log_for_0(f"Last batch has {num_padded} padded samples.")
            pred = pred[:-num_padded] if num_padded > 0 else pred

        all_samples.append(jax.device_get(pred))

    all_samples = np.concatenate(all_samples, axis=0) # [N, SeqLen]
    log_for_0(f"Inference on {len(all_samples)} samples done in {timer}.")
    if jax.process_index() == 0:
        np.save(f"{workdir}/inference_results.npy", all_samples)
    
    log_for_0(f"Inference done. Samples saved to {workdir}.")
