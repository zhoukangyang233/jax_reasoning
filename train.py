import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import re
from functools import partial
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

def act_loss_and_metrics(buffer, outputs, loss_fn, train=True):
    # TODO{zhh}: verify correctness, especially data validity
    B, L, D = outputs["logits"].shape
    # labels.shape: [B, SeqLen]

    carry = buffer['carry']
    labels = carry.current_data["labels"]

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
    ce_loss = (ce_loss / jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1)).sum(axis=-1).mean(axis=0) # [B,]
    q_halt_loss = optax.sigmoid_binary_cross_entropy(outputs['q_halt_logits'], seq_is_correct).mean(axis=0) # [B,]

    q_continue_loss = q_halt_loss*0
    if train:
        q_continue_loss = optax.sigmoid_binary_cross_entropy(outputs['q_continue_logits'], outputs['target_q_continue']).mean(axis=0)
        
    loss = ce_loss + (q_halt_loss + q_continue_loss) * 0.5
    
    metrics = {
        "valid_data_rate": (valid_data_num > 0).mean(axis=0),
        "valid_output_rate": valid_outputs.mean(axis=0),
        "acc_per_token": is_correct.mean(axis=1).mean(axis=0), # just monitor ce
        "pass@1": (valid_outputs & seq_is_correct).mean(axis=0), # this is the final acc
        "valid_output_acc": valid_mean(valid_outputs, seq_is_correct),
        "q_halt_acc": valid_mean(valid_outputs, q_halt_acc),
        "inference_steps": valid_mean(valid_outputs, carry.steps),

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
    metrics = dict_losses.copy()
    metrics = jax.lax.all_gather(metrics, axis_name="batch")
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
    out, new_state = model.apply(variables, batch, mutable=['buffer'], method=model.inference)
    _, metrics, vis = act_loss_and_metrics(new_state['buffer'], out, loss_fn, train=False)
    return compute_metrics(metrics), vis

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
    train_dl, train_steps_per_epoch, train_metadata = input_pipeline.create_split(dataset_cfg, split='train', batch_size=device_batch_size)
    eval_dl, eval_steps_per_epoch, eval_metadata = input_pipeline.create_split(dataset_cfg, split='test', batch_size=device_eval_batch_size)
    log_for_0(f"train_steps_per_epoch: {train_steps_per_epoch}, eval_steps_per_epoch: {eval_steps_per_epoch}")
    total_steps = config.training.epochs * train_steps_per_epoch
    
    # init model
    rng, init_rng = jax.random.split(rng)
    model, state, lr_fn = create_train_state(init_rng, config, total_steps, device_batch_size // LDC)
    del init_rng
    
    # restore checkpoint if any
    epoch_offset = 0
    if config.load_from:
        state = restore_checkpoint(state, config.load_from)
        epoch_offset = int(state.step) // train_steps_per_epoch
        log_for_0(f"Resuming from checkpoint {config.load_from}, epoch_offset: {epoch_offset}")

    state = R(state)

    # compile train_step, eval_step
    loss_fn = LOSS_FUNCTIONS[config.training.loss_fn]
    rng, train_rng = jax.random.split(rng)
    p_train_step = jax.pmap(partial(train_step, loss_fn=loss_fn, init_rng=train_rng, model=model, lr_fn=lr_fn), axis_name='batch', donate_argnums=(0,))
    p_eval_step = jax.pmap(partial(eval_step, loss_fn=loss_fn, model=model), axis_name='batch')
    test_batch = input_pipeline.prepare_batch_data(next(iter(train_dl)), batch_size=device_batch_size, dataset_metdata=train_metadata)
    log_for_0("Compiling p_train_step and p_eval_step ...")
    timer = Timer()
    p_train_step = p_train_step.lower(state, test_batch).compile()
    p_eval_step = p_eval_step.lower(state, test_batch).compile()
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

    timer.reset()
    for epoch in range(epoch_offset, config.training.epochs):
        log_for_0(f"Epoch {epoch} ...")
        train_metrics = MyMetrics(reduction='avg')
        for n_batch, batch in enumerate(train_dl):
            batch = input_pipeline.prepare_batch_data(batch, batch_size=device_batch_size, dataset_metdata=train_metadata)
            state, metrics, vis = p_train_step(state, batch)
            train_metrics.update(metrics)
            step = epoch * train_steps_per_epoch + n_batch
            ep = epoch + n_batch / train_steps_per_epoch
            
            if (n_batch + 1) % config.training.log_per_step == 0:
                summary = train_metrics.compute_and_reset()
                summary = {f"train/{k}": v for k, v in summary.items()}
                summary["steps_per_second"] = (
                    config.training.log_per_step / timer.elapse_with_reset()
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
            for n_batch, batch in enumerate(eval_dl):
                batch = input_pipeline.prepare_batch_data(batch, batch_size=device_batch_size, dataset_metdata=eval_metadata)
                metrics, vis = p_eval_step(state, batch)
                eval_metrics.update(metrics)
                if epoch == -1: # means debug run
                    break
            log_for_0(f"Epoch {epoch} evaluation done. " + ", ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.compute().items()]))
            summary = eval_metrics.compute()
            summary = {f"eval/{k}": v for k, v in summary.items()}
            logger.log(step + 1, summary)
            log_for_0(f"Epoch {epoch} eval done in {timer}.")
            
            # TODO: log visualizations
            logger.log_image(step + 1, {f'data_{i}': sudoku_to_image(batch['labels'][0][i], prompt=batch['inputs'][0][i]) for i in range(4)})
            logger.log_image(step + 1, {f'completion_{i}': sudoku_to_image(vis[0][i], prompt=batch['inputs'][0][i]) for i in range(4)})
            
            if config.just_evaluate:
                return

        # save checkpoint
        if (epoch + 1) % config.training.checkpoint_interval == 0 or (epoch + 1) == config.training.epochs:
            save_checkpoint(state, workdir)