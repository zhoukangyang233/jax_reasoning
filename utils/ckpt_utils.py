from absl import logging
import jax
from flax.training import checkpoints
from .logging_util import log_for_0
import os
import dataclasses
import flax
from flax.jax_utils import unreplicate as U

try:
    import gcsfs
except Exception as e:
    raise RuntimeError('please install gcsfs') from e

FS = gcsfs.GCSFileSystem()

def convert_to_gs(path):
    assert os.path.isabs(path), f'ckpt path {path} is not absolute.'
    # assert path.startswith('/')
    
    # /kmh-nfs-us-mount/staging/siri/unknown/launch_20250910_025630_git_5351ee2c/logs/log2_20250910_030359_39a8088a
    subpaths = path.strip('/').split('/')
    assert subpaths[0] in ['kmh-nfs-ssd-us-mount', 'kmh-nfs-us-mount'], f'cannot handle checkpoint path {path}'

    pref = 'kmh-gcp-us-central2'
    out = '/' + '/'.join(subpaths[3:])
    out = f'gs://{pref}/qiao_zhicheng_hanhong_files' + out
    return out

def exist_general(path):
    if path.startswith('gs://'):
        return FS.exists(path)
    return os.path.exists(path)

def is_checkpoint(path):
    print('check', path)
    if not exist_general(path):
        return False
    if not os.path.basename(path).startswith('checkpoint_'):
        path = checkpoints.latest_checkpoint(path)
        return path is not None and is_checkpoint(path)
    return True

def restore_checkpoint(state, workdir, allow_nockpt=False, ignore_keys=()):
    retain_states = {k.name: getattr(state, k.name) for k in dataclasses.fields(state) if k.name in ignore_keys}

    def _restore():
        if workdir:
            for try_dir in [
                workdir,
            ]:
                if is_checkpoint(try_dir):
                    return checkpoints.restore_checkpoint(try_dir, state)
                
            for try_dir in [
                convert_to_gs(workdir),
            ]:
                if is_checkpoint(try_dir):
                    return checkpoints.restore_checkpoint(try_dir, state)
        if allow_nockpt:
            log_for_0(f'[WARNING] checkpoint does not exist on {workdir}, start from scratch')
            return state
        
        raise RuntimeError(f'checkpoint does not exist on {workdir}')
    
    return _restore().replace(**retain_states)


def save_checkpoint(state, workdir, ignore_keys=()):
    state = jax.device_get(U(state))
    step = int(state.step)
    log_for_0("Saving checkpoint step %d. Type: %s", step, type(state))
    checkpoints.save_checkpoint_multiprocess(convert_to_gs(workdir), state, step, keep=2)


def restore_pretrained(state, path, config):
    raise NotImplementedError
    pretrained = checkpoints.restore_checkpoint(path, target=None)
    pretrained_params = pretrained["ema_params"]
    log_for_0(f"pretrained model: {pretrained_params.keys()}")
    assert all(key.startswith("blocks_") for key in pretrained_params.keys())
    teacher_nblocks = len(pretrained_params)
    student_nblocks = len(state.params)
    map_fn = get_map_fn(config.load_pretrain_method, teacher_nblocks, student_nblocks)
    for i in range(student_nblocks):
        student_block = state.params[f"blocks_{i}"]
        # example_element = jax.tree_leaves(student_block)[0].reshape(-1)
        # logging.info(f'example_element at layer {i}: {example_element[0]}')
        teacher_block = pretrained_params[f"blocks_{map_fn(i)}"]
        # example_element = jax.tree_leaves(teacher_block)[0].reshape(-1)
        # logging.info(f'example_element from teacher at (student) layer {i}: {example_element[0]}')
        assert jax.tree_structure(student_block) == jax.tree_structure(teacher_block)
        state.params[f"blocks_{i}"] = teacher_block
        logging.info(f"Restored block {i} from teacher block {map_fn(i)}")

    # assert jax.tree_structure(state.params["Encoder"]) == jax.tree_structure(
    #     pretrained["ema_params"]["Encoder"]
    # )
    # assert jax.tree_structure(state.params["Decoder"]) == jax.tree_structure(
    #     pretrained["ema_params"]["Decoder"]
    # )


    # just in case
    # assert jax.tree_structure(state.batch_stats) == \
    #   jax.tree_structure(pretrained['batch_stats'])
    # state = state.replace(batch_stats=pretrained['batch_stats'])
    
    # new_block = state.params["blocks_0"]
    # example_element = jax.tree_leaves(new_block)[0].reshape(-1)
    # logging.info(f'final example_element at layer 0: {example_element[0]}')

    log_for_0("Loaded.")
    # ema
    # state = state.replace(ema_params=jax.tree_map(lambda x: jax.numpy.array(x), state.params))
    return state