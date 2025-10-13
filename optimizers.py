import jax
import jax.numpy as jnp
import optax
import math
from functools import partial
from typing import NamedTuple

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):  
    progress = (current_step - num_warmup_steps) / jnp.maximum(1, num_training_steps - num_warmup_steps)
    return base_lr * jnp.where(
        current_step < num_warmup_steps,
        current_step / jnp.maximum(1, num_warmup_steps),
        min_ratio + jnp.maximum(0.0, (1 - min_ratio) * 0.5 * (1.0 + jnp.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    )

def create_lr_fn(training_config, total_steps):
    schedule_type = training_config.lr_schedule.lower()
    assert schedule_type == 'cos', f'Only "cos" lr_schedule is supported for now, got {schedule_type}'
    
    return lambda step: cosine_schedule_with_warmup_lr_lambda(
        step,
        base_lr=training_config.learning_rate,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=total_steps,
        min_ratio=training_config.lr_min_ratio,
    )
    # return partial(
    #     cosine_schedule_with_warmup_lr_lambda,
    #     base_lr=training_config.learning_rate,
    #     num_warmup_steps=training_config.warmup_steps,
    #     num_training_steps=total_steps,
    #     min_ratio=training_config.lr_min_ratio,
    # )
    

def build_optimizer(training_config, total_steps):
    opt_type = training_config.optimizer.lower()
    lr_fn = create_lr_fn(training_config, total_steps)
    
    if opt_type == 'adamw':
        return optax.adamw(learning_rate=lr_fn, weight_decay=training_config.weight_decay, b1=0.9, b2=0.95), lr_fn
    elif opt_type == 'adam_atan2':
        return optax.chain(
            adam_atan2(b1=0.9, b2=0.95),
            optax.add_decayed_weights(training_config.weight_decay),
            optax.scale_by_learning_rate(lr_fn),
        ), lr_fn
    else:
        raise ValueError(f"Unknown optimizer {opt_type}.")

class AdamAtan2State(NamedTuple):
    count: int
    mu: jnp.ndarray
    nu: jnp.ndarray

def adam_atan2(b1, b2, regen_reg_rate=0., decoupled_wd=False, cautious_factor=1., a=1.27, b=1.) -> optax.GradientTransformation:
    """credit: https://github.com/lucidrains/adam-atan2-pytorch/blob/main/adam_atan2_pytorch/adam_atan2.py"""
    
    # mu_dtype = utils.canonicalize_dtype(mu_dtype)
    assert regen_reg_rate == 0., "Not implemented"
    assert not decoupled_wd, "Not implemented"
    assert cautious_factor == 1., "Not implemented"

    def init_fn(params):
        mu = optax.tree.zeros_like(params)  # First moment
        nu = optax.tree.zeros_like(params)  # Second moment
        return AdamAtan2State(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = optax.tree.update_moment(updates, state.mu, b1, 1)
        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = state.count + 1
        if False:
            # nerstov = True, TODO{zhh}: this should not be used.
            mu_hat = jax.tree.map(
                lambda m, g: b1 * m + (1 - b1) * g,
                optax.tree.bias_correction(mu, b1, count_inc + 1),
                optax.tree.bias_correction(updates, b1, count_inc),
            )
        else:
            mu_hat = optax.tree.bias_correction(mu, b1, count_inc)
            # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
            # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
            # unclear why. Other Nadam implementations also omit the extra b2 factor.
            nu_hat = optax.tree.bias_correction(nu, b2, count_inc)
        
        updates = jax.tree.map(
            # lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
            lambda m, v: None if m is None else a * jnp.arctan2(m, b * jnp.sqrt(v)),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        return updates, AdamAtan2State(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)