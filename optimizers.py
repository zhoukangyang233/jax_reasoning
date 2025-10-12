import jax
import jax.numpy as jnp
import optax
import math
from functools import partial

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))

def create_lr_fn(training_config, total_steps):
    schedule_type = training_config.lr_schedule.lower()
    assert schedule_type == 'cos', f'Only "cos" lr_schedule is supported for now, got {schedule_type}'
    
    return partial(
        cosine_schedule_with_warmup_lr_lambda,
        base_lr=training_config.learning_rate,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=total_steps,
        min_ratio=training_config.lr_min_ratio,
    )
    

def build_optimizer(training_config, total_steps):
    opt_type = training_config.optimizer.lower()
    lr_fn = create_lr_fn(training_config, total_steps)
    
    if opt_type == 'adamw':
        return optax.adamw(learning_rate=lr_fn, weight_decay=training_config.weight_decay, b1=0.9, b2=0.95), lr_fn