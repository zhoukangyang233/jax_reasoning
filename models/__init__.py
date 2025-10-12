# credit: all code are copied from https://github.com/sapientinc/HRM

# NOTE{zhh}: we write all models here

from functools import partial
from .hrm.hrm_act_v1 import HRM_ACT_V1

HRM_default = partial(HRM_ACT_V1, H_cycles=2, L_cycles=2, H_layers=4, L_layers=4, halt_max_steps=16, halt_exploration_prob=0.1)