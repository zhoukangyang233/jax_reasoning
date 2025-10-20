from typing import Tuple, List, Dict, Optional
# from dataclasses import dataclass
from flax import struct
from flax.core import FrozenDict
import math

# import torch
# import torch.nn.functional as F
# from torch import nn

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

# from models.common import trunc_normal_init_
trunc_normal_init_ = jax.nn.initializers.truncated_normal(stddev=1.0)
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

SG = jax.lax.stop_gradient

# @dataclass
@struct.dataclass
class InnerCarry:
    z_H: jnp.ndarray
    z_L: jnp.ndarray
    

@struct.dataclass
class Carry:
    inner_carry: InnerCarry
    
    steps: jnp.ndarray
    halted: jnp.ndarray
    finish_count: jnp.ndarray

    current_data: FrozenDict[str, jnp.ndarray]

class HRMBlock(nn.Module):
    hidden_size: int
    num_heads: int
    expansion: float
    rms_norm_eps: float
    
    @nn.compact
    def __call__(self, cos_sin: CosSin, hidden_states: jnp.ndarray):
        assert self.hidden_size % self.num_heads == 0, f"hidden size must be divisible by number of heads, got {self.hidden_size} and {self.num_heads}"
        hidden_states = rms_norm(
            hidden_states + Attention(
                hidden_size=self.hidden_size,
                head_dim=self.hidden_size // self.num_heads,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_heads,
                causal=False
            )(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.rms_norm_eps
        )
        hidden_states = rms_norm(
            hidden_states + SwiGLU(
                hidden_size=self.hidden_size,
                expansion=self.expansion
            )(hidden_states),
            variance_epsilon=self.rms_norm_eps
        )
        return hidden_states


class HRMReasoningModule(nn.Module):
    layers: tuple[HRMBlock]

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, input_injection: jnp.ndarray, cos_sin: CosSin):
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)

        return hidden_states

def i_chain(*funcs, common_args=2):
    def chained_func(*args):
        *common, x = args
        assert len(common) == common_args, f'Expected {common_args} common args, got {len(common)}'
        for f in funcs:
            x = f(*common, x)
        return x
    return chained_func

def nn_for_i_loop(lower, upper, body_fun, module, init_val):
    # lower: int
    # upper: int
    # body_fun: (module, i, val) -> val
    # init_val: any

    # This is not runable due to bad versions
    # def fn(m, carry, x):
    #     return carry, body_fun(m, x, carry)
    # i_arr = jnp.arange(lower, upper)
    # scan_fn = nn.scan(fn, variable_broadcast='params', split_rngs={'params': False}, in_axes=0, out_axes=0, length=upper - lower)
    # _, out = scan_fn(module, init_val, i_arr)
    # return jax.tree_map(lambda x: x[-1], out)
    
    # ZHH: use while loop to implement for-i-loop
    def cond_fun(module, val):
        return val[0] < upper
    def body_fun_wrap(module, val):
        i, val = val
        return i + 1, body_fun(module, i, val)
    _, out = nn.while_loop(cond_fun, body_fun_wrap, module, (lower, init_val))#, broadcast_variables='params')
    return out

class HRM_ACTV1(nn.Module):
    # dataset-related configs
    batch_size: int = None
    seq_len: int = None
    vocab_size: int = None
    num_puzzle_identifiers: int = None

    # algo configs
    H_cycles: int = None
    L_cycles: int = None
    halt_max_steps: int = None
    halt_exploration_prob: float = None

    # model configs
    H_layers: int = None
    L_layers: int = None
    
    hidden_size: int = 512
    puzzle_emb_ndim: int = 512
    num_heads: int = 8
    pos_encodings: str = 'rope' # 'rope' or 'learned'
    expansion: float = 4.0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "float32"
    # forward_dtype: str = "bfloat16"

    def setup(self):
        self._forward_dtype = getattr(jnp, self.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.vocab_size, self.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.hidden_size, self.vocab_size, bias=False)
        
        # NOTE: special init for q_head
        self.q_head       = CastedLinear(self.hidden_size, 2, bias=True, initialization='-5')

        self.puzzle_emb_len = -(self.puzzle_emb_ndim // -self.hidden_size)  # ceil div
        assert self.puzzle_emb_len == 1, f'Not supported: {self.puzzle_emb_len=}'
        if self.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            # self.puzzle_emb = CastedSparseEmbedding(self.num_puzzle_identifiers, self.puzzle_emb_ndim,
            #                                         batch_size=self.batch_size, init_std=0, cast_to=self.forward_dtype)
            
            # FIXME{zhh}: we currently only use naive nn.Embed. TODO: implement CastedSparseEmbedding and its optimizer
            self.puzzle_emb = nn.Embed(self.num_puzzle_identifiers, self.puzzle_emb_ndim, embedding_init=nn.initializers.zeros)

        # LM Blocks
        if self.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.hidden_size // self.num_heads,
                                              max_position_embeddings=self.seq_len + self.puzzle_emb_len,
                                              base=self.rope_theta)
        elif self.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.seq_len + self.puzzle_emb_len, self.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HRMReasoningModule(layers=[HRMBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, expansion=self.expansion, rms_norm_eps=self.rms_norm_eps) for _ in range(self.H_layers)])
        self.L_level = HRMReasoningModule(layers=[HRMBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, expansion=self.expansion, rms_norm_eps=self.rms_norm_eps) for _ in range(self.L_layers)])

        # Initial states
        # These states are not trainble, and also will NOT be updated
        self.H_init = self.variable('const', 'H_init', lambda: trunc_normal_init_(self.make_rng('const'), (self.hidden_size,)))
        self.L_init = self.variable('const', 'L_init', lambda: trunc_normal_init_(self.make_rng('const'), (self.hidden_size,)))

        self.carry = self.variable('buffer', 'carry', lambda: None)

    def _input_embeddings(self, input: jnp.ndarray, puzzle_identifiers: jnp.ndarray):
        # Token embedding
        embedding = self.embed_tokens(input.astype(jnp.int32))

        # Puzzle embeddings
        if self.puzzle_emb_ndim > 0:
            # raise NotImplementedError
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.hidden_size - puzzle_embedding.shape[-1]
            assert pad_count == 0, f'Not supported'
            # if pad_count > 0:
            #     puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = jnp.concatenate((puzzle_embedding.reshape((-1, self.puzzle_emb_len, self.hidden_size)), embedding), axis=-2)
        else:
            raise NotImplementedError('puzzle_emb_ndim = 0 is very, very bad. q_head implementation is very wrong.')

        # Position embeddings
        if self.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def _init_carry(self, batch_size):
        self.carry.value = Carry(
            inner_carry=InnerCarry(
                z_H=jnp.empty((batch_size, self.seq_len + self.puzzle_emb_len, self.hidden_size), dtype=self.forward_dtype),
                z_L=jnp.empty((batch_size, self.seq_len + self.puzzle_emb_len, self.hidden_size), dtype=self.forward_dtype),
            ),
            steps=jnp.zeros((batch_size, ), dtype=jnp.int32),
            halted=jnp.ones((batch_size, ), dtype=jnp.bool_),  # Default to halted
            finish_count=jnp.zeros((batch_size, ), dtype=jnp.int32),
            current_data={
                "inputs": jnp.zeros((batch_size, self.seq_len), dtype=jnp.int32),
                "labels": jnp.zeros((batch_size, self.seq_len), dtype=jnp.int32),
                "puzzle_identifiers": jnp.zeros((batch_size,), dtype=jnp.int32)
            }
        )

    def _update_carry(self):
        """
        "Update Carry" is reset the inner carry (z_H, z_L) when the outer carry indicates a reset (halted).
        """
        reset_flag = self.carry.value.halted
        carry = self.carry.value
        self.carry.value = self.carry.value.replace(inner_carry=InnerCarry(
            z_H=jnp.where(reset_flag.reshape((-1, 1, 1)), self.H_init.value, carry.inner_carry.z_H),
            z_L=jnp.where(reset_flag.reshape((-1, 1, 1)), self.L_init.value, carry.inner_carry.z_L),
        ))
        # self.carry.value.inner_carry = InnerCarry(
        #     z_H=jnp.where(reset_flag.reshape((-1, 1, 1)), self.H_init.value, carry.inner_carry.z_H),
        #     z_L=jnp.where(reset_flag.reshape((-1, 1, 1)), self.L_init.value, carry.inner_carry.z_L),
        # )

    def _inner_step(self, batch: FrozenDict[str, jnp.ndarray], update_inner_carry: bool):
        cos_sin = self.rotary_emb() if self.pos_encodings == "rope" else None
        carry = self.carry.value.inner_carry

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations (NOTE{zhh}: main algorithm is here)
        z_H, z_L = carry.z_H, carry.z_L

        # for _H_step in range(self.H_cycles):
        #     for _L_step in range(self.L_cycles):
        #         if not ((_H_step == self.H_cycles - 1) and (_L_step == self.L_cycles - 1)):
        #             z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)

        #     if not (_H_step == self.H_cycles - 1):
        #         z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)
        
        # change to for-i-loop?
        z_H, z_L = nn_for_i_loop(
            0, self.H_cycles * self.L_cycles,
            i_chain(
                # val: (z_H, z_L)
                lambda m, i, val: ( # update L
                    val[0],
                    nn.cond(
                        i == self.H_cycles * self.L_cycles - 1,
                        lambda mo, z_L: mo.L_level(z_L, val[0] + input_embeddings, cos_sin=cos_sin),
                        lambda mo, z_L: z_L,
                        m,
                        val[1]
                    )
                ),
                lambda m, i, val: (
                    nn.cond(
                        ((i + 1) % self.L_cycles == 0) & (i != self.H_cycles * self.L_cycles - 1),
                        lambda mo, z_H: mo.H_level(z_H, val[1], cos_sin=cos_sin),
                        lambda mo, z_H: z_H,
                        m,
                        val[0]
                    ),
                    val[1]
                ),
                common_args=2
            ),
            self,
            (z_H, z_L)
        )

        z_H = SG(z_H)
        z_L = SG(z_L)

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        # LM Outputs
        new_carry = InnerCarry(z_H=SG(z_H), z_L=SG(z_L))
        
        if update_inner_carry:
            self.carry.value = self.carry.value.replace(inner_carry=new_carry) # overwrite inner carry
        
        # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).astype(jnp.float32)
        
        return output, (q_logits[..., 0], q_logits[..., 1])

    def __call__(self, batch: FrozenDict[str, jnp.ndarray], rng, train: bool):
        batch_size = batch["inputs"].shape[0]
        
        # need to update carry each call
        self._update_carry()
        carry = self.carry.value
        
        new_steps = jnp.where(carry.halted, 0, carry.steps)
        new_current_data = {k: jnp.where(carry.halted.reshape((batch_size, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        
        # Forward inner step
        logits, (q_halt_logits, q_continue_logits) = self._inner_step(new_current_data, update_inner_carry=True)
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.halt_max_steps
        halted = is_last_step

        # TODO{zhh}: understand this ACT implementation
        if train and (self.halt_max_steps > 1):
            # Halt signal
            # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
            halted = halted | (q_halt_logits > q_continue_logits)

            # Exploration
            rng, _ = jax.random.split(rng)
            rng1, rng2 = jax.random.split(_)
            min_halt_steps = (jax.random.uniform(rng1, q_halt_logits.shape) < self.halt_exploration_prob) * jax.random.randint(rng2, new_steps.shape, 2, self.halt_max_steps + 1)

            halted = halted & (new_steps >= min_halt_steps)

            # Compute target Q
            # NOTE: No replay buffer and target networks for computing target Q-value.
            # As batch_size is large, there're many parallel envs.
            # Similar concept as PQN https://arxiv.org/abs/2407.04811
            
            # NOTE{zhh}: since this is only for RL, no update of carry
            next_q_halt_logits, next_q_continue_logits = self._inner_step(new_current_data, update_inner_carry=False)[-1]

            outputs["target_q_continue"] = jax.nn.sigmoid(jnp.where(is_last_step, next_q_halt_logits, jnp.maximum(next_q_halt_logits, next_q_continue_logits)))

        self.carry.value = self.carry.value.replace(
            steps=new_steps,
            halted=halted,
            # Update finish_count: increment when a sample becomes halted this step
            finish_count=self.carry.value.finish_count + (halted & (~self.carry.value.halted)).astype(jnp.int32),
            current_data=new_current_data
        )
        return outputs
    
    def init_fn(self, batch: FrozenDict[str, jnp.ndarray]):
        self._init_carry(batch["inputs"].shape[0])
        self._update_carry() # Use this, otherwise H_init will not be inited
        
        # ret = self._inner_step(batch, update_inner_carry=False)
        # should not loop when init
        cos_sin = self.rotary_emb() if self.pos_encodings == "rope" else None
        emb = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        r = self.H_level(self.carry.value.inner_carry.z_H, self.carry.value.inner_carry.z_L, cos_sin=cos_sin)
        r = self.L_level(self.carry.value.inner_carry.z_L, r + emb, cos_sin=cos_sin)
        ret = self.lm_head(r), self.q_head(r[:, 0])
        
        # hack: let init have no carry
        self._init_carry(self.batch_size)
        return ret
        

    def inference(self, batch: FrozenDict[str, jnp.ndarray]):
        # during inference, rewrite the carry to null
        self._init_carry(batch["inputs"].shape[0])
        # call once for shape
        output = self(batch, None, train=False)

        def cond_fun(module, output):
            return ~module.carry.value.halted.all()
        
        def body_fun(module, output):
            return module(batch, None, train=False)

        return nn.while_loop(cond_fun, body_fun, self, output, carry_variables='buffer')

HRM_debug = partial(HRM_ACTV1, H_cycles=2, L_cycles=2, H_layers=1, L_layers=1, halt_max_steps=2, halt_exploration_prob=0.1, hidden_size=4, puzzle_emb_ndim=4, num_heads=2, expansion=1.0)
HRM_default = partial(HRM_ACTV1, H_cycles=2, L_cycles=2, H_layers=4, L_layers=4, halt_max_steps=16, halt_exploration_prob=0.1)

if __name__ == "__main__":
    # TODO{zhh}: perform model test here
    inputs = jnp.ones((7, 81), dtype=jnp.int32)
    labels = jnp.ones((7, 81), dtype=jnp.int32)
    identifiers = jnp.zeros((7,), dtype=jnp.int32)
    batch = {"inputs": inputs, "labels": labels, "puzzle_identifiers": identifiers}
    model = HRM_debug(batch_size=7, seq_len=81, vocab_size=11, num_puzzle_identifiers=1)
    
    # init: simulate the correct init (which use 2 bs)
    variables = model.init({'params': jax.random.PRNGKey(0), 'const': jax.random.PRNGKey(0)}, jax.tree_map(lambda a: a[:2], batch), method=model.init_fn,)# mutable=['buffer', 'const'])
    print('init passed')
    
    print('variables:', variables.keys())
    print('buffer:', variables['buffer'])
    output, new_variables = model.apply(variables, batch, rng=jax.random.PRNGKey(0), train=True, mutable=['buffer'])
    print('shape test passed')
    
    output, new_variables = model.apply(variables, batch, mutable=['buffer'], method=model.inference)
    print('inference test passed')

    print('const vals:', variables['const']['H_init'], variables['const']['L_init'])