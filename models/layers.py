# import torch
# from torch import nn
# import torch.nn.functional as F

import jax
import jax.numpy as jnp
import flax.linen as nn

# try:
#     from flash_attn_interface import flash_attn_func  # type: ignore[import]
# except ImportError:
#     # Fallback to FlashAttention 2
#     from flash_attn import flash_attn_func  # type: ignore[import]

# from models.common import trunc_normal_init_
trunc_normal_init_ = nn.initializers.truncated_normal

CosSin = tuple[jnp.ndarray, jnp.ndarray]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: jnp.ndarray):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
    assert cos.ndim == sin.ndim == 2
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.astype(cos.dtype)
    k = k.astype(cos.dtype)
    q_embed = (q * cos[:, None, :]) + (rotate_half(q) * sin[:, None, :])
    k_embed = (k * cos[:, None, :]) + (rotate_half(k) * sin[:, None, :])

    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


class CastedLinear(nn.Module):
    in_features: int
    out_features: int
    bias: bool
    initialization: str = 'default' # 'default' or '-5'
    
    @nn.compact
    def __call__(self, input):
        assert self.initialization in ['default', '-5'], f"Unknown initialization {self.initialization}"
        return nn.Dense(
            self.out_features,
            use_bias=self.bias,
            kernel_init=trunc_normal_init_(stddev=1.0 / (self.in_features ** 0.5)) if self.initialization == 'default' else nn.initializers.zeros,
            bias_init=nn.initializers.zeros if self.initialization == 'default' else nn.initializers.constant(-5.0)
        )(input)

class CastedEmbedding(nn.Module):
    num_embeddings: int
    embedding_dim: int
    init_std: float
    cast_to: jnp.dtype

    @nn.compact
    def __call__(self, input: jnp.ndarray) -> jnp.ndarray:
        embedding_weight = self.param(
            'embedding_weight',
            trunc_normal_init_(stddev=self.init_std),
            (self.num_embeddings, self.embedding_dim)
        )
        return embedding_weight.astype(self.cast_to)[input]

class RotaryEmbedding(nn.Module):
    dim: int
    max_position_embeddings: int
    base: int
    
    def __call__(self) -> CosSin:
        # RoPE
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.einsum('i,j->ij', t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.cos(emb), jnp.sin(emb)

class Attention(nn.Module):
    hidden_size: int
    head_dim: int
    num_heads: int
    num_key_value_heads: int
    causal: bool = False
    
    def setup(self):
        self.output_size = self.head_dim * self.num_heads

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)
        
        assert self.causal is False, 'Unsupported'
        
    def __call__(self, cos_sin: CosSin, hidden_states: jnp.ndarray):
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        attn_output = nn.attention.dot_product_attention(query, key, value)
        return self.o_proj(attn_output.reshape(batch_size, seq_len, self.output_size))


class SwiGLU(nn.Module):
    hidden_size: int
    expansion: float

    def setup(self):
        inter = _find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(self.hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, self.hidden_size, bias=False)

    def __call__(self, x):
        gate, up = jnp.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)

def rms_norm(hidden_states: jnp.ndarray, variance_epsilon: float) -> jnp.ndarray:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(jnp.float32)

    variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * jax.lax.rsqrt(variance + variance_epsilon)
    return hidden_states.astype(input_dtype)