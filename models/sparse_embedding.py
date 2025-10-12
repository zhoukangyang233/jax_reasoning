from typing import Union

# import torch
# from torch import nn
# import torch.distributed as dist
# from torch.optim.optimizer import Optimizer, ParamsT

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

class CastedSparseEmbedding(nn.Module):
    cast_to: jnp.dtype
    num_embeddings: int
    embedding_dim: int
    init_std: float
    batch_size: int
    
    def setup(self):
        raise NotImplementedError('This has not been tested')
        self.weights = self.param(
            "weights",
            nn.initializers.truncated_normal(stddev=self.init_std),
        )
        self.local_weights = self.variable("buffers", "local_weights", nn.zeros)
        self.local_ids = self.variable("buffers", "local_ids", partial(nn.zeros, dtype=jnp.int32))

    def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
        if not train:
            # Test mode, no gradient
            return jnp.asarray(self.weights)[inputs].astype(self.cast_to)

        # Training mode, fill puzzle embedding from weights
        self.local_weights.value = jax.lax.stop_gradient(jnp.asarray(self.weights)[inputs])
        self.local_ids.value = jax.lax.stop_gradient(inputs)

        return self.local_weights.value.astype(self.cast_to)

# TODO{zhh}: Implement this optimizer

# class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
#     def __init__(
#         self,
#         params: ParamsT,

#         world_size: int,
#         lr: Union[float, torch.Tensor] = 1e-3,
#         weight_decay: float = 1e-2,
#     ):
#         if not 0.0 <= lr:
#             raise ValueError(f"Invalid learning rate: {lr}")
#         if not 0.0 <= weight_decay:
#             raise ValueError(f"Invalid weight_decay value: {weight_decay}")

#         defaults = dict(
#             lr=lr,
#             weight_decay=weight_decay,
#             world_size=world_size
#         )
#         super().__init__(params, defaults)

#     @torch.no_grad
#     def step(self, closure=None):  # type: ignore
#         for group in self.param_groups:
#             # Find the sparse embedding weights
#             local_weights_grad = None
#             local_ids = None
#             weights = None
            
#             assert len(group["params"]) == 3
#             for p in group["params"]:
#                 if p.requires_grad:
#                     local_weights_grad = p.grad
#                 elif p.ndim == 1:
#                     local_ids = p
#                 elif p.ndim == 2:
#                     weights = p
#                 else:
#                     assert False
                
#             assert local_weights_grad is not None
#             assert local_ids is not None
#             assert weights is not None
        
#             # Apply SignSGD
#             # Adam ≈ SignSGD if gradient is very sparse
#             _sparse_emb_signsgd_dist(
#                 local_weights_grad,
#                 local_ids,
#                 weights,
                
#                 lr=group["lr"],
#                 weight_decay=group["weight_decay"],
#                 world_size=group["world_size"]
#             )


# def _sparse_emb_signsgd_dist(
#     local_weights_grad: torch.Tensor,
#     local_ids: torch.Tensor,
#     weights: torch.Tensor,
    
#     lr: float,
#     weight_decay: float,
#     world_size: int
# ) -> None:
#     N, D = local_weights_grad.shape
    
#     # All-gather
#     all_weights_grad = local_weights_grad
#     all_ids = local_ids

#     if world_size > 1:
#         all_weights_grad = torch.empty((world_size * N, D), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
#         all_ids = torch.empty(world_size * N,               dtype=local_ids.dtype,          device=local_ids.device)
    
#         dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
#         dist.all_gather_into_tensor(all_ids,          local_ids)

#     # Unique
#     grad_ids, inv = all_ids.unique(return_inverse=True)

#     grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
#     grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

#     # SignSGD with decoupled weight decay
#     p = weights[grad_ids]

#     p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)

#     # Write updated slices back
#     weights[grad_ids] = p
