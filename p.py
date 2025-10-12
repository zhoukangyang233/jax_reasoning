import jax
import jax.numpy as jnp
import flax.linen as nn

# class Test(nn.Module):
#     def setup(self):
#         self.fake = self.variable("buffers", "fake", lambda:None)
        
#     def __call__(self, x):
#         self.fake.value = (x, 'nmsl')
#         return x
    
# if __name__ == "__main__":
#     model = Test()
#     x = jnp.ones((2,3))
#     variables = model.init(jax.random.PRNGKey(0), x)
#     y, updated_vars = model.apply(variables, x, mutable=["buffers"])
#     y, updated_vars = model.apply(variables, jnp.ones((3, 3)), mutable=["buffers"])
#     print(updated_vars["buffers"]["fake"])

import flax.linen as nn
import jax, jax.numpy as jnp

class WhileLoopExample(nn.Module):
  def setup(self):
      self.state = self.variable('state', 'acc', lambda: jnp.array(0))
      self.dense = nn.Dense(2)
  def init_fn(self, x):
      return self.dense(x)
    
  def __call__(self, x):
    def cond_fn(mdl, c):
      return mdl.state.value < 10
    def body_fn(mdl, c):
    #   acc = mdl.variable('state', 'acc', lambda: jnp.array(0))
    #   acc.value += 1
      acc = mdl.state
      acc.value += 1
      y = mdl.dense(c)
      return y
    c = x
    # if self.is_mutable_collection('params'):
    #   return body_fn(self, c)
    # else:
    return nn.while_loop(cond_fn, body_fn, self, c, carry_variables='state')
k = jax.random.key(0)
x = jnp.ones((2, 2))
model = WhileLoopExample()
initial_vars = model.init(k, x, method='init_fn')
result, state = model.apply(initial_vars, x, mutable=['state'])

print(initial_vars)
print(result)
print(state)