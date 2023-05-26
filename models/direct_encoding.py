import jax
import jax.numpy as jnp
import flax.linen as nn

class DirectEncoding(nn.Module):

	@nn.compact
	def __call__(self, x):
		return x