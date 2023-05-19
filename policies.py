import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import *
from utils import MLP

class MLPPolicy(nn.Module):

	output_dims: int
	hidden_layers: int
	hidden_dims: int
	mode: str = "categorical"


	def setup(self):

		if self.mode == "continuous-distribution":
			self.mlp = MLP(self.output_dims*2, self.hidden_dims, self.hidden_layers)
		else :
			self.mlp = MLP(self.output_dims, self.hidden_dims, self.hidden_layers)
		


	def __call__(self, obs, key):

		a = self.mlp(obs)
		
		if self.mode == "categorical":

			a = jax.random.categorical(key, a)
			return a

		elif self.mode == "continuous":

			return a

		elif self.mode == "continuous-distribution":
			x = self.output_dims
			mu, s = a[..., :x], a[..., x:]
			a = jax.random.normal(key, (x,)) * s + mu
			return a


if __name__ == "__main__":

	pi = MLPPolicy(1, 2, 10, mode="continuous-distribution")
	o = jnp.ones((4,4))
	k = jax.random.PRNGKey(42)
	params = pi.init(k, o, k)
	o = jnp.ones((4,))
	for i in range(100):
		k, sk = jax.random.split(k)
		a = pi.apply(params, o, sk)
		print(a, a.shape)



			
			
