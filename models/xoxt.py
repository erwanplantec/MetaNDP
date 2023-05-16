import jax
import jax.numpy as jnp
import chex
import flax.linen as nn
from functools import partial


@chex.dataclass
class GEM_Config:
	n_genes: int

class GEM(nn.Module):

	config: GEM_Config

	@nn.compact
	def __call__(self, X):

		O = self.param("O", partial(jax.random.uniform, minval=-1., maxval=1.), 
			(X.shape[-1], X.shape[-1]))
		return X @ O @ X.T


class S_GEM(nn.Module):

	def __call__(self, X):
		pass


if __name__ == "__main__":
	N = 100
	G = 10
	conf = GEM_Config(n_genes=G)
	gem = GEM(conf)
	X = jax.random.uniform(jax.random.PRNGKey(42), (N, G))
	params = gem.init(jax.random.PRNGKey(42), X)

	W = gem.apply(params, X)

	print(W.shape)
	print(params)