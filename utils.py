import flax.linen as nn
import jax
import jax.numpy as jnp

class MLP(nn.Module):
	output_dims: int
	hidden_dims: int
	hidden_layers: int

	def setup(self):
		self.layers = [nn.Dense(self.hidden_dims, use_bias=False) for _ in range(self.hidden_layers)]
		self.out_layer = nn.Dense(self.output_dims, use_bias=False)

	def __call__(self, x):
		for layer in self.layers:
			x = nn.relu(layer(x))
		return self.out_layer(x)


def sparsity(x):
	dists = jnp.sqrt(jnp.sum(x[:, None, :] - x[None, :, :], axis=-1)**2)
	return jnp.mean(dists)

def ind_sparsity(x):
	dists = jnp.sqrt(jnp.sum(x[:, None, :] - x[None, :, :], axis=-1)**2)
	return jnp.mean(dists, axis=-1)


if __name__ == "__main__":
	key = jax.random.PRNGKey(44)
	x = jax.random.normal(key, (100, 2)) * 10 
	print(sparsity(x))