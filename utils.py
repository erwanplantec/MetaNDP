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

def f(x):
	def g(y):
		return y*x
	return g(x)

if __name__ == "__main__":
	print(f(4))
	print(f(5))
	print(f(6))