import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.experimental import host_callback

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


def scan_print(rate=50, formatter=None):

	if formatter is None:
		formatter = lambda i, c, y: f"iteration #{i}"

	def tap_func(args, transforms):
		print(formatter(*args))

	def _print(i, c, y):

		_ = jax.lax.cond(
			i % rate == 0,
			lambda _: host_callback.id_tap(tap_func, [i, c, y], result=i),
			lambda _: i,
			operand = None
		)

	def func_wrapper(func):

		def wrapped_func(carry, x):
			if type(x) is tuple:
				it, *_ = x
			else:
				it = x  
			n_carry, y = func(carry, x)
			_print(it, carry, y)
			return n_carry, y

		return wrapped_func

	return func_wrapper




if __name__ == "__main__":

	@scan_print(rate=1)
	def f(c, x):
		return c, x

	c, ys = jax.lax.scan(f, 1., jnp.arange(10))





