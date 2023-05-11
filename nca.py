import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as random
import chex
from functools import partial

@chex.dataclass
class NcaConfig:
	channels: int
	alpha: float = 0.1
	perception_dims: int = 3
	hidden_dims: int = 64


class PerceptionNet(eqx.Module):

	config: NcaConfig
	perception: eqx.Module

	def __init__(self, config: NcaConfig, key: random.PRNGKey):

		self.config = config
		self.perception = nn.Conv3d(in_channels=config.channels, out_channels=config.channels*config.perception_dims,
			kernel_size=3, groups=config.channels, stride=1, use_bias=False, padding=1, key=key)

	def __call__(self, x):
		return self.perception(x)


class UpdateNet(eqx.Module):

	config: NcaConfig
	layers: list

	def __init__(self, config: NcaConfig, key: random.PRNGKey):
		
		self.config = config
		key1, key2, key3 = random.split(key, 3)
		self.layers = [
			nn.Conv3d(in_channels=config.channels*config.perception_dims, out_channels=config.hidden_dims, kernel_size=1, use_bias=False, key=key1),
			nn.Conv3d(in_channels=config.hidden_dims, out_channels=config.hidden_dims, kernel_size=1, use_bias=False, key=key2),
			nn.Conv3d(in_channels=64, out_channels=config.channels, kernel_size=1, use_bias=False, key=key3)
		]

	def __call__(self, x):

		for layer in self.layers[:-1]:
			x = jax.nn.relu(layer(x))
		return self.layers[-1](x)


class Nca3d(eqx.Module):

	config: NcaConfig
	perception: eqx.Module
	update: eqx.Module
	is_alive: eqx.Module

	def __init__(self, config: NcaConfig, key: random.PRNGKey):
		
		key1, key2 = random.split(key)
		self.config = config
		
		self.perception = PerceptionNet(config, key1)

		self.update = UpdateNet(config, key2)

		self.is_alive = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)

	def __call__(self, x):
		
		alive_mask = self.is_alive(x[-1:, ...]) > self.config.alpha
		p = self.perception(x)
		dx = self.update(p)
		x_ = x + dx
		alive_mask = alive_mask & (self.is_alive(x_[-1:, ...]) > self.config.alpha)
		alive_mask = alive_mask.astype(float)

		return x_ * 

def zero_nca(nca):
	"""zero out the zeights of the last layer of the NCA update network"""
	return eqx.tree_at(lambda nca: nca.update.layers[-1].weight, nca, replace_fn=lambda w: w*0.)


@eqx.filter_jit
def rollout(nca: Nca3d, x: jnp.array):
	def step(carry, x):
		x_ = nca(carry)
		return x_, x_
	xs = jnp.arange(3)
	return jax.lax.scan(step, x, xs)

@eqx.filter_jit
def rollout_final(nca, x):
	return jax.lax.fori_loop(0, 5, lambda i, x : nca(x), x)