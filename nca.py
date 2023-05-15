import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import chex
from functools import partial
from typing import *
from dataclasses import field
from evosax import ParameterReshaper

from metandp import NDP_Trainer, Config


#==================================================================================================
#==================================================================================================
#==================================================================================================

@chex.dataclass
class NCA_Config:
	channels: int
	alpha: float = 0.1
	perception_dims: int = 3
	update_features: Iterable[int] = (64, 64)
	mask: jnp.array = None

default_config = lambda : NCA_Config(channels = 8)

class PerceptionNet(nn.Module):
	config: NCA_Config

	def setup(self):
		total_features = self.config.perception_dims * self.config.channels
		self.conv = nn.Conv(features=total_features, kernel_size=(3,3,3), strides=(1,1,1), padding="SAME", 
			feature_group_count=self.config.channels, use_bias=False)

	def __call__(self, x):
		return self.conv(x)

class UpdateNetwork(nn.Module):
	config: NCA_Config

	def setup(self):
		self.layers = [
			nn.Conv(feat, kernel_size=(1,1,1), strides=(1,1,1), padding="SAME", use_bias=False)
			for feat in self.config.update_features
		]
		self.out_layer = nn.Conv(self.config.channels, kernel_size=(1,1,1), strides=(1,1,1), padding="SAME", use_bias=False)

	def __call__(self, x):
		for layer in self.layers:
			x = nn.relu(layer(x))
		return self.out_layer(x)

class NCA3D(nn.Module):

	config: NCA_Config

	def setup(self):
		self.perception_net = PerceptionNet(self.config)
		self.update_net = UpdateNetwork(self.config)
		self.mask = self.config.mask if self.config.mask is not None else 1.

	def __call__(self, x):
		
		life_mask = nn.max_pool(x[..., -1:], (3,3,3), padding="SAME") > self.config.alpha
		percept = self.perception_net(x)
		dx = self.update_net(percept)
		x = x + dx
		life_mask = life_mask & (nn.max_pool(x[..., -1:], (3,3,3), padding="SAME") > self.config.alpha)
		life_mask = life_mask.astype(float) * self.mask
		return x * life_mask

#==================================================================================================
#==================================================================================================
#==================================================================================================


class NCA_Trainer(NDP_Trainer):

	#-------------------------------------------------------------------------

	def __init__(self, config: Config):
		assert isinstance(config.ndp_config, NCA_Config)
		super().__init__(config)	

	#-------------------------------------------------------------------------

	def init_ndp(self):


		H, W, D = self.config.hidden_dims, self.config.hidden_dims, self.config.hidden_layers+1
		z_dims = C = self.config.ndp_config.channels
		#Generate mask
		mask = jnp.zeros((H,W,D,1))
		idims = self.obs_dims
		odims = self.action_dims
		xi = H//2 - idims//2
		mask = mask.at[xi:xi+idims, :, 0, :].set(1.)
		mask = mask.at[:, :, 1:-1, :].set(1.)
		xo = W//2 - odims//2
		mask = mask.at[:, xo:xo+odims, -1, :].set(1.)
		#initiate nca
		nca_config = self.config.ndp_config
		nca_config.mask = mask
		nca = NCA3D(self.config.ndp_config)
		x = jnp.zeros((H,W,D,C))
		params = nca.init(random.PRNGKey(42), x)
		params_shaper = ParameterReshaper(params)

		def init_w(z):
			"""compute initial w matrix with seed z at the center"""
			w = jnp.zeros((H, W, D, C))
			w = w.at[H//2, W//2, D//2, :].set(z.at[-1].set(1.))
			return w

		def ndp(ndp_params: Collection, z: jnp.array)->Collection:
			w = init_w(z)
			w = jax.lax.foriloop(0, self.config.iterations, 
				lambda i, x: nca.apply(ndp_params, x), w)
			mlp_params = {
				"layers_0": {"kernel": w[xi:xi+idims, :, 0, 0]},
				"out_layer": {"kernel": w[:, xo:xo+odims, -1, 0]},
				**{
					f"layers_{i}": {"kernel": w[:, :, i, 0]}
				for i in range(1, D)}
			}
			mlp_params = {"params": mlp_params}
			return mlp_params

		return ndp, params_shaper, z_dims

	#-------------------------------------------------------------------------


#==================================================================================================
#==================================================================================================
#==================================================================================================

		

if __name__ == "__main__":
	config = NCA_Config(channels=8)
	nca = NCA3D(config)
	x = jnp.zeros((32, 32, 32, config.channels))
	# params = nca.init(random.PRNGKey(42), x)
	x_ = nca.apply(params, x)

