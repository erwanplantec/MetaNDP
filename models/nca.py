import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import chex
from functools import partial
from typing import *
from dataclasses import field


#==================================================================================================
#==================================================================================================
#==================================================================================================

@chex.dataclass
class NCA_Config:
	channels: int
	alpha: float = 0.1
	perception_dims: int = 3
	update_features: Iterable[int] = (64, 64)

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

	def __call__(self, x):
		
		life_mask = nn.max_pool(x[..., -1:], (3,3,3), padding="SAME") > self.config.alpha
		percept = self.perception_net(x)
		dx = self.update_net(percept)
		x = x + dx
		life_mask = life_mask & (nn.max_pool(x[..., -1:], (3,3,3), padding="SAME") > self.config.alpha)
		life_mask = life_mask.astype(float)
		return x * life_mask

#==================================================================================================
#==================================================================================================
#==================================================================================================

@chex.dataclass
class HyperNCA_Config(NCA_Config):
	iterations: int = 10
	action_dims: int = 2
	obs_dims: int = 2
	hidden_dims: int = 64
	hidden_layers: int = 2


class HyperNCA(nn.Module):

	config: HyperNCA_Config

	#-------------------------------------------------------------------------

	@property
	def z_dims(self):
		return self.config.channels	

	#-------------------------------------------------------------------------

	def setup(self):

		H, W, D = self.config.hidden_dims, self.config.hidden_dims, self.config.hidden_layers+1
		C = self.config.channels
		#Generate mask
		mask = jnp.zeros((H,W,D,1))
		idims = self.config.obs_dims
		odims = self.config.action_dims
		xi = H//2 - idims//2
		mask = mask.at[xi:xi+idims, :, 0, :].set(1.)
		mask = mask.at[:, :, 1:-1, :].set(1.)
		xo = W//2 - odims//2
		self.mask = mask.at[:, xo:xo+odims, -1, :].set(1.)
		self.nca = NCA3D(self.config)

	#-------------------------------------------------------------------------

	def __call__(self, z):

		H, W, D = self.config.hidden_dims, self.config.hidden_dims, self.config.hidden_layers+1
		idims = self.config.obs_dims
		odims = self.config.action_dims
		xi = H//2 - idims//2
		xo = W//2 - odims//2
		C = self.config.channels
		
		w = jnp.zeros((H, W, D, C))
		w = w.at[H//2, W//2, D//2, :].set(z.at[-1].set(1.))

		for i in range(self.config.iterations):
			w = self.nca(w) * self.mask
		
		mlp_params = {
			"layers_0": {"kernel": w[xi:xi+idims, :, 0, 0]},
			"out_layer": {"kernel": w[:, xo:xo+odims, -1, 0]},
			**{
				f"layers_{i}": {"kernel": w[:, :, i, 0]}
			for i in range(1, D-1)}
		}
		mlp_params = {"params": mlp_params}
		
		return mlp_params



#==================================================================================================
#==================================================================================================
#==================================================================================================

		

if __name__ == "__main__":
	config = NCA_NDP_Config(channels=8)
	ndp = NCA_NDP(config)
	params = ndp.init(jax.random.PRNGKey(42), jnp.ones((8,)))
	print(params["params"].keys())

