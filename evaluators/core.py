import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
import chex
from typing import *
from dataclasses import field
from evosax import ParameterReshaper
from utils import MLP
from envs import env_step_scan


@chex.dataclass
class Config:
	epochs: int

	env: object
	env_params: Collection
	env_steps: int

	mlp_hidden_dims: int
	mlp_hidden_layers: int

	n_params: int


class Evaluator:

	def __init__(self, config: Config, ndp: nn.Module):

		self.config = config
		
		self.env_rollout, self.policy, self.obs_dims, self.action_dims = self.init_env()

		self.ndp = ndp

		self.eval = self._build_eval()

	#-------------------------------------------------------------------------

	def __call__(self, key: jax.random.PRNGKey, ndp_params: Collection):
		
		return self.eval(key, ndp_params)

	#-------------------------------------------------------------------------

	def init_env(self):


		env, env_params = self.config.env, self.config.env_params
		obs_shape = env.observation_space(env_params).shape
		n_actions = env.action_space(env_params).n

		mlp = MLP(n_actions, self.config.mlp_hidden_dims, self.config.mlp_hidden_layers)
		_ = mlp.init(random.PRNGKey(42), jnp.zeros(obs_shape))
			
		def policy(key, params, obs):
			logits = mlp.apply(params, obs)
			return logits

		scan_step = env_step_scan(env.step, policy, env_params)
		def rollout(key, policy_params):
			key_reset, key_step = random.split(key)
			obs, state = env.reset(key_reset, env_params)
			xs = jnp.arange(self.config.env_steps)
			# get env rollout
			_, scan_out = jax.lax.scan(
				scan_step, [key_step, obs, state, policy_params], xs
			)
			
			return scan_out

		return rollout, policy, obs_shape[0], n_actions


	#-------------------------------------------------------------------------


	def _build_eval(self):

		raise NotImplementederror()