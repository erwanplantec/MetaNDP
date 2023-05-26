import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
import chex
from typing import *
from dataclasses import field
from evosax import ParameterReshaper
from utils import MLP

def env_step_scan(env_step: Callable, policy: nn.Module, env_params: Collection):
	"""
	env_step : key x state x action -> obs, state, rew, done, _
	policy : key x params x obs -> action 
	"""
	def scan_step(carry, x):
		key, obs, state, policy_params = carry
		key, key_step, key_policy = jax.random.split(key, 3)
		action = policy.apply(policy_params, obs, key_policy)
		n_obs, n_state, rew, done, _ = env_step(key_step, state, action, env_params)
		
		n_carry = [key, n_obs, n_state, policy_params]
		y = {
			"states": n_state, 
			"rewards": rew,
			"obs": obs,
			"dones": done,
			"actions": action,
		}

		return n_carry, y

	return scan_step


@chex.dataclass
class Config:
	epochs: int

	env: object
	env_backend: str
	env_params: Collection
	env_steps: int

	n_params: int


class Evaluator:

	def __init__(self, config: Config, ndp: nn.Module, policy: nn.Module):

		self.config = config
		
		self.policy = policy

		self.env_rollout = self.init_env()

		self.ndp = ndp

		self.eval = self._build_eval()

	#-------------------------------------------------------------------------

	def __call__(self, key: jax.random.PRNGKey, ndp_params: Collection):
		
		return self.eval(key, ndp_params)

	#-------------------------------------------------------------------------

	def init_env(self):

		if self.config.env_backend == "gymnax":

			env, env_params = self.config.env, self.config.env_params
			obs_shape = env.observation_space(env_params).shape
			n_actions = env.action_space(env_params).n

			scan_step = env_step_scan(env.step, self.policy, env_params)
			def rollout(key, policy_params):
				key_reset, key_step = random.split(key)
				obs, state = env.reset(key_reset, env_params)
				xs = jnp.arange(self.config.env_steps)
				# get env rollout
				_, scan_out = jax.lax.scan(
					scan_step, [key_step, obs, state, policy_params], xs
				)
				
				return scan_out

		elif self.config.env_backend == "brax":

			env = self.config.env

			def scan_step(carry, x):
				key, state, policy_params = carry
				key, key_policy = jax.random.split(key)
				action = self.policy.apply(policy_params, state.obs, key_policy)
				new_state = self.env.step(state, action)

				y = {
					"states": new_state,
					"actions": action
				}

				return [key, new_state, policy_params], y
			
			def rollout(key, policy_params):
				key_reset, key_step = random.split(key)
				state = env.reset(key_reset)
				xs = jnp.arange(self.config.env_steps)
				# get env rollout
				_, scan_out = jax.lax.scan(
					scan_step, [key_step, state, policy_params], xs
				)
				
				return scan_out


		return rollout


	#-------------------------------------------------------------------------


	def _build_eval(self):

		raise NotImplementederror()