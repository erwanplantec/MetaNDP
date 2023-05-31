import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import *
from dataclasses import field
from evosax import (
	OpenES, CMA_ES, PGPE, SimpleGA, LES, RandomSearch, DES,
	ParameterReshaper, FitnessShaper
)

from evaluators import core
from utils import scan_print

def reward_score(data):
	return jnp.sum(data["rewards"], axis=-1)

def scan_print_formatter(i, c, y):
	max_fit = jnp.max(y["fitness"])
	msg = f"	inner gen {i} : max fit = {max_fit}"
	return msg

es_map = {
	"openes": OpenES,
	"cma-es": CMA_ES,
	"pgpe": PGPE,
	"simple-ga": SimpleGA,
	"les": LES,
	"des": DES,
	"random": RandomSearch
}

@chex.dataclass
class SimpleEvaluator_Config(core.Config):
	es: str = "openes"
	popsize: int = 128
	es_config: Collection = field(default_factory=dict)
	es_params: Collection = None
	score_fn: Callable = lambda data: data["cum_rewards"]#fitness of a latent representation
	ndp_score_fn: Callable = lambda data: jnp.max(data["fitness"])

class SimpleEvaluator(core.Evaluator):

	#-------------------------------------------------------------------------

	def __init__(self, config: SimpleEvaluator_Config, ndp: nn.Module, policy: nn.Module):

		self.config = config
		self.es, self.es_params, self.es_fitness_shaper = self.init_es()
		super().__init__(config, ndp, policy)

	#-------------------------------------------------------------------------

	def init_es(self):

		es = es_map[self.config.es](num_dims=self.config.n_params, 
			popsize=self.config.popsize, **self.config.es_config)
		es_params = es.default_params if self.config.es_params is None \
									  else self.config.es_params
		fitness_shaper = FitnessShaper(maximize=True)
		
		return es, es_params, fitness_shaper

	#-------------------------------------------------------------------------

	def init_env(self):

		if self.config.env_backend == "gymnax":

			env, env_params = self.config.env, self.config.env_params

			def scan_step(carry, x):

				key, obs, state, policy_params, cum_rewards, valid_mask = carry
				key, key_step, key_policy = jax.random.split(key, 3)
				
				action = self.policy.apply(policy_params, obs, key_policy)
				
				n_obs, n_state, rew, done, _ = env.step(key_step, state, action, env_params)
				
				new_cum_rewards = cum_rewards + rew * valid_mask
				new_valid_mask = valid_mask * (1-done)
				
				n_carry = [key, n_obs, n_state, policy_params, new_cum_rewards, new_valid_mask]
				y = {
					"rewards": rew,
					"dones": done,
				}

				return n_carry, y

			def rollout(key, policy_params):
				key_reset, key_step = jax.random.split(key)
				obs, state = env.reset(key_reset, env_params)
				xs = jnp.arange(self.config.env_steps)
				# get env rollout
				init_carry = [
					key_step,
					obs,
					state,
					policy_params,
					0.0,
					1.0
				]
				data, scan_out = jax.lax.scan(
					scan_step, init_carry, xs
				)
				scan_out["cum_rewards"] = data[-2]
				
				return scan_out

		elif self.config.env_backend == "brax":

			raise NotImplementederror("Simple Evaluator not implemented for brax envs")


		return rollout

	#-------------------------------------------------------------------------

	def _build_eval(self):

		def evaluate(ndp_params: Collection, key: jax.random.PRNGKey):
			
			@scan_print(rate=10, formatter=scan_print_formatter)
			def es_step(carry, iter):

				es_state, key = carry
				key, ask_key, eval_key = jax.random.split(key, 3)

				z, es_state = self.es.ask(ask_key, es_state, self.es_params) 
				policy_params = jax.vmap(self.ndp.apply, in_axes=(None, 0))(ndp_params, z)
				
				eval_keys = jax.random.split(eval_key, self.config.popsize)
				rollout_data = jax.vmap(self.env_rollout, in_axes=(0, 0))(eval_keys, policy_params)
				fitness = self.config.score_fn(rollout_data)
				fitness_re = self.es_fitness_shaper.apply(z, fitness)

				es_state = self.es.tell(z, fitness_re, es_state, self.es_params)

				data = {
					'fitness': fitness, 
					'es_state': es_state,
				}

				return [es_state, key], data

			key, init_key = jax.random.split(key)
			es_state = self.es.initialize(init_key, self.es_params)
			gens = jnp.arange(self.config.epochs)
			[es_state, _], data = jax.lax.scan(es_step, [es_state, key], gens)

			ndp_fitness = self.config.ndp_score_fn(data)

			return ndp_fitness, data

		return evaluate





