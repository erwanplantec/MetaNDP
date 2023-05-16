from evaluators import core
from envs import env_step_scan, bd_mountain_car

import jax
import jax.numpy as jnp
import jax.random as random
import chex
from typing import *
from functools import partial
from evosax import (
	OpenES, CMA_ES, PGPE, SimpleGA, LES, RandomSearch,
	ParameterReshaper, FitnessShaper
)
from utils import scan_print
from gymnax.visualize import Visualizer

def scan_print_formatter(i, c, y):
	bds = y["bd"]
	msg = f"	INNER LOOP #{i}"
	return msg


def sparsity(x):
	dists = jnp.sqrt(jnp.sum(x[:, None, :] - x[None, :, :], axis=-1)**2)
	return jnp.mean(dists)

def ind_sparsity(x):
	dists = jnp.sqrt(jnp.sum(x[:, None, :] - x[None, :, :], axis=-1)**2)
	return jnp.mean(dists, axis=-1)


@chex.dataclass
class DiversityEvaluator_Config(core.Config):
	bd_extractor: Callable
	popsize: int = 100


class DiversityEvaluator(core.Evaluator):

	#-------------------------------------------------------------------------

	def _build_eval(self):

		bd_extractor = self.config.bd_extractor
	
		def evaluate(ndp_params: Collection, key:random.PRNGKey)->jnp.array:

			@scan_print(rate=1, formatter=scan_print_formatter)
			def es_step(carry, iter):
				
				key = carry
				key, ask_key, eval_key = random.split(key, 3)
				
				z = jax.random.uniform(ask_key, (self.config.popsize, self.config.n_params),
					minval=-1., maxval=1.) # get random seeds
				policy_params, *_ = jax.vmap(self.ndp.apply, in_axes=(None, 0))(ndp_params, z)
				
				eval_keys = jax.random.split(eval_key, self.config.popsize)
				rollout_data = jax.vmap(self.env_rollout, in_axes=(0, 0))(eval_keys, policy_params)
				bd = jax.vmap(bd_extractor)(rollout_data) #(popsize, bd_dims)
				rollout_data["bd"] = bd
				
				spars = sparsity(bd)

				data = {
					'sparsity': spars, 
					"bd": bd,
					"rollout_data": rollout_data,
					"policy_params": policy_params
				}

				return key, data


			key, init_key = random.split(key)
			gens = jnp.arange(self.config.epochs)
			_, data = jax.lax.scan(es_step, key, gens)

			# compute final fitness of ndp params
			fitness = jnp.mean(data["sparsity"]) # average sparsity accross generations

			return fitness, data

		return evaluate

	#-------------------------------------------------------------------------

	def test(self, key: random.PRNGKey, ndp_params: Collection, n_samples: int=5,
		render: bool=False, save_file: str="anim"):
		key, ask_key, rollout_key = random.split(key, 3)
		z = jax.random.uniform(ask_key, (n_samples, self.config.n_params),
			minval=-1., maxval=1.) # get random seeds
		policy_params, *_ = jax.vmap(self.ndp.apply, in_axes=(None, 0))(ndp_params, z)
		rollout_keys = jax.random.split(rollout_key, n_samples)
		rollout_data = jax.vmap(self.env_rollout, in_axes=(0, 0))(rollout_keys, policy_params)
		bds = jax.vmap(self.config.bd_extractor)(rollout_data)
		spars = sparsity(bd)

		if render:
			files = []

			states = rollout_data['states']
			states = [jax.tree_map(lambda x: x[i], states) for i in range(n_samples)]
			states = [
				[jax.tree_map(lambda x: x[i], stacked_seq) for i in range(self.config.env_steps)]
			for stacked_seq in states]

			for ind, seq in enumerate(states):
				vis = Visualizer(self.config.env, self.config.env_params, seq)
				file = f"{save_file}_{ind}.gif"
				files.append(file)
				vis.animate(file)
			return spars, rollout_data, files


		return spars, rollout_data

if __name__ == "__main__":
	x = jax.random.uniform(jax.random.PRNGKey(62), (10, 2))
	sp = sparsity(x)
	print(sp)