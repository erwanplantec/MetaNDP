from evaluators import core
from envs import bd_mountain_car
from evaluators.metrics import sparsity, knn_sparsity

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
	msg=""
	return msg

@chex.dataclass
class DiversityEvaluator_Config(core.Config):
	bd_extractor: Callable
	popsize: int = 100

	score_fn: str = "sparsity"


class DiversityEvaluator(core.Evaluator):

	#-------------------------------------------------------------------------

	def _build_eval(self):

		bd_extractor = self.config.bd_extractor

		score_fn_map = {
			'sparsity': sparsity,
			'neg_sparsity': lambda x: -sparsity(x),
			'knn_sparsity': partial(knn_sparsity, n=self.config.popsize, k=5),
			'neg_knn_sparsity': lambda x: -knn_sparsity(x, self.config.popsize, 5)
		}
		score_fn = score_fn_map.get(self.config.score_fn, sparsity)

	
		def evaluate(ndp_params: Collection, key:random.PRNGKey)->jnp.array:

			def es_step(carry, iter):
				
				key = carry
				key, ask_key, eval_key = random.split(key, 3)
				
				z = jax.random.uniform(ask_key, (self.config.popsize, self.config.n_params),
					minval=-1., maxval=1.) # get random seeds
				policy_params = jax.vmap(self.ndp.apply, in_axes=(None, 0))(ndp_params, z)
				
				eval_keys = jax.random.split(eval_key, self.config.popsize)
				rollout_data = jax.vmap(self.env_rollout, in_axes=(0, 0))(eval_keys, policy_params)
				bd = jax.vmap(bd_extractor)(rollout_data) #(popsize, bd_dims)
				rollout_data["bd"] = bd
				
				score = score_fn(bd)

				data = {
					'score': score, 
					"bd": bd,
					"z": z,
					"rollout_data": rollout_data,
				}

				return key, data


			key, init_key = random.split(key)
			gens = jnp.arange(self.config.epochs)
			_, data = jax.lax.scan(es_step, key, gens)

			# compute final fitness of ndp params
			fitness = jnp.mean(data["score"]) # average sparsity accross generations

			return fitness, data

		return evaluate

	#-------------------------------------------------------------------------

	def test(self, key: random.PRNGKey, ndp_params: Collection, n_samples: int=5,
		render: bool=False, save_file: str="anim"):
		key, ask_key, rollout_key = random.split(key, 3)
		z = jax.random.uniform(ask_key, (n_samples, self.config.n_params),
			minval=-1., maxval=1.) # get random seeds
		policy_params = jax.vmap(self.ndp.apply, in_axes=(None, 0))(ndp_params, z)
		rollout_keys = jax.random.split(rollout_key, n_samples)
		rollout_data = jax.vmap(self.env_rollout, in_axes=(0, 0))(rollout_keys, policy_params)
		bds = jax.vmap(self.config.bd_extractor)(rollout_data)
		spars = sparsity(bds)

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