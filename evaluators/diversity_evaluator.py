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


if __name__ == "__main__":
	x = jax.random.uniform(jax.random.PRNGKey(62), (10, 2))
	sp = sparsity(x)
	print(sp)