from evaluators import core
from evaluators.metrics import knn_sparsity, C
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
	msg = f"	INNER LOOP #{i}"
	return msg


@chex.dataclass
class CompoDivEvaluator_Config(core.Config):
	bd_extractor: Callable
	popsize: int = 100

class CompoDivEvaluator(core.Evaluator):

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
				policy_params = jax.vmap(self.ndp.apply, in_axes=(None, 0))(ndp_params, z)
				
				eval_keys = jax.random.split(eval_key, self.config.popsize)
				rollout_data = jax.vmap(self.env_rollout, in_axes=(0, 0))(eval_keys, policy_params)
				bd = jax.vmap(bd_extractor)(rollout_data) #(popsize, bd_dims)
				rollout_data["bd"] = bd
				
				compo = C(z, bd) #compute compositionality coefficient
				div = knn_sparsity(bd, self.config.popsize, 5)
				score = compo + 10*div

				data = {
					'score': score,
					'compo': compo,
					'div': div, 
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

	def test(self, key: random.PRNGKey, ndp_params: Collection, n_samples: int=5):
		key, ask_key, rollout_key = random.split(key, 3)
		z = jax.random.uniform(ask_key, (n_samples, self.config.n_params),
			minval=-1., maxval=1.) # get random seeds
		policy_params = jax.vmap(self.ndp.apply, in_axes=(None, 0))(ndp_params, z)
		rollout_keys = jax.random.split(rollout_key, n_samples)
		rollout_data = jax.vmap(self.env_rollout, in_axes=(0, 0))(rollout_keys, policy_params)
		bd = jax.vmap(self.config.bd_extractor)(rollout_data)
		compo = C(z, bd) #compute compositionality coefficient
		div = knn_sparsity(bd, self.config.popsize, 5)
		score = compo + 10*div

		data = {
			"env_data": rollout_data,
			'compo': compo,
			'div': div, 		
			"z": z,
			"bd": bd
		}


		return score, data

if __name__ == '__main__':
	key = jax.random.PRNGKey(42)
	key1, key2 = jax.random.split(key)
	X = jax.random.uniform(key1, (20, 10))
	Y = jax.random.uniform(key2, (20, 3))

	print(C(X, Y))