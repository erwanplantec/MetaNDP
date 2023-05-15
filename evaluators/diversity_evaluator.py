from evaluators.core import Evaluator, Config
from envs import env_step_scan, bd_mountain_car

import jax
import jax.numpy as jnp
import chex
from evosax import (
	OpenES, CMA_ES, PGPE, SimpleGA, LES, RandomSearch,
	ParameterReshaper, FitnessShaper
)

@chex.dataclass
class DiversityConfig(Config):
	bd_extractor: Callable
	popsize: int = 100


class DiversityEvaluator(Evaluator):

	def __init__(self, config: DiversityConfig):

		super().__init__(config)


	#-------------------------------------------------------------------------

	def _build_eval(self):

		bd_extractor = self.config.bd_extractor

		@partial(jax.vmap, in_axes=(0, None)) #vmap eval over ndp parans		
		def evaluate(ndp_params: Collection, key:random.PRNGKey)->jnp.array:

			@jax.jit
			def es_step(carry, iter):
				
				 = carry
				key, ask_key, eval_key = random.split(key, 3)
				
				z = jax.random.uniform(ask_key, (self.config.popsize, self.config.n_params),
					minval=-1., maxval=1.)
				policy_params = jax.vmap(self.ndp.apply, in_axes=(None, 0))(ndp_params, z)
				
				rollout_data = jax.vmap(self.env_rollout, in_axes=(None, 0))(eval_key, policy_params)
				bd = jax.vmap(bd_extractor)(rollout_data) #(popsize, bd_dims)
				rollout_data["bd"] = bd
				
				fitness = sparsity(bd)

				return key, fitness


			key, init_key = random.split(key)
			gens = jnp.arange(self.config.epochs)
			_, gens_fitness = jax.lax.scan(es_step, key, gens)

			# compute final fitness of ndp params
			fitness = jnp.mean(gens_fitness) # average sparsity accross generations

			return fitness

		return evaluate