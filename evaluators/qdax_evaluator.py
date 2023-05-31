import jax
import jax.numpy as jnp
from evaluators import core
import flax.linen as nn
from typing import *

from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax import environments
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter
from qdax.core.emitters.cma_rnd_emitter import CMARndEmitter
from qdax.core.emitters.cma_improvement_emitter import CMAImprovementEmitter
from qdax.core.emitters.cma_pool_emitter import CMAPoolEmitter

@chex.dataclass
class QdaxEvaluator_Config(core.Config):
	bd_extractor: Callable
	min_bd: float
	max_bd: float

	init_size: int = 64
	grid_shape: Tuple[int] = (64, 64)

class QdaxEvaluator(core.Evaluator):

	#-------------------------------------------------------------------------

	def __init__(self, config: QdaxEvaluator_Config, ndp: nn.Module, policy: nn.Module):

		self.config = config

		self.emitter, self.metrics_fn, self.centroids = self.init_es()

		super().__init__(config, ndp, policy)

	#-------------------------------------------------------------------------

	def init_es(self):

		centroids = compute_euclidean_centroids(
			self.config.grid_shape,
			min_bd,
			max_bd
		)

		emitter_kwargs = {
			"batch_size": self.config.popsize,
			"genotype_dim": self.config.n_params,
			"centroids": centroids,
			"sigma_g": sigma_g,
			"min_count": 1,
			"max_count": None,
		}

		if self.config.emitter_type == "opt":
			emitter = CMAOptimizingEmitter(**emitter_kwargs)
		elif self.config.emitter_type == "imp":
	    	emitter = CMAImprovementEmitter(**emitter_kwargs)
		elif self.config.emitter_type == "rnd":
	    	emitter = CMARndEmitter(**emitter_kwargs)
		else:
	    	raise Exception("Invalid emitter type")

		emitter = CMAPoolEmitter(
		    num_states=pool_size,
		    emitter=emitter
		)

		metrics_fn = None #TODO

		return emitter, metrics_fn, centroids

	#-------------------------------------------------------------------------

	def _build_eval(self):
		
		def evaluate(ndp_params: Collection, key: jax.random.PRNGKey):

			# Create the scoring function
			def scoring_fn(z, key): #TODO
				policy_params = jax.vmap(self.ndp.apply)(ndp_params, z)
				roll_data = self.env_rollout(key, policy_params)
				data = {}
				return fitness, descr, data, key
			
			# Initialize map elites with scoring fn
			map_elites = MAPElites(
				scoring_function = scoring_fn,
				emitter = self.emitter,
				metrics_function = self.metrics_fn
			)

			# Create the map elites step function (for scan)
			def me_scan_update(carry, x):
				repertoire, emitter_state, key = carry
				repertoire, emitter_state, metrics, key = map_elites.update(
						repertoire, emitter_state, key
					)
				return [repertoire, emitter_state, key], metrics

			# Initialize map elites population
			key, key_init, key_es_init = jax.random.split(key, 3)
			z_init = jax.randon.uniform(
				key_init,
				shape=(self.config.init_size, self.config.n_params),
				minval=-1.,
				maxval=1.
			)

			# Initialize map elites
			repertoire, emitter_state, key = map_elites.init(z_init, centroids, key_es_init)

			# Run QD search
			(repertoire, emitter_state, key), metrics = jax.lax.scan(
				me_scan_update,
				[repertoire, emitter_state, key],
				jnp.arange(self.config.epochs)
			)
			return metrics["qd_score"][-1]

		return evaluate






