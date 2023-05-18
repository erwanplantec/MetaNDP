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

def scan_print_formatter(i, c, y):
	msg = f"	INNER LOOP #{i}"
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

class SimpleEvaluator(core.Evaluator):

	#-------------------------------------------------------------------------

	def __init__(self, config: SimpleEvaluator_Config, ndp: nn.Module):

		self.config = config
		self.es, self.es_params, self.es_fitness_shaper = self.init_es()
		super().__init__(config, ndp)

	#-------------------------------------------------------------------------

	def init_es(self):

		es = es_map[self.config.es](num_dims=self.config.n_params, 
			popsize=self.config.popsize, **self.config.es_config)
		es_params = es.default_params if self.config.es_params is None \
									  else self.config.es_params
		fitness_shaper = FitnessShaper(maximize=True)
		
		return es, es_params, fitness_shaper

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
				fitness = jnp.sum(rollout_data["rewards"], axis=-1)
				fitness_re = self.es_fitness_shaper.apply(z, fitness)

				es_state = self.es.tell(z, fitness_re, es_state, self.es_params)

				data = {
					'fitness': fitness, 
					'es_state': es_state,
					"rollout_data": rollout_data,
				}

				return [es_state, key], data

			key, init_key = jax.random.split(key)
			es_state = self.es.initialize(init_key, self.es_params)
			gens = jnp.arange(self.config.epochs)
			[es_state, _], data = jax.lax.scan(es_step, [es_state, key], gens)

			fitness = data["fitness"]
			ndp_fitness = jnp.max(fitness)

			return ndp_fitness, data

		return evaluate





