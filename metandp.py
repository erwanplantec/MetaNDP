import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import chex
import gymnax as gym
from dataclasses import field
from functools import partial
from typing import *
from evosax import (
	OpenES, CMA_ES, PGPE, SimpleGA, LES, RandomSearch, DES,
	ParameterReshaper, FitnessShaper
)
from evaluators.core import Evaluator
from utils import scan_print

#=========================================================================
#=================================UTILS===================================
#=========================================================================

es_map = {
	"openes": OpenES,
	"cma-es": CMA_ES,
	"pgpe": PGPE,
	"simple-ga": SimpleGA,
	"les": LES,
	"des": DES,
	"random": RandomSearch
}

def scan_print_formatter(i, c, y):
	fit = y["fitness"]
	es_state = y["es_state"]
	avg_fit = jnp.mean(fit)
	top_fit = jnp.max(fit)
	best_fit = es_state.best_fitness

	return f"OUTER LOOP #{i} : avg = {avg_fit}, top = {top_fit}, best = {best_fit}"

#=========================================================================
#==================================META-NDP===============================
#=========================================================================


@chex.dataclass
class Config:

	epochs: int

	n_params: int
	params_shaper: ParameterReshaper
	
	es: str = "openes"
	popsize: int = 128
	es_config: Collection = field(default_factory=dict)
	es_params: Collection = None


class NDP_Trainer:
	#-------------------------------------------------------------------------
	
	def __init__(self, config: Config, ndp: nn.Module, evaluator: Evaluator):
		
		self.config = config
		self.es, self.es_params, self.es_fitness_shaper = self.init_es()
		self.eval = evaluator
		self.train = self._build_trainer()

	#-------------------------------------------------------------------------

	def init_es(self):

		es = es_map[self.config.es](num_dims=self.config.n_params, 
			popsize=self.config.popsize, **self.config.es_config)
		es_params = es.default_params if self.config.es_params is None \
									  else self.config.es_params
		fitness_shaper = FitnessShaper(maximize=True)
		
		return es, es_params, fitness_shaper

	#-------------------------------------------------------------------------
	
	def _build_trainer(self)->Callable:

		@jax.jit
		@scan_print(rate=1, formatter=scan_print_formatter)
		def es_step(carry, iter):
			
			es_state, key = carry
			key, ask_key, eval_key = random.split(key, 3)
			
			ndp_params_flat, es_state = self.es.ask(ask_key, es_state, self.es_params)
			ndp_params = self.config.params_shaper.reshape(ndp_params_flat)
			
			eval_key = jax.random.split(eval_key, self.config.popsize)
			fitness, eval_data = jax.jit(jax.vmap(self.eval, in_axes=(0, 0)))(ndp_params, eval_key)
			fitness_re = self.es_fitness_shaper.apply(ndp_params_flat, fitness)
			
			es_state = self.es.tell(ndp_params_flat, fitness_re, es_state, self.es_params)

			data = {
				"fitness": fitness, 
				"es_state": es_state, 
				"ndp_params": ndp_params,
				"eval_data": eval_data
			}

			return [es_state, key], data

		def train(key: random.PRNGKey):
			
			key, init_key = random.split(key)
			es_state = self.es.initialize(init_key, self.es_params)
			gens = jnp.arange(self.config.epochs)
			[es_state, _], data = jax.lax.scan(es_step, [es_state, key], gens)

			return es_state, data

		return train
		
