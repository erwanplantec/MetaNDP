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
	OpenES, CMA_ES, PGPE, SimpleGA, LES, RandomSearch,
	ParameterReshaper, FitnessShaper
)
from utils import MLP, sparsity, ind_sparsity
from envs import env_step_scan, bd_mountain_car

#=========================================================================
#=================================UTILS===================================
#=========================================================================

es_map = {
	"openes": OpenES,
	"cma-es": CMA_ES,
	"pgpe": PGPE,
	"simple-ga": SimpleGA,
	"les": LES,
	"random": RandomSearch
}

bd_extractors = {
	"MountainCar-v0": bd_mountain_car,
}

def sparsity_score_fn(rollout_data):
	return ind_sparsity(rollout_data["bd"])

#=========================================================================
#==================================META-NDP===============================
#=========================================================================


@chex.dataclass
class Config:

	epochs: int
	
	ndp: object

	evaluator: Evaluator

	n_params: int
	params_shaper: ParamterReshaper
	es: str = "openes"
	es_config: Collection = field(default_factory=partial(dict, popsize=64))
	es_params: Collection = None




class NDP_Trainer:
	#-------------------------------------------------------------------------
	
	def __init__(self, config: Config):
		
		self.config = config
		self.env_rollout, self.policy, self.obs_dims, self.action_dims = self.init_env()
		self.es, self.es_params, self.es_fitness_shaper = self.init_es()
		self.eval = self.config.evaluator
		self.train = self._build_trainer()

	#-------------------------------------------------------------------------

	def init_es(self):

		es = es_map[self.config.oes](num_dims=self.config.n_params, 
			**self.config.es_config)
		os_params = es.default_params if self.config.es_params is None \
									  else self.config.os_params
		fitness_shaper = FitnessShaper(maximize=True)
		
		return es, es_params, fitness_shaper

	#-------------------------------------------------------------------------
	
	def _build_trainer(self)->Callable:

		@jax.jit
		def oes_step(carry, iter):
			oes_state, key = carry
			key, ask_key, eval_key = random.split(key, 3)
			ndp_params_flat, oes_state = self.oes.ask(ask_key, oes_state, self.oes_params)
			ndp_params = self.config.params_shaper.reshape(ndp_params_flat)
			fitness = self.eval(ndp_params, eval_key)
			fitness_re = self.oes_fitness_shaper.apply(ndp_params_flat, fitness)
			oes_state = self.oes.tell(ndp_params_flat, fitness_re, oes_state, self.oes_params)

			return [oes_state, key], fitness

		def train(key: random.PRNGKey):
			
			key, init_key = random.split(key)
			oes_state = self.oes.initialize(init_key, self.oes_params)
			gens = jnp.arange(self.config.epochs)
			[oes_state, _], data = jax.lax.scan(oes_step, [oes_state, key], gens)

		return train
		
