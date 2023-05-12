import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import chex
from dataclasses import field
from functools import partial
from typing import *
from evosax import (
	OpenES, CMA_ES, PGPE, SimpleGA, LES, RandomSearch,
	ParameterReshaper, FitnessShaper
)
from utils import MLP

es_map = {
	"openes": OpenES,
	"cma-es": CMA_ES,
	"pgpe": PGPE,
	"simple-ga": SimpleGA,
	"les": LES,
	"random": RandomSearch
}

bd_extractors = {
	"ant": AntBDExtractor
}


@chex.dataclass
class Config:
	outer_epochs: int
	inner_epochs: int

	input_dims: int
	output_dims: int
	hidden_dims: int
	hidden_layers: int

	oes: str = "openes"
	oes_config: Collection = field(default_factory=dict)
	oes_params: Collection = None

	ies: str = "random"
	ies_config: Collection = field(default_factory=dict)
	ies_params: Collection = None

	env_name: str = "ant"
	env_config: Collection = field(default_factory=dict)


class NDP_Trainer:
	#-------------------------------------------------------------------------
	
	def __init__(self, config: Config):
		
		self.config = config
		self.mlp = self.init_mlp()
		self.ndp, self.params_shaper, self.z_dims = self.init_ndp()
		self.oes, self.oes_params, self.oes_fitness_shaper = self.init_oes()
		self.ies, self.ies_params, self.ies_fitness_shaper = self.init_ies()

		self.eval = self._build_eval()

	#-------------------------------------------------------------------------

	def init_mlp(self):

		mlp = MLP(self.config.output_dims, self.config.hidden_dims, 
			self.config.hidden_layers)
		_ = mlp.init(ramdom.PRNGKey(42), jnp.zeros((self.config.input_dims,)))
		return mlp

	#-------------------------------------------------------------------------

	def init_ndp(self):

		raise NotImplemetedError()

	#-------------------------------------------------------------------------

	def init_oes(self):

		oes = es_map[self.config.oes](num_dims=self.params_shaper.total_params, 
			**config.oes_config)
		oes_params = oes.default_params if self.config.oes_params is None else self.config.oes_params
		fitness_shaper = FitnessShaper(maximize=True)
		
		return oes, oes_params, fitness_shaper

	#-------------------------------------------------------------------------

	def init_ies(self):

		ies = es_map[self.config.ies](num_dims=self.z_dims, **self.config.ies_config)
		ies_params = ies.default_params if self.config.ies_params is None else self.config.ies_params
		fitness_shaper = FitnessShaper(maximize=True)

		return ies, ies_params, fitness_shaper

	#-------------------------------------------------------------------------
	
	def _build_eval(self):
		
		# Init Environment
		
		@jax.vmap		
		def evaluate(ndp_params: Collection, key:random.PRNGKey)->jnp.array:

			@jax.jit
			def ies_step(carry, iter):
				ies_state, key = carry
				key, ask_key, eval_key = random.split(key, 3)
				z, ies_state = self.ies.ask(ask_key, ies_state, self.ies_params)
				mlp_params = self.ndp(ndp_params, z)
				fitness = env_eval(mlp_params)
				fitness_re = FitnessShaper.apply(fitness)
				ies_state = self.ies.tell(z, fitness_re, ies_state, self.ies_params)

				return [ies_state, key], fitness


			key, init_key = random.split(key)
			ies_state = self.ies.initialize(init_key, self.ies_params)
			gens = jnp.arange(self.config.inner_epochs)
			[ies_state, _], data = jax.lax.scan(ies_step, [ies_state, key], gens)


		return evaluate

	#-------------------------------------------------------------------------
	
	def _build_trainer(self):

		@jax.jit
		def oes_step(carry, iter):
			oes_state, key = carry
			key, ask_key, eval_key = random.split(key, 3)
			ndp_params_flat, oes_state = self.oes.ask(ask_key, oes_state, self.oes_params)
			ndp_params = self.params_shaper.reshape(ndp_params_flat)
			fitness = self.eval(ndp_params, random.split(eval_key, self.oes.popsize))
			fitness_re = self.oes_fitness_shaper.apply(fitness)
			oes_state = self.oes.tell(ndp_params_flat, fitness_re, oes_state, self.oes_params)

			return [oes_state, key], fitness

		def train(key: random.PRNGKey):
			
			key, init_key = random.split(key)
			oes_state = self.oes.initialize(init_key, self.oes_params)
			gens = jnp.arange(self.config.outer_epochs)
			[oes_state, _], data = jax.lax.scan(oes_step, [oes_state, key], gens)

		return train
		
