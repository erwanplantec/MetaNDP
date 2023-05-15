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
	
	ndp_config: Collection

	outer_epochs: int
	inner_epochs: int

	hidden_dims: int
	hidden_layers: int

	oes: str = "openes"
	oes_config: Collection = field(default_factory=partial(dict, popsize=64))
	oes_params: Collection = None

	ies: str = "random"
	ies_config: Collection = field(default_factory=partial(dict, popsize=16))
	ies_params: Collection = None

	env_name: str = "MountainCar-v0"
	bd_extractor: Callable = bd_mountain_car
	action_space: str = "discrete"
	env_config: Collection = field(default_factory=dict)
	env_steps_per_epidode: int = 200



class NDP_Trainer:
	#-------------------------------------------------------------------------
	
	def __init__(self, config: Config):
		
		self.config = config
		self.env_rollout, self.policy, self.obs_dims, self.action_dims = self.init_env()
		self.ndp, self.params_shaper, self.z_dims = self.init_ndp()
		self.oes, self.oes_params, self.oes_fitness_shaper = self.init_oes()
		self.ies, self.ies_params, self.ies_fitness_shaper = self.init_ies()

		self.eval = self._build_eval()

	#-------------------------------------------------------------------------

	def init_ndp(self):

		raise NotImplemetedError()

	#-------------------------------------------------------------------------

	def init_oes(self):

		oes = es_map[self.config.oes](num_dims=self.params_shaper.total_params, 
			**self.config.oes_config)
		oes_params = oes.default_params if self.config.oes_params is None \
										else self.config.oes_params
		fitness_shaper = FitnessShaper(maximize=True)
		
		return oes, oes_params, fitness_shaper

	#-------------------------------------------------------------------------

	def init_ies(self):

		ies = es_map[self.config.ies](num_dims=self.z_dims, **self.config.ies_config)
		ies_params = ies.default_params if self.config.ies_params is None \
										else self.config.ies_params
		fitness_shaper = FitnessShaper(maximize=True)

		return ies, ies_params, fitness_shaper

	#-------------------------------------------------------------------------

	def init_env(self):


		env_name = self.config.env_name
		env, env_params = gym.make(env_name)
		obs_shape = env.observation_space(env_params).shape
		n_actions = env.action_space(env_params).n

		mlp = MLP(n_actions, self.config.hidden_dims, self.config.hidden_layers)
		_ = mlp.init(random.PRNGKey(42), jnp.zeros(obs_shape))
			
		def policy(key, params, obs):
			logits = mlp.apply(params, action)
			return random.categorical(key, logits)

		scan_step = env_step_scan(env.step, policy, env_params)
		def rollout(self, key, policy_params):
			key_reset, key_step = random.split(key)
			xs = jnp.arange(self.config.env_steps_per_epidode)
			# get env rollout
			_, scan_out = jax,lax.scan(
				scan_step, [key_step, obs, state, policy_params], xs
			)
			
			return scan_out

		return rollout, policy, obs_shape[0], n_actions


	#-------------------------------------------------------------------------
	
	def _build_eval(self)->Callable:
		"""Build evaluation function for ndps"""
		bd_extractor = self.config.bd_extractor

		@partial(jax.vmap, in_axes=(0, None)) #vmap eval over ndp parans		
		def evaluate(ndp_params: Collection, key:random.PRNGKey)->jnp.array:

			@jax.jit
			def ies_step(carry, iter):
				
				ies_state, key = carry
				key, ask_key, eval_key = random.split(key, 3)
				
				z, ies_state = self.ies.ask(ask_key, ies_state, self.ies_params)
				policy_params = self.ndp(ndp_params, z)
				
				rollout_data = self.env_rollout(eval_key, policy_params)
				bd = bd_extractor(rollout_data) #(popsize, bd_dims)
				rollout_data["bd"] = bd
				
				fitness = sparsity_score_fn(rollout_data)
				fitness_re = FitnessShaper.apply(fitness)
				ies_state = self.ies.tell(z, fitness_re, ies_state, self.ies_params)

				data = {
					"fitness": fitness, 
					"bd": bd, 
					"best_bd": bd[jnp.argmax(fitness)],
					"ies_state": ies_state
				}

				return [ies_state, key], data


			key, init_key = random.split(key)
			ies_state = self.ies.initialize(init_key, self.ies_params)
			gens = jnp.arange(self.config.inner_epochs)
			[ies_state, _], data = jax.lax.scan(ies_step, [ies_state, key], gens)

			# compute final fitness of ndp params
			fitnesss = data["fitness"] #(gens, pop)
			fitness = jnp.mean(fitnesss) # average sparsity accross generations

			return fitness

		return evaluate

	#-------------------------------------------------------------------------
	
	def _build_trainer(self)->Callable:

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

		return 
		
