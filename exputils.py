import jax
import jax.numpy as jnp
import chex
import numpy as np

@chex.dataclass
class ExperimentalConfig:

	env_name: str
	env_backend: str 

	policy_type:str # mlp
	



def make_expe(config: ExperimentalConfig):
	pass