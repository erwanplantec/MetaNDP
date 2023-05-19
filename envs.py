import jax
import jax.numpy as jnp
import gymnax as gym
import chex
import flax.linen as nn
from typing import *


def bd_mountain_car(data):
	"""bd extractor for mountain car env"""
	pos = data['states'].position
	max_x = jnp.max(pos)
	min_x = jnp.min(pos)

	return jnp.array([min_x, max_x])


bd_extractors = {
	"MountainCar-v0": bd_mountain_car
}


