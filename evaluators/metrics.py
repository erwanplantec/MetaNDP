import jax.numpy as jnp
import jax
from typing import *

#============================DIVERSITY==========================

def sparsity(x):
	dists = jnp.sqrt(jnp.sum(x[:, None, :] - x[None, :, :], axis=-1)**2)
	return jnp.mean(dists)

def ind_sparsity(x):
	dists = jnp.sqrt(jnp.sum(x[:, None, :] - x[None, :, :], axis=-1)**2)
	return jnp.mean(dists, axis=-1)

def build_knn_sparsity(n: int, k: int = 3):
	def knn_sparsity(x):
		dists = jnp.sqrt(jnp.sum(x[:, None, :] - x[None, :, :], axis=-1)**2)
		res = 0.
		for i in range(n):
			idxs = jnp.argsort(dists[i])
			knn = idxs[1:k+1]
			res += jnp.mean(dists[i, knn])
		return res / n
	return knn_sparsity

def knn_sparsity(x, n, k):
	dists = jnp.sqrt(jnp.sum(x[:, None, :] - x[None, :, :], axis=-1)**2)
	res = 0.
	for i in range(n):
		idxs = jnp.argsort(dists[i])
		knn = idxs[1:k+1]
		res += jnp.mean(dists[i, knn])
	return res / n

#=======================COMPOSITIONALITY==========================

def C(X, Y):
	"""measure compositionality of process f such that Y = f(X)
	X and Y are 2D arrays"""
	dists_X = jnp.sqrt(jnp.sum(X[:, None, :] - X[None, :, :], axis=-1)**2)
	dists_Y = jnp.sqrt(jnp.sum(Y[:, None, :] - Y[None, :, :], axis=-1)**2)

	dists_X = jnp.reshape(dists_X, -1)
	dists_Y = jnp.reshape(dists_Y, -1)

	return jnp.corrcoef(dists_X, dists_Y)[0, -1]