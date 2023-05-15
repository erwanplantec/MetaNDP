import jax
import jax.numpy as jnp
import gymnax as gym
import chex


def bd_mountain_car(data):
	"""bd extractor for mountain car env"""
	pos = data['states'].position
	max_x = jnp.max(pos)
	min_x = jnp.min(pos)

	return jnp.array([min_x, max_x])

def env_step_scan(env_step, policy, env_params):
	"""
	env_step : key x state x action -> obs, state, rew, done, _
	policy : key x params x obs -> action 
	"""
	def scan_step(carry, x):
		key, obs, state, policy_params = carry
		key, key_step, key_policy = jax.random.split(key, 3)
		action = policy(key_policy, policy_params, obs)
		n_obs, n_state, rew, done, _ = env_step(key_step, state, action, env_params)
		
		n_carry = [key, n_obs, n_state, policy_params]
		y = {
			"states": n_state, 
			"rewards": rew,
			"obs": obs,
			"dones": done,
			"actions": action
		}

		return n_carry, y

	return scan_step



if __name__ == "__main__":
	env, env_params = gym.make("MountainCar-v0")
	key = jax.random.PRNGKey(42)
	obs, state = env.reset(key, env_params)
	pi = lambda key, p, o: env.action_space().sample(key)
	scan_step = env_step_scan(env.step, pi, env_params)
	xs = jnp.arange(200)
	_, scan_out = jax.lax.scan(
		scan_step, [key, obs, state, None], xs
	)
	bd = bd_mountain_car(scan_out)
	print(bd)

