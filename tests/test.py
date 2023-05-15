import jax
import jax.numpy as jnp
from nca import NCA_Config, NCA_Trainer, default_config
from metandp import Config

config = Config(
    ndp_config = default_config(), 
    outer_epochs = 10,
    inner_epochs = 2,
    hidden_dims = 64, hidden_layers = 2,
)

trainer = NCA_Trainer(config)

key = jax.random.PRNGKey(42)
trainer.train(key)