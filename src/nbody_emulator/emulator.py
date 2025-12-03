# emulator.py

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices("cpu")[0])
import jax.numpy as jnp
import equinox as eqx
import optax

class NNEmulator(eqx.Module):
    # TODO: Define layers as attributes
    # Hint: eqx.nn.Linear, eqx.nn.MLP, or build your own
    
    def __init__(self, key):
        # TODO: Initialize layers
        pass
    
    def __call__(self, x):
        # TODO: Forward pass
        # x: (2,) array of [Q0, a] (normalized)
        # returns: (3,) array of [f_bound, sigma_v, r_h] (normalized)
        pass

def mse_loss(model, x, y):
    # TODO: Compute mean squared error
    pass

def train_step(model, opt_state, optimizer, x, y):
    # TODO: Single gradient descent step
    # Hint: use eqx.filter_grad
    pass

def train_ensemble(key, x_train, y_train, n_models=5):
    # TODO: Train multiple models with different initializations
    pass

def predict_ensemble(models, x):
    # TODO: Return mean and std across ensemble
    pass