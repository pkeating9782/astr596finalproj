# data.py

from scipy.stats import qmc
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

def generate_training_data(n_train=80, n_test=20, seed=42):
    """Generate LHS samples for N-body emulator."""
    l_bounds = [0.5, 50.0]   # Q0_min, a_min
    u_bounds = [1.5, 200.0]  # Q0_max, a_max
    
    # Training set
    sampler_train = qmc.LatinHypercube(d=2, seed=seed)
    train_unit = sampler_train.random(n=n_train)
    train_params = qmc.scale(train_unit, l_bounds, u_bounds)
    
    # Test set
    sampler_test = qmc.LatinHypercube(d=2, seed=seed + 1000)
    test_unit = sampler_test.random(n=n_test)
    test_params = qmc.scale(test_unit, l_bounds, u_bounds)
    
    return train_params, test_params

