# test_ensemble.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.nbody_emulator import emulator, utils
import pickle

# Load everything
with open('outputs/ensemble_models.pkl', 'rb') as f:
    models = pickle.load(f)

input_normalizer = utils.Normalizer.load('outputs/input_normalizer.npz')
output_normalizer = utils.Normalizer.load('outputs/output_normalizer.npz')

# Load test data
train_params = np.load('outputs/data/train_params.npy')[:30]
train_outputs = np.load('outputs/data/train_outputs.npy')[:30]

# Pick a test example
test_idx = 5
x_test = train_params[test_idx]  # (2,) - [Q0, a]
y_true = train_outputs[test_idx]  # (3,) - [f_bound, sigma_v, r_h]

print(f"Test case: Q0={x_test[0]:.3f}, a={x_test[1]:.1f}")
print(f"True outputs: f_bound={y_true[0]:.3f}, sigma_v={y_true[1]:.3f}, r_h={y_true[2]:.1f}")

# Normalize input
x_norm = input_normalizer.normalize(x_test)

# Get ensemble prediction
mean_norm, std_norm = emulator.predict_ensemble(models, x_norm)

# Denormalize
mean_pred = output_normalizer.denormalize(mean_norm)
std_pred = std_norm * output_normalizer.std  # Scale uncertainty

print(f"\nEnsemble predictions:")
print(f"  f_bound: {mean_pred[0]:.3f} ± {std_pred[0]:.3f}")
print(f"  sigma_v: {mean_pred[1]:.3f} ± {std_pred[1]:.3f}")
print(f"  r_h:     {mean_pred[2]:.1f} ± {std_pred[2]:.1f}")

print(f"\nErrors:")
print(f"  f_bound: {abs(mean_pred[0] - y_true[0]):.3f}")
print(f"  sigma_v: {abs(mean_pred[1] - y_true[1]):.3f}")
print(f"  r_h:     {abs(mean_pred[2] - y_true[2]):.1f}")