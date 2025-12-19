# train_emulator.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.nbody_emulator import emulator, utils

# Load pilot data
train_params = np.load('outputs/data/train_params.npy')[:30]
train_outputs = np.load('outputs/data/train_outputs.npy')[:30]

# Normalize
input_normalizer = utils.Normalizer()
output_normalizer = utils.Normalizer()

input_normalizer.fit(train_params)
output_normalizer.fit(train_outputs)

x_train_norm = input_normalizer.normalize(train_params)
y_train_norm = output_normalizer.normalize(train_outputs)

print("Training data shapes:")
print(f"  Inputs: {x_train_norm.shape}")
print(f"  Outputs: {y_train_norm.shape}")
print()

# TRAIN ENSEMBLE (5 models)
key = jax.random.PRNGKey(42)
n_models = 5

models, all_losses = emulator.train_ensemble(
    key=key,
    x_train=x_train_norm,
    y_train=y_train_norm,
    n_models=n_models,
    learning_rate=1e-2,
    num_epochs=1200,
    print_every=100
)

print(f"ENSEMBLE TRAINING COMPLETE")

# PLOT ALL LOSS CURVES
plt.figure(figsize=(10, 6))
for i, losses in enumerate(all_losses):
    plt.plot(losses, alpha=0.7, label=f'Model {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE on normalized data)')
plt.title(f'Training Loss - Ensemble of {n_models} Models')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/figures/ensemble_training_loss.png', dpi=150, bbox_inches='tight')
plt.show()

# SAVE TRAINED ENSEMBLE
import pickle

# Save models
with open('outputs/ensemble_models.pkl', 'wb') as f:
    pickle.dump(models, f)

# Save normalizers (CRITICAL for inference!)
input_normalizer.save('outputs/input_normalizer.npz')
output_normalizer.save('outputs/output_normalizer.npz')