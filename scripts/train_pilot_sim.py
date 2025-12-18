# train_pilot_sim.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.nbody_emulator import emulator, utils

# Loading your pilot data
train_params = np.load('outputs/data/train_params.npy')[:30]
train_outputs = np.load('outputs/data/train_outputs.npy')[:30]

input_normalizer = utils.Normalizer()
output_normalizer = utils.Normalizer()

input_normalizer.fit(train_params)
output_normalizer.fit(train_outputs)

x_train_norm = input_normalizer.normalize(train_params)
y_train_norm = output_normalizer.normalize(train_outputs)

print("Training data shapes:")
print(f"  Inputs: {x_train_norm.shape}")
print(f"  Outputs: {y_train_norm.shape}")
print(f"\nNormalization check:")
print(f"  Input mean: {jnp.mean(x_train_norm, axis=0)}")
print(f"  Input std: {jnp.std(x_train_norm, axis=0)}")
print(f"  Output mean: {jnp.mean(y_train_norm, axis=0)}")
print(f"  Output std: {jnp.std(y_train_norm, axis=0)}")
print()

key = jax.random.PRNGKey(42)
model, losses = emulator.train_single_model(
    key, 
    x_train_norm, 
    y_train_norm,
    learning_rate=1e-3,
    num_epochs=1000,
    print_every=50
)

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE on normalized data)')
plt.yscale('log')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig('outputs/figures/training_loss_pilot.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*60}")
print("TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Final loss: {losses[-1]:.6f}")
print(f"Loss should be << 0.01 for good predictions")
print(f"\nNext steps:")
print(f"  1. Check that loss decreased smoothly")
print(f"  2. Test predictions on a few training examples")
print(f"  3. If loss is too high, try: lower LR, more epochs, check data")