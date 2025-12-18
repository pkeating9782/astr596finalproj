# test_emulator.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.nbody_emulator import emulator, utils
import pickle

# ============================================
# LOAD TRAINED ENSEMBLE AND NORMALIZERS
# ============================================
print("\nLoading trained ensemble...")
with open('outputs/ensemble_models.pkl', 'rb') as f:
    models = pickle.load(f)

input_normalizer = utils.Normalizer.load('outputs/input_normalizer.npz')
output_normalizer = utils.Normalizer.load('outputs/output_normalizer.npz')

print(f"Loaded {len(models)} ensemble members")

# ============================================
# LOAD TEST DATA
# ============================================
print("\nLoading test data...")
test_params = np.load('outputs/data/test_params.npy')
test_outputs = np.load('outputs/data/test_outputs.npy')

print(f"Test set size: {len(test_params)} simulations")
print(f"  Input shape: {test_params.shape}")
print(f"  Output shape: {test_outputs.shape}")

# ============================================
# EVALUATE ON ENTIRE TEST SET
# ============================================
print("\n" + "="*60)
print("EVALUATING ENSEMBLE ON TEST SET")
print("="*60)

# Normalize test inputs
x_test_norm = input_normalizer.normalize(test_params)

# Get ensemble predictions (normalized)
mean_pred_norm, std_pred_norm = emulator.predict_ensemble(models, x_test_norm)

# Denormalize predictions
mean_pred = output_normalizer.denormalize(mean_pred_norm)
std_pred = std_pred_norm * output_normalizer.std  # Scale uncertainty back

# ============================================
# COMPUTE ACCURACY METRICS
# ============================================
print("\nComputing accuracy metrics...")

# Extract individual outputs
y_true = test_outputs
y_pred = mean_pred

output_names = ['f_bound', 'sigma_v', 'r_h']

# Compute MAE and RMSE for each output
mae_values = []
rmse_values = []

print("\n" + "-"*60)
print("ACCURACY METRICS")
print("-"*60)
print(f"{'Output':<15} {'MAE':<15} {'RMSE':<15}")
print("-"*60)

for i, name in enumerate(output_names):
    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
    
    # RMSE: Root Mean Square Error
    rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i])**2))
    
    mae_values.append(mae)
    rmse_values.append(rmse)
    
    print(f"{name:<15} {mae:<15.4f} {rmse:<15.4f}")

print("-"*60)

# Save metrics to file
metrics_dict = {
    'output_names': output_names,
    'MAE': mae_values,
    'RMSE': rmse_values
}
np.savez('outputs/data/test_metrics.npz', **metrics_dict)
print("\n✓ Metrics saved to outputs/data/test_metrics.npz")

# ============================================
# FIGURE 3: PREDICTED VS TRUE SCATTER PLOTS
# ============================================
print("\n" + "="*60)
print("GENERATING PREDICTED VS. TRUE PLOTS (Figure 3)")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

output_labels = [
    r'$f_{\rm bound}$',
    r'$\sigma_v$ [AU/yr]',
    r'$r_h$ [AU]'
]

for i, (ax, name, label) in enumerate(zip(axes, output_names, output_labels)):
    # Scatter plot: predicted vs true
    ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    
    # Perfect prediction line (y=x)
    min_val = min(y_true[:, i].min(), y_pred[:, i].min())
    max_val = max(y_true[:, i].max(), y_pred[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Labels and formatting
    ax.set_xlabel(f'True {label}', fontsize=12)
    ax.set_ylabel(f'Predicted {label}', fontsize=12)
    ax.set_title(f'{name}\nMAE={mae_values[i]:.4f}, RMSE={rmse_values[i]:.4f}', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('outputs/figures/predicted_vs_true.png', dpi=150, bbox_inches='tight')
print("\n✓ Figure 3 saved to outputs/figures/predicted_vs_true.png")
plt.show()

# ============================================
# ADDITIONAL: RESIDUAL PLOTS
# ============================================
print("\nGenerating residual plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (ax, name, label) in enumerate(zip(axes, output_names, output_labels)):
    residuals = y_pred[:, i] - y_true[:, i]
    
    # Residual vs true value
    ax.scatter(y_true[:, i], residuals, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    
    # Labels
    ax.set_xlabel(f'True {label}', fontsize=12)
    ax.set_ylabel(f'Residual (Pred - True)', fontsize=12)
    ax.set_title(f'{name} Residuals', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/residuals.png', dpi=150, bbox_inches='tight')
print("✓ Residual plots saved to outputs/figures/residuals.png")
plt.show()

# ============================================
# SINGLE EXAMPLE VERIFICATION
# ============================================
print("\n" + "="*60)
print("SINGLE TEST CASE EXAMPLE")
print("="*60)

test_idx = 5
x_test = test_params[test_idx]
y_true_single = test_outputs[test_idx]

print(f"\nTest case {test_idx}: Q0={x_test[0]:.3f}, a={x_test[1]:.1f}")
print(f"\nTrue outputs:")
print(f"  f_bound = {y_true_single[0]:.3f}")
print(f"  sigma_v = {y_true_single[1]:.3f}")
print(f"  r_h     = {y_true_single[2]:.1f}")

print(f"\nEnsemble predictions:")
print(f"  f_bound = {mean_pred[test_idx, 0]:.3f} ± {std_pred[test_idx, 0]:.3f}")
print(f"  sigma_v = {mean_pred[test_idx, 1]:.3f} ± {std_pred[test_idx, 1]:.3f}")
print(f"  r_h     = {mean_pred[test_idx, 2]:.1f} ± {std_pred[test_idx, 2]:.1f}")

print(f"\nAbsolute errors:")
print(f"  f_bound: {abs(mean_pred[test_idx, 0] - y_true_single[0]):.3f}")
print(f"  sigma_v: {abs(mean_pred[test_idx, 1] - y_true_single[1]):.3f}")
print(f"  r_h:     {abs(mean_pred[test_idx, 2] - y_true_single[2]):.1f}")

# ============================================
# UNCERTAINTY ANALYSIS
# ============================================
print("\n" + "="*60)
print("UNCERTAINTY STATISTICS")
print("="*60)

print(f"\n{'Output':<15} {'Mean Unc.':<15} {'Min Unc.':<15} {'Max Unc.':<15}")
print("-"*60)
for i, name in enumerate(output_names):
    mean_unc = np.mean(std_pred[:, i])
    min_unc = np.min(std_pred[:, i])
    max_unc = np.max(std_pred[:, i])
    print(f"{name:<15} {mean_unc:<15.4f} {min_unc:<15.4f} {max_unc:<15.4f}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print("\nGenerated:")
print("  ✓ outputs/figures/predicted_vs_true.png (Figure 3)")
print("  ✓ outputs/figures/residuals.png")
print("  ✓ outputs/data/test_metrics.npz")
print("\nMetrics summary:")
for i, name in enumerate(output_names):
    print(f"  {name}: MAE={mae_values[i]:.4f}, RMSE={rmse_values[i]:.4f}")