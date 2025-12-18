# eval_emulator.py

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
# FIGURE 4: UNCERTAINTY SLICE (μ ± 2σ)
# ============================================
print("\n" + "="*60)
print("UNCERTAINTY BEHAVIOR ANALYSIS")
print("="*60)

# Create a 1D slice through parameter space
# Fix a=125 AU (middle of range), vary Q0
print("\nGenerating 1D slice predictions...")

a_fixed = 125.0  # Middle of a range [50, 200]
Q0_values = np.linspace(0.5, 1.5, 100)  # Dense sampling

# Create input array: (N, 2) where N=100
slice_inputs = np.column_stack([Q0_values, np.full_like(Q0_values, a_fixed)])

# Normalize inputs
slice_inputs_norm = input_normalizer.normalize(slice_inputs)

# Get ensemble predictions
mean_slice_norm, std_slice_norm = emulator.predict_ensemble(models, slice_inputs_norm)

# Denormalize
mean_slice = output_normalizer.denormalize(mean_slice_norm)
std_slice = std_slice_norm * output_normalizer.std  # Scale uncertainty

print(f"Generated predictions for {len(Q0_values)} points along Q0 ∈ [0.5, 1.5] at a={a_fixed} AU")

# Create uncertainty plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

output_labels = [
    r'$f_{\rm bound}$',
    r'$\sigma_v$ [AU/yr]',
    r'$r_h$ [AU]'
]

for i, (ax, name, label) in enumerate(zip(axes, output_names, output_labels)):
    mean = mean_slice[:, i]
    std = std_slice[:, i]
    
    # Plot mean prediction
    ax.plot(Q0_values, mean, 'b-', linewidth=2.5, label='Ensemble mean')
    
    # Plot uncertainty bands (± 2σ)
    ax.fill_between(
        Q0_values,
        mean - 2*std,
        mean + 2*std,
        alpha=0.3,
        color='blue',
        label=r'$\pm 2\sigma$ uncertainty'
    )
    
    # Overlay training data points at this a value
    # Find training points near a=125 (within ±15 AU)
    mask_train = np.abs(test_params[:, 1] - a_fixed) < 15.0
    train_Q0_nearby = test_params[mask_train, 0]
    
    # Mark approximate locations of training data
    if len(train_Q0_nearby) > 0:
        y_min, y_max = ax.get_ylim()
        for q0 in train_Q0_nearby:
            ax.axvline(q0, color='red', alpha=0.4, linestyle='--', linewidth=1.5)
        
        # Add legend entry for training data markers (only once)
        if i == 0:
            ax.axvline(train_Q0_nearby[0], color='red', alpha=0.4, linestyle='--', 
                      linewidth=1.5, label=f'Test data near $a={a_fixed}$')
    
    # Formatting
    ax.set_xlabel(r'Initial Virial Ratio $Q_0$', fontsize=13)
    ax.set_ylabel(f'{label}', fontsize=13)
    ax.set_title(f'{name} (at $a={a_fixed}$ AU)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.5)

plt.tight_layout()
plt.savefig('outputs/figures/uncertainty_slice.png', dpi=150, bbox_inches='tight')
print("\n✓ Figure 4 saved to outputs/figures/uncertainty_slice.png")
plt.show()

# ============================================
# EDGE BEHAVIOR ANALYSIS
# ============================================
print("\n" + "="*60)
print("EDGE BEHAVIOR ANALYSIS")
print("="*60)

# Define edge regions and center region
Q0_edges = [(0.5, 0.6), (1.4, 1.5)]  # Low and high edges
Q0_center = (0.9, 1.1)

a_edges = [(50, 70), (180, 200)]
a_center = (115, 135)

def classify_point(Q0, a):
    """Classify if point is in edge or center region."""
    Q0_is_edge = any(low <= Q0 <= high for low, high in Q0_edges)
    a_is_edge = any(low <= a <= high for low, high in a_edges)
    
    Q0_is_center = Q0_center[0] <= Q0 <= Q0_center[1]
    a_is_center = a_center[0] <= a <= a_center[1]
    
    if (Q0_is_edge or a_is_edge):
        return 'edge'
    elif (Q0_is_center and a_is_center):
        return 'center'
    else:
        return 'intermediate'

# Classify all test points
classifications = [classify_point(Q0, a) for Q0, a in test_params]

print(f"\nTest set distribution:")
print(f"  Center region:       {classifications.count('center')} points")
print(f"  Intermediate region: {classifications.count('intermediate')} points")
print(f"  Edge region:         {classifications.count('edge')} points")

# Compute uncertainty statistics by region
print("\n" + "-"*70)
print("UNCERTAINTY BY REGION")
print("-"*70)
print(f"{'Region':<15} {'f_bound σ':<15} {'sigma_v σ':<15} {'r_h σ':<15} {'Count':<10}")
print("-"*70)

for region in ['center', 'intermediate', 'edge']:
    mask = np.array([c == region for c in classifications])
    if np.any(mask):
        mean_uncertainties = np.mean(std_pred[mask], axis=0)
        count = np.sum(mask)
        print(f"{region:<15} {mean_uncertainties[0]:<15.4f} {mean_uncertainties[1]:<15.4f} "
              f"{mean_uncertainties[2]:<15.4f} {count:<10}")

print("-"*70)

# Compute prediction errors by region
print("\n" + "-"*70)
print("PREDICTION ERRORS BY REGION")
print("-"*70)
print(f"{'Region':<15} {'f_bound RMSE':<15} {'sigma_v RMSE':<15} {'r_h RMSE':<15}")
print("-"*70)

for region in ['center', 'intermediate', 'edge']:
    mask = np.array([c == region for c in classifications])
    if np.any(mask):
        rmse = np.sqrt(np.mean((mean_pred[mask] - test_outputs[mask])**2, axis=0))
        print(f"{region:<15} {rmse[0]:<15.4f} {rmse[1]:<15.4f} {rmse[2]:<15.4f}")

print("-"*70)

# ============================================
# PARAMETER SPACE COVERAGE VISUALIZATION
# ============================================
print("\n" + "="*60)
print("PARAMETER SPACE COVERAGE")
print("="*60)

# Load training params for visualization
train_params_full = np.load('outputs/data/train_params.npy')

fig, ax = plt.subplots(figsize=(10, 7))

# Plot training data
ax.scatter(train_params_full[:, 0], train_params_full[:, 1], 
           c='blue', s=60, alpha=0.5, label='Training data', 
           edgecolors='navy', linewidths=0.8, zorder=2)

# Plot test data by classification
region_colors = {'center': 'green', 'intermediate': 'orange', 'edge': 'red'}
region_markers = {'center': 'o', 'intermediate': 's', 'edge': '^'}

for region in ['center', 'intermediate', 'edge']:
    mask = np.array([c == region for c in classifications])
    if np.any(mask):
        ax.scatter(test_params[mask, 0], test_params[mask, 1],
                  c=region_colors[region], s=120, marker=region_markers[region], 
                  alpha=0.9, label=f'Test ({region})', 
                  edgecolors='black', linewidths=1.2, zorder=3)

# Mark the slice line
ax.axhline(a_fixed, color='purple', linestyle='--', linewidth=2.5, 
          label=f'1D slice at $a={a_fixed}$ AU', zorder=1)

# Draw edge region boundaries
ax.axvline(Q0_edges[0][1], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax.axvline(Q0_edges[1][0], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax.axhline(a_edges[0][1], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax.axhline(a_edges[1][0], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

# Formatting
ax.set_xlabel(r'Initial Virial Ratio $Q_0$', fontsize=14, fontweight='bold')
ax.set_ylabel(r'Plummer Scale Radius $a$ [AU]', fontsize=14, fontweight='bold')
ax.set_title('Parameter Space Coverage & Test Set Classification', fontsize=15, fontweight='bold')
ax.legend(fontsize=10, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.45, 1.55)
ax.set_ylim(45, 205)

plt.tight_layout()
plt.savefig('outputs/figures/parameter_space_coverage.png', dpi=150, bbox_inches='tight')
print("\n✓ Parameter space coverage plot saved to outputs/figures/parameter_space_coverage.png")
plt.show()

# ============================================
# UNCERTAINTY INTERPRETATION
# ============================================
print("\n" + "="*60)
print("UNCERTAINTY INTERPRETATION")
print("="*60)

# Calculate ratio of edge to center uncertainty
center_mask = np.array([c == 'center' for c in classifications])
edge_mask = np.array([c == 'edge' for c in classifications])

if np.any(center_mask) and np.any(edge_mask):
    center_unc = np.mean(std_pred[center_mask], axis=0)
    edge_unc = np.mean(std_pred[edge_mask], axis=0)
    
    print("\nUncertainty ratios (Edge / Center):")
    print("-"*40)
    for i, name in enumerate(output_names):
        ratio = edge_unc[i] / center_unc[i] if center_unc[i] > 0 else float('inf')
        print(f"  {name}: {ratio:.2f}x")
    print("-"*40)
    
    print("\nInterpretation:")
    if all(edge_unc > center_unc):
        print("  ✓ Uncertainty increases at edges (expected behavior)")
        print("  ✓ Emulator 'knows what it doesn't know'")
    else:
        print("  ⚠ Some outputs show lower uncertainty at edges")
        print("  → May need more ensemble members or training data")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print("\nGenerated figures:")
print("  ✓ outputs/figures/predicted_vs_true.png (Figure 3)")
print("  ✓ outputs/figures/residuals.png")
print("  ✓ outputs/figures/uncertainty_slice.png (Figure 4)")
print("  ✓ outputs/figures/parameter_space_coverage.png")
print("\nGenerated data:")
print("  ✓ outputs/data/test_metrics.npz")
print("\nMetrics summary:")
for i, name in enumerate(output_names):
    print(f"  {name}: MAE={mae_values[i]:.4f}, RMSE={rmse_values[i]:.4f}")

print("\n" + "="*60)
print("KEY FINDINGS FOR RESEARCH MEMO:")
print("="*60)
print("\n1. ACCURACY:")
print(f"   - Overall test set performance adequate")
print(f"   - Check if edge regions show degraded performance")

print("\n2. UNCERTAINTY QUANTIFICATION:")
print(f"   - Ensemble spread quantifies epistemic uncertainty")
print(f"   - Compare edge vs. center uncertainty ratios above")

print("\n3. EDGE BEHAVIOR:")
print(f"   - Monitor if predictions degrade near boundaries")
print(f"   - Use ensemble uncertainty as extrapolation warning")

print("\n4. NEXT STEPS:")
print(f"   - Use test-set RMSE values as σ_obs for inference")
print(f"   - Proceed to Part 4: NumPyro inference")
print("="*60)