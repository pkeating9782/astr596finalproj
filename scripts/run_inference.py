# run_inference.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.nbody_emulator import inference, utils
import pickle
import corner

# ============================================
# 1. LOAD TRAINED ENSEMBLE AND NORMALIZERS
# ============================================
print("\nLoading trained ensemble and normalizers...")
with open('outputs/ensemble_models.pkl', 'rb') as f:
    models = pickle.load(f)

input_normalizer = utils.Normalizer.load('outputs/input_normalizer.npz')
output_normalizer = utils.Normalizer.load('outputs/output_normalizer.npz')

print(f"✓ Loaded {len(models)} ensemble members")
print(f"✓ Loaded normalizers")

# ============================================
# 2. COMPUTE σ_obs FROM TEST SET RMSE
# ============================================
print("\nComputing observation uncertainties (σ_obs)...")
test_metrics = np.load('outputs/data/test_metrics.npz')
sigma_obs = jnp.array(test_metrics['RMSE'])

print(f"σ_obs values (test-set RMSE):")
print(f"  σ(f_bound) = {sigma_obs[0]:.4f}")
print(f"  σ(sigma_v) = {sigma_obs[1]:.4f}")
print(f"  σ(r_h)     = {sigma_obs[2]:.4f}")

# ============================================
# 3. SELECT A TEST CASE FOR VALIDATION
# ============================================
print("\n" + "="*60)
print("PARAMETER RECOVERY TEST")
print("="*60)

# Pick a test case with known true parameters
test_idx = 5  # You can change this
test_params = np.load('outputs/data/test_params.npy')
test_outputs = np.load('outputs/data/test_outputs.npy')

true_Q0 = test_params[test_idx, 0]
true_a = test_params[test_idx, 1]
observed = jnp.array(test_outputs[test_idx])

print(f"\nTest case {test_idx}:")
print(f"  True Q0 = {true_Q0:.3f}")
print(f"  True a  = {true_a:.1f}")
print(f"\nObserved statistics:")
print(f"  f_bound = {observed[0]:.3f}")
print(f"  sigma_v = {observed[1]:.3f}")
print(f"  r_h     = {observed[2]:.1f}")

# ============================================
# 4. CREATE NUMPYRO MODEL
# ============================================
print("\nCreating NumPyro model...")
model = inference.make_model(
    emulator=models,
    input_normalizer=input_normalizer,
    output_normalizer=output_normalizer,
    sigma_obs=sigma_obs
)

# ============================================
# 5. RUN NUTS INFERENCE
# ============================================
print("\n" + "="*60)
print("RUNNING NUTS SAMPLER")
print("="*60)

samples = inference.run_inference(
    model=model,
    observed_stats=observed,
    num_warmup=500,
    num_samples=2000,
    seed=42
)

# ============================================
# 6. ANALYZE POSTERIOR
# ============================================
print("\n" + "="*60)
print("POSTERIOR ANALYSIS")
print("="*60)

# Compute credible intervals
Q0_samples = samples['Q0']
a_samples = samples['a']

Q0_mean = jnp.mean(Q0_samples)
Q0_std = jnp.std(Q0_samples)
Q0_ci = jnp.percentile(Q0_samples, jnp.array([2.5, 97.5]))

a_mean = jnp.mean(a_samples)
a_std = jnp.std(a_samples)
a_ci = jnp.percentile(a_samples, jnp.array([2.5, 97.5]))

print(f"\nPosterior for Q0:")
print(f"  Mean: {Q0_mean:.3f} ± {Q0_std:.3f}")
print(f"  95% CI: [{Q0_ci[0]:.3f}, {Q0_ci[1]:.3f}]")
print(f"  True value: {true_Q0:.3f}")
print(f"  {'✓ RECOVERED' if Q0_ci[0] <= true_Q0 <= Q0_ci[1] else '✗ MISSED'}")

print(f"\nPosterior for a:")
print(f"  Mean: {a_mean:.1f} ± {a_std:.1f}")
print(f"  95% CI: [{a_ci[0]:.1f}, {a_ci[1]:.1f}]")
print(f"  True value: {true_a:.1f}")
print(f"  {'✓ RECOVERED' if a_ci[0] <= true_a <= a_ci[1] else '✗ MISSED'}")

# ============================================
# 7. CORNER PLOT (Figure 5)
# ============================================
print("\nGenerating corner plot...")

# Prepare data for corner plot
posterior_samples = np.column_stack([Q0_samples, a_samples])

fig = corner.corner(
    posterior_samples,
    labels=[r'$Q_0$', r'$a$ [AU]'],
    truths=[true_Q0, true_a],
    truth_color='red',
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.3f'
)

plt.savefig('outputs/figures/posterior_corner_plot.png', dpi=150, bbox_inches='tight')
print("✓ Figure 5 saved to outputs/figures/posterior_corner_plot.png")
plt.show()

# ============================================
# 8. SAVE RESULTS
# ============================================
print("\nSaving inference results...")
np.savez(
    f'outputs/data/inference_results_test{test_idx}.npz',
    Q0_samples=Q0_samples,
    a_samples=a_samples,
    true_Q0=true_Q0,
    true_a=true_a,
    observed=observed,
    sigma_obs=sigma_obs
)

print(f"\n{'='*60}")
print("INFERENCE COMPLETE!")
print(f"{'='*60}")