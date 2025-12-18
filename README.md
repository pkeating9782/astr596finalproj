# N-Body Emulator for Star Cluster Dynamics

A machine learning pipeline that accelerates Bayesian inference for N-body simulations of star clusters. This project trains a neural network to emulate expensive simulations, then uses the fast surrogate model for parameter inference with NumPyro.

## Quick Start

### Installation

```bash
# Clone the repository
git clone 
cd nbody_emulator

# Install dependencies
pip install -e .

# Required packages:
# jax, jaxlib, equinox, optax, numpyro, numpy, scipy, matplotlib, corner
```

### Usage Pipeline

#### 1. Generate Training Data

Run N-body simulations across parameter space using Latin Hypercube Sampling:

```bash
python scripts/generate_data.py
```

**Output:** 
- `outputs/data/train_params.npy` - Training inputs (Q₀, a)
- `outputs/data/train_outputs.npy` - Summary statistics (f_bound, σ_v, r_h)
- `outputs/data/test_params.npy` - Test set for validation
- `outputs/data/test_outputs.npy` - Summary statistics for test set

#### 2. Train Neural Network Emulator

Train an ensemble of 5 MLPs (2→64→64→3 architecture):

```bash
python scripts/train_emulator.py
```

**Output:**
- `outputs/ensemble_models.pkl` - Trained ensemble
- `outputs/input_normalizer.npz` - Normalization statistics for training set parameters
- `outputs/output_normalizer.npz` - Normalization statistics for training set predictions
- `outputs/figures/ensemble_training_loss.png` - Training curves

#### 3. Evaluate Emulator Performance

Compute test-set metrics and generate diagnostic plots:

```bash
python scripts/test_emulator.py
```

**Output:**
- `outputs/figures/predicted_vs_true.png` - Accuracy assessment
- `outputs/data/test_metrics.npz` - MAE and RMSE values
- Uncertainty statistics

#### 4. Bayesian Inference

Recover parameters from observed cluster properties using NUTS:

```bash
python scripts/run_inference.py
```

**Output:**
- `outputs/figures/posterior_corner_plot.png` - Joint posterior
- `outputs/data/inference_results_*.npz` - MCMC samples
- Parameter recovery validation

## Project Structure

```
nbody_emulator/
├── src/nbody_emulator/
│   ├── data.py                 # Latin Hypercube sampling
│   ├── utils.py                # Summary statistics, normalization
│   ├── emulator.py             # Neural network (Equinox)
│   └── inference.py            # NumPyro models
├── scripts/
│   ├── generate_data.py        # Step 1: Run simulations
│   ├── train_emulator.py       # Step 2: Train NN
│   ├── test_emulator.py        # Step 3: Evaluate
│   └── run_inference.py        # Step 4: MCMC
├── outputs/
│   ├── data/                   # Generated datasets
│   └── figures/                # Plots and visualizations
└── tests/
    ├── test_emulator.py        # Runs emulator evaluation
    └── unit_test_emulator.py   # Runs unit tests om emulator
```

## Dependencies

- **JAX ecosystem:** `jax`, `equinox` (NNs), `optax` (optimization), `numpyro` (probabilistic programming)
- **Scientific Python:** `numpy`, `scipy`, `matplotlib`
- **Utilities:** `corner` (posterior visualization)
- **N-body package:** `jax_nbody` package 

## Expected Results

- **Inference time:** Minutes instead of days
- **Parameter recovery:** True values within posterior 95% CI

---

**Author:** Paige Keating

**Date:** December 2025