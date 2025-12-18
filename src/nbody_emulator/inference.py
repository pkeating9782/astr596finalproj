# inference.py

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices("cpu")[0])
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def make_model(emulator, input_normalizer, output_normalizer, sigma_obs):
    """
    Returns a NumPyro model function for inference.
    
    Args:
        emulator: trained NN emulator (or list of models for ensemble)
        input_normalizer: Normalizer for inputs (Q0, a)
        output_normalizer: Normalizer for outputs (f_bound, sigma_v, r_h)
        sigma_obs: (3,) array of observation uncertainties [sigma_f, sigma_v, sigma_r]
    """
    def model(observed_stats):
        # Step 1: Sample from uniform priors (over training range)
        Q0 = numpyro.sample("Q0", dist.Uniform(0.5, 1.5))
        a = numpyro.sample("a", dist.Uniform(50.0, 200.0))
        
        # Stack into input array
        params = jnp.array([Q0, a])
        
        # Step 2: Forward model with proper normalization
        # Normalize inputs using TRAINING statistics
        params_norm = input_normalizer.normalize(params)
        
        # Get prediction from emulator
        if isinstance(emulator, list):
            # Ensemble: use predict_ensemble to get mean prediction
            from src.nbody_emulator import emulator as em_module
            pred_norm, _ = em_module.predict_ensemble(emulator, params_norm)
        else:
            # Single model
            pred_norm = emulator(params_norm)
        
        # Denormalize outputs back to physical units
        predicted_stats = output_normalizer.denormalize(pred_norm)
        
        # Step 3: Likelihood - compare predictions to observations
        # Each output has independent Gaussian noise with σ_obs
        numpyro.sample("obs", dist.Normal(predicted_stats, sigma_obs), obs=observed_stats)
    
    return model

def run_inference(model, observed_stats, num_warmup=500, num_samples=2000, seed=0):
    """
    Run MCMC inference using NUTS.
    
    Args:
        model: NumPyro model function (from make_model)
        observed_stats: (3,) array of observed [f_bound, sigma_v, r_h]
        num_warmup: number of warmup steps for adaptation
        num_samples: number of posterior samples to collect
        seed: random seed for reproducibility
    
    Returns:
        samples: dictionary of posterior samples with keys 'Q0' and 'a'
    """
    # Set up NUTS sampler
    nuts_kernel = NUTS(model)
    
    # Set up MCMC
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=True
    )
    
    # Run inference
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, observed_stats)
    
    # Print diagnostics
    print("\n" + "="*60)
    print("MCMC DIAGNOSTICS")
    print("="*60)
    mcmc.print_summary()
    
    # Get posterior samples
    samples = mcmc.get_samples()
    
    return samples