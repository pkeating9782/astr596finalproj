# inference.py

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices("cpu")[0])
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def make_model(emulator, sigma_obs):
    """
    Returns a NumPyro model function for inference.
    
    Args:
        emulator: trained NN emulator (or ensemble predict function)
        sigma_obs: (3,) array of observation uncertainties [sigma_f, sigma_v, sigma_r]
    """
    def model(observed_stats):
        # TODO: Sample Q0 from prior (Uniform over your training range)
        # TODO: Sample a from prior (Uniform over your training range)
        
        # TODO: Call emulator to get predicted stats
        # (Remember to normalize inputs and unnormalize outputs!)
        
        # TODO: Define likelihood 
        # Hint: numpyro.sample("obs", dist.Normal(...), obs=observed_stats)
        pass
    
    return model

def run_inference(model, observed_stats, num_warmup=500, num_samples=2000):
    # TODO: Set up NUTS sampler
    # TODO: Run MCMC
    # TODO: Return samples
    pass