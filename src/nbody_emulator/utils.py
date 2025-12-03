# utils.py

import jax.numpy as jnp

def compute_crossing_time(a, M_total, G=1.0):
    """
    Compute crossing time for a Plummer sphere.
    
    t_cross ~ sqrt(a^3 / GM)
    
    Args:
        a: Plummer scale radius
        M_total: Total system mass
        G: Gravitational constant
    
    Returns:
        Crossing time
    """
    return jnp.sqrt(a**3 / (G * M_total))