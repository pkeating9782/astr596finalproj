# test_existing_code.py

import jax
import jax.numpy as jnp
import jax_nbody
from jax_nbody.samplers import IMFPlum
from jax_nbody.integrators import integrator
from jax_nbody.diagnostics import total_energy, virial_ratio

def main():
    # Test parameters
    key = jax.random.PRNGKey(42)
    N = 200
    a = 100.0  # AU
    G = 1.0
    eps2 = (0.1 * a / N**(1/3))**2  # ~10% of mean interparticle spacing

    masses, positions, velocities = IMFPlum(key, N, a=a, G=G)

    K_init, W_init, E_init = total_energy(positions, velocities, masses, G, eps2)
    Q_init = virial_ratio(K_init, W_init)
    print(f"Initial virial ratio: {Q_init:.4f}")
    print(f"Initial energy: {E_init:.6f}")

if __name__ == '__main__':
    main()