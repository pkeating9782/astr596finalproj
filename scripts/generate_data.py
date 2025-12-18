# generate_data.py


import jax 
import jax.numpy as jnp
import numpy as np
from src.nbody_emulator import utils, data
from scipy.stats import qmc
from jax_nbody import samplers, integrators
jax.config.update("jax_enable_x64", True)

# Usage
train, test = data.generate_training_data(n_train=80, n_test=20)
# train.shape = (80, 2), test.shape = (20, 2)

# Save for later use
np.save('outputs/data/train_params.npy', train)
np.save('outputs/data/test_params.npy', test)

# Save as CSV (human-readable)
np.savetxt('outputs/data/train_params.csv', train, delimiter=',', 
           header='Q0,a', comments='')

# Load parameters
train_params = np.load('outputs/data/train_params.npy')

results=[]
# Loop through each parameter set
for i, (Q0, a) in enumerate(train_params):
    print(f"Simulation {i+1}/{len(train_params)}: Q0={Q0:.3f}, a={a:.1f}")
    
    # Generate initial conditions with these parameters
    masses, positions, velocities = samplers.IMFPlum(
        key=jax.random.PRNGKey(i),
        N=200,
        a=a,
        G=1.0
    )

    init_state = (positions, velocities, masses)

    N = 200
    dt = 0.005 * utils.compute_crossing_time(a, np.sum(masses))
    e2 = utils.compute_softening(a,N)**2 # softening parameter 
    T = 10 * utils.compute_crossing_time(a, np.sum(masses)) # duration
    n_steps = T // dt
    
    # Adjust velocities to achieve desired Q0
    # (scale velocity to match target virial ratio)
    
    # Run simulation
    final_state, trajectory = integrators.integrator(init_state, dt, n_steps, G=1.0, eps2=e2)
    positions1, velocities1, masses1 = final_state

    # Extract summary statistics
    stats = utils.compute_all_summary_stats(positions1, velocities1, masses1, G=1.0, eps2=e2)
    
    # Save results
    results.append({'Q0': Q0, 'a': a, **stats})

    # Check good values
    # Sanity check
    check = utils.sanity_check_summary_stats(stats)
    print(f"  f_bound={stats['f_bound']:.3f}, sigma_v={stats['sigma_v']:.3f}, r_h={stats['r_h']:.1f}")
    if not check['in_range']:
        print(f"  WARNING: {check['warnings']}")
    print()

    # Convert to arrays
    train_outputs = np.array([
        [r['f_bound'], r['sigma_v'], r['r_h']] 
        for r in results])

    # Save outputs
    np.save('outputs/data/train_outputs.npy', train_outputs)
    np.savetxt('outputs/data/train_outputs.csv', train_outputs, delimiter=',',
            header='f_bound,sigma_v,r_h', comments='')