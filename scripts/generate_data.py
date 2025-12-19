# generate_data.py

import jax 
import jax.numpy as jnp
import numpy as np
from src.nbody_emulator import utils, data
from scipy.stats import qmc
from jax_nbody import samplers, integrators
from jax_nbody.diagnostics import kinetic, gravitational_potential
jax.config.update("jax_enable_x64", True)

def run_simulation_for_params(params, seed_offset=0):
    """
    Run N-body simulations for a set of parameters.
    
    Args:
        params: (N, 2) array of (Q0, a) values
        seed_offset: offset for random seeds (use different values for train/test)
    
    Returns:
        results: list of dicts with simulation outputs
    """
    results = []
    
    # Loop through each parameter set
    for i, (Q0, a) in enumerate(params):
        print(f"Simulation {i+1}/{len(params)}: Q0={Q0:.3f}, a={a:.1f}")
        
        # Generate initial conditions with these parameters
        masses, positions, velocities = samplers.IMFPlum(
            key=jax.random.PRNGKey(seed_offset + i),
            N=200,
            a=a,
            G=1.0
        )

        N = 200
        dt = 0.005 * utils.compute_crossing_time(a, np.sum(masses))
        e2 = utils.compute_softening(a, N)**2  # softening parameter 
        T = 10 * utils.compute_crossing_time(a, np.sum(masses))  # duration
        n_steps = int(T / dt)  # Make sure this is an integer
        
        # Adjust velocities to achieve desired Q0
        # Initial energy
        K_init = kinetic(masses, velocities)
        W_init = gravitational_potential(masses, positions, G=1.0, eps2=e2)
        Q_init = 2 * K_init / jnp.abs(W_init)

        # Scale velocities to achieve target Q0
        scale_factor = jnp.sqrt(Q0 / Q_init)
        velocities = velocities * scale_factor

        # Verify
        K_scaled = kinetic(masses, velocities)
        Q_check = 2 * K_scaled / jnp.abs(W_init)
        print(f"  Target Q0={Q0:.3f}, Actual Q0={Q_check:.3f}")

        init_state = (positions, velocities, masses)
        
        # Run simulation
        final_state, trajectory = integrators.integrator(
            init_state, dt, n_steps, G=1.0, eps2=e2
        )
        positions_final, velocities_final, masses_final = final_state

        # Extract summary statistics
        stats = utils.compute_all_summary_stats(
            positions_final, velocities_final, masses_final, G=1.0, eps2=e2
        )
        
        # Save results
        results.append({'Q0': Q0, 'a': a, **stats})

        # Sanity check
        check = utils.sanity_check_summary_stats(stats)
        print(f"  f_bound={stats['f_bound']:.3f}, sigma_v={stats['sigma_v']:.3f}, r_h={stats['r_h']:.1f}")
        if not check['in_range']:
            print(f"  WARNING: {check['warnings']}")
        print()
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("GENERATING TRAINING AND TEST DATA")
    print("="*60)
    
    # Generate parameter samples using LHS
    print("\nGenerating parameter samples...")
    train_params, test_params = data.generate_training_data(n_train=100, n_test=30)
    
    # Save parameters
    np.save('outputs/data/train_params.npy', train_params)
    np.save('outputs/data/test_params.npy', test_params)
    np.savetxt('outputs/data/train_params.csv', train_params, delimiter=',', 
               header='Q0,a', comments='')
    np.savetxt('outputs/data/test_params.csv', test_params, delimiter=',',
               header='Q0,a', comments='')
    
    print(f"Generated {len(train_params)} training parameters")
    print(f"Generated {len(test_params)} test parameters")
    
    # ========================================
    # TRAINING SET SIMULATIONS
    # ========================================
    print("\n" + "="*60)
    print("RUNNING TRAINING SIMULATIONS")
    print("="*60 + "\n")
    
    train_results = run_simulation_for_params(train_params, seed_offset=0)
    
    # Convert to arrays
    train_outputs = np.array([
        [r['f_bound'], r['sigma_v'], r['r_h']] 
        for r in train_results
    ])
    
    # Save training outputs
    np.save('outputs/data/train_outputs.npy', train_outputs)
    np.savetxt('outputs/data/train_outputs.csv', train_outputs, delimiter=',',
               header='f_bound,sigma_v,r_h', comments='')
    
    print(f"\n✓ Training data complete: {len(train_outputs)} simulations")
    
    # ========================================
    # TEST SET SIMULATIONS
    # ========================================
    print("\n" + "="*60)
    print("RUNNING TEST SIMULATIONS")
    print("="*60 + "\n")
    
    # Use different seed offset for test set (important!)
    test_results = run_simulation_for_params(test_params, seed_offset=10000)
    
    # Convert to arrays
    test_outputs = np.array([
        [r['f_bound'], r['sigma_v'], r['r_h']] 
        for r in test_results
    ])
    
    # Save test outputs
    np.save('outputs/data/test_outputs.npy', test_outputs)
    np.savetxt('outputs/data/test_outputs.csv', test_outputs, delimiter=',',
               header='f_bound,sigma_v,r_h', comments='')
    
    print(f"\n✓ Test data complete: {len(test_outputs)} simulations")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"\nTraining set:")
    print(f"  - {len(train_params)} parameter combinations")
    print(f"  - Saved to outputs/data/train_params.npy")
    print(f"  - Saved to outputs/data/train_outputs.npy")
    
    print(f"\nTest set:")
    print(f"  - {len(test_params)} parameter combinations")
    print(f"  - Saved to outputs/data/test_params.npy")
    print(f"  - Saved to outputs/data/test_outputs.npy")
    
    print(f"\nOutput ranges:")
    print(f"  f_bound: [{train_outputs[:, 0].min():.3f}, {train_outputs[:, 0].max():.3f}]")
    print(f"  sigma_v: [{train_outputs[:, 1].min():.3f}, {train_outputs[:, 1].max():.3f}]")
    print(f"  r_h:     [{train_outputs[:, 2].min():.1f}, {train_outputs[:, 2].max():.1f}]")
    
    print(f"\nNext steps:")
    print(f"  1. Verify output ranges are reasonable")
    print(f"  2. Plot training data distribution (Part 1 deliverable)")
    print(f"  3. Train your emulator (Part 2)")