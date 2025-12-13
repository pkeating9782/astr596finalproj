# fix_timestep.py

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax_nbody.samplers import IMFPlum
from jax_nbody.integrators import integrator
from jax_nbody.diagnostics import total_energy
from src.nbody_emulator import utils
jax.config.update("jax_enable_x64", True)

# Fixed parameters for testing
key = jax.random.PRNGKey(42)
N = 200
a = 100.0  # AU
Q0 = 1.0  # Start with virial equilibrium
G = 1.0
eps2 = (0.1 * a / N**(1/3))**2

# Generate test system
masses, positions, velocities = IMFPlum(key, N, a=a, G=G)

# Scale velocities to desired Q0
K_init, W_init, _ = total_energy(positions, velocities, masses, G, eps2)
Q_actual = 2 * K_init / jnp.abs(W_init)
velocities = velocities * jnp.sqrt(Q0 / Q_actual)

# Recompute energy with scaled velocities
K_init, W_init, E_init = total_energy(positions, velocities, masses, G, eps2)
print(f"Initial Q: {2*K_init/jnp.abs(W_init):.4f}")
print(f"Initial energy: {E_init:.6f}")

# Compute crossing time and simulation duration
M_total = jnp.sum(masses)
t_cross = utils.compute_crossing_time(a, M_total, G)
t_final = 10 * t_cross
print(f"Crossing time: {t_cross:.2f}")
print(f"Final time (10 t_cross): {t_final:.2f}")

# Try different timesteps
dt_factors = [0.02, 0.01, 0.005]  # Fractions of t_cross

for dt_factor in dt_factors:
    dt = dt_factor * t_cross
    n_steps = int(t_final / dt)
    
    print(f"\nTesting dt = {dt:.4f} ({dt_factor} * t_cross)")
    print(f"Number of steps: {n_steps}")
    
    # Run integration
    state0 = (positions, velocities, masses)
    final_state, trajectory = integrator(state0, dt, n_steps, G, eps2)
    
    # Compute energy at each saved snapshot
    pos_traj = trajectory  # Shape: (n_steps, N, 3)
    
    # Compute energies along trajectory
    energies = []
    for i in range(0, n_steps, max(1, n_steps//100)):  # Sample ~100 points
        pos = pos_traj[i]
        # Reconstruct velocity from position differences
        if i > 0:
            vel = (pos_traj[i] - pos_traj[i-1]) / dt
        else:
            vel = velocities
        _, _, E = total_energy(pos, vel, masses, G, eps2)
        energies.append(E)
    
    energies = jnp.array(energies)
    rel_error = jnp.abs((energies - E_init) / E_init)
    
    print(f"Max relative energy error: {jnp.max(rel_error):.2e}")
    print(f"Mean relative energy error: {jnp.mean(rel_error):.2e}")
    
    # Plot energy evolution
    plt.figure(figsize=(10, 4))
    times = jnp.linspace(0, t_final, len(energies))
    plt.plot(times / t_cross, rel_error)
    plt.xlabel('Time (crossing times)')
    plt.ylabel('Relative Energy Error')
    plt.title(f'dt = {dt:.4f} ({dt_factor} * t_cross)')
    plt.axhline(1e-4, color='r', linestyle='--', label='Target: 1e-4')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'outputs/figures/energy_error_dt{dt_factor}.png', dpi=150)
    plt.close()
    
    print(f"Saved figure to outputs/figures/energy_error_dt{dt_factor}.png")