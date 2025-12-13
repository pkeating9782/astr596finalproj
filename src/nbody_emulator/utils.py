# utils.py

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

@jax.jit
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

@jax.jit
def compute_softening(a, N):
    """
    Compute softening parameter
    
    ϵ ~ 0.1*a/N ^(1/3)
    
    Args:
        a: Plummer scale radius
        N: Number of bodies
    
    Returns:
        Softening parameter
    """
    return 0.1 * a/(N**(1/3))


def compute_potential_at_particle(i, positions, masses, G, eps2):
    """
    Compute gravitational potential at particle i due to all other particles.
    
    φ(r_i) = -G * sum_{j≠i} m_j / sqrt(|r_j - r_i|^2 + eps^2)
    
    Args:
        i: particle index
        positions: (N, 3) array of positions
        masses: (N,) array of masses
        G: gravitational constant
        eps2: softening length squared
    
    Returns:
        float: potential energy per unit mass at position i
    """
    N = len(masses)
    r_i = positions[i]
    
    # Compute separations from particle i to all others
    r_vec = positions - r_i[None, :]  # (N, 3)
    r_mag = jnp.sqrt(jnp.sum(r_vec**2, axis=-1) + eps2)  # (N,)
    
    # Compute potential contributions from all particles
    # (excluding self-interaction)
    potential_contributions = -G * masses / r_mag
    
    # Mask out self-interaction
    mask = jnp.arange(N) != i
    phi = jnp.sum(potential_contributions * mask)
    
    return phi


def compute_center_of_mass_velocity(velocities, masses):
    """
    Compute center-of-mass velocity.
    
    Args:
        velocities: (N, 3) array
        masses: (N,) array
    
    Returns:
        array: COM velocity (3,)
    """
    total_mass = jnp.sum(masses)
    v_com = jnp.sum(masses[:, None] * velocities, axis=0) / total_mass
    return v_com

@jax.jit
def identify_bound_particles(positions, velocities, masses, G, eps2):
    """
    Identify gravitationally bound particles.
    
    Args:
        positions: (N, 3) array
        velocities: (N, 3) array
        masses: (N,) array
        G: gravitational constant
        eps2: softening squared
    
    Returns:
        boolean array: bound particles (N,)
    """
    N = len(masses)
    
    # Initial guess: all particles bound
    bound_mask = jnp.ones(N, dtype=bool)
    
    def iteration_step(i, bound_mask):
        """
        Single iteration of bound particle identification.
        
        Args:
            i: iteration index
            bound_mask: current bound particle mask from previous iteration
        
        Returns:
            new_bound_mask: updated mask for next iteration
        """
        # Step 1: Compute COM velocity of bound particles
        bound_masses = masses * bound_mask
        bound_vels = velocities * bound_mask[:, None]
        total_bound_mass = jnp.sum(bound_masses)
        
        # Handle edge case: no bound particles (use jnp.where, not if)
        v_com = jnp.where(
            total_bound_mass > 0,
            # If bound particles exist: use bound COM
            jnp.sum(bound_masses[:, None] * bound_vels, axis=0) / total_bound_mass,
            # Else: use full system COM (shouldn't happen, but be safe)
            jnp.sum(masses[:, None] * velocities, axis=0) / jnp.sum(masses)
        )
        
        # Step 2: Compute kinetic energy (relative to COM)
        v_rel = velocities - v_com[None, :]
        ke_specific = 0.5 * jnp.sum(v_rel**2, axis=1)  # (N,)
        
        # Step 3: Compute potential energy
        # Vectorize over all particles
        phi = jax.vmap(
            compute_potential_at_particle,
            in_axes=(0, None, None, None, None)
        )(jnp.arange(N), positions, masses, G, eps2)
        
        # Step 4: Specific energy
        epsilon = ke_specific + phi
        
        # Step 5: Update bound mask
        new_bound_mask = epsilon < 0
        
        return new_bound_mask
    
    # Run the loop: iterate from 0 to 10 (exclusive)
    # This calls iteration_step(0, bound_mask), then iteration_step(1, result), etc.
    bound_mask = jax.lax.fori_loop(
        0,                  # lower: start at iteration 0
        10,                 # upper: stop before iteration 10 (so 0-9)
        iteration_step,     # body_fun: function to call each iteration
        bound_mask          # init_val: initial value to carry through
    )
    
    return bound_mask


def compute_bound_mass_fraction(positions, velocities, masses, G=1.0, eps2=1e-20):
    """
    Compute bound mass fraction.
    
    f_bound = sum(m_i for bound particles) / sum(all m_i)
    
    Args:
        positions: (N, 3) final positions
        velocities: (N, 3) final velocities
        masses: (N,) particle masses
        G: gravitational constant
        eps2: softening length squared
    
    Returns:
        float: bound mass fraction in [0, 1]
    """
    bound_mask = identify_bound_particles(positions, velocities, masses, G, eps2)
    
    bound_mass = jnp.sum(masses * bound_mask)
    total_mass = jnp.sum(masses)
    
    return bound_mass / total_mass

def compute_velocity_dispersion(positions, velocities, masses, G=1.0, eps2=1e-20):
    """
    Compute mass-weighted velocity dispersion for bound particles.
    
    sigma_v^2 = sum(m_i * |v_i - v_COM|^2) / sum(m_i)  [bound particles only]
    
    Note: v_COM is computed from bound particles only
    
    Args:
        positions: (N, 3) final positions
        velocities: (N, 3) final velocities  
        masses: (N,) particle masses
        G: gravitational constant
        eps2: softening length squared
    
    Returns:
        float: velocity dispersion (km/s or code units)
    """
    bound_mask = identify_bound_particles(positions, velocities, masses, G, eps2)
    
    # Handle case of no bound particles
    n_bound = jnp.sum(bound_mask)
    if n_bound == 0:
        return 0.0
    
    # Extract bound particles
    bound_masses = masses[bound_mask]
    bound_vels = velocities[bound_mask]
    
    # Compute COM velocity of bound population
    v_com = compute_center_of_mass_velocity(bound_vels, bound_masses)
    
    # Compute mass-weighted velocity dispersion
    v_rel = bound_vels - v_com[None, :]
    v_squared = jnp.sum(v_rel**2, axis=1)  # |v_i - v_COM|^2
    
    sigma_v_squared = jnp.sum(bound_masses * v_squared) / jnp.sum(bound_masses)
    
    return jnp.sqrt(sigma_v_squared)


def compute_center_of_mass_position(positions, masses, bound_mask):
    """
    Compute center-of-mass position for bound particles.
    
    r_COM = sum(m_i * r_i) / sum(m_i)  [bound particles only]
    
    Args:
        positions: (N, 3) array
        masses: (N,) array
        bound_mask: (N,) boolean array
    
    Returns:
        (3,) array: COM position
    """
    bound_masses = masses * bound_mask
    bound_positions = positions * bound_mask[:, None]
    
    total_bound_mass = jnp.sum(bound_masses)
    r_com = jnp.sum(bound_masses[:, None] * bound_positions, axis=0) / total_bound_mass
    
    return r_com


def compute_half_mass_radius(positions, velocities, masses, G=1.0, eps2=1e-20):
    """
    Compute half-mass radius for bound particles.
    
    r_h is the radius from the COM containing half the bound mass.
    
    Algorithm:
    1. Identify bound particles
    2. Compute COM of bound particles
    3. Compute distances from COM
    4. Sort by distance
    5. Find radius enclosing half the bound mass
    
    Args:
        positions: (N, 3) final positions
        velocities: (N, 3) final velocities
        masses: (N,) particle masses
        G: gravitational constant
        eps2: softening length squared
    
    Returns:
        float: half-mass radius (AU or code units)
    """
    bound_mask = identify_bound_particles(positions, velocities, masses, G, eps2)
    
    # Handle case of no bound particles
    n_bound = jnp.sum(bound_mask)
    if n_bound == 0:
        return 0.0
    
    # Extract bound particles
    bound_positions = positions[bound_mask]
    bound_masses = masses[bound_mask]
    
    # Compute COM of bound population
    r_com = compute_center_of_mass_position(positions, masses, bound_mask)
    
    # Compute distances from COM
    r_vec = bound_positions - r_com[None, :]
    distances = jnp.sqrt(jnp.sum(r_vec**2, axis=1))
    
    # Sort particles by distance
    sort_indices = jnp.argsort(distances)
    sorted_distances = distances[sort_indices]
    sorted_masses = bound_masses[sort_indices]
    
    # Compute cumulative mass
    cumulative_mass = jnp.cumsum(sorted_masses)
    total_bound_mass = jnp.sum(bound_masses)
    half_mass = 0.5 * total_bound_mass
    
    # Find first particle where cumulative mass >= half mass
    # Using searchsorted for clean implementation
    half_mass_index = jnp.searchsorted(cumulative_mass, half_mass)
    
    # Handle edge case where index equals length
    half_mass_index = jnp.minimum(half_mass_index, len(sorted_distances) - 1)
    
    r_h = sorted_distances[half_mass_index]
    
    return r_h


def compute_all_summary_stats(positions, velocities, masses, G=1.0, eps2=1e-20):
    """
    Compute all three summary statistics from simulation final state.
    
    Args:
        positions: (N, 3) final positions
        velocities: (N, 3) final velocities
        masses: (N,) particle masses
        G: gravitational constant
        eps2: softening length squared
    
    Returns:
        dict with keys:
            'f_bound': bound mass fraction
            'sigma_v': velocity dispersion
            'r_h': half-mass radius
    """
    # Compute bound particles once and reuse
    bound_mask = identify_bound_particles(positions, velocities, masses, G, eps2)
    
    # Bound mass fraction
    bound_mass = jnp.sum(masses * bound_mask)
    total_mass = jnp.sum(masses)
    f_bound = bound_mass / total_mass
    
    # Velocity dispersion
    n_bound = jnp.sum(bound_mask)
    if n_bound > 0:
        bound_masses = masses[bound_mask]
        bound_vels = velocities[bound_mask]
        v_com = compute_center_of_mass_velocity(bound_vels, bound_masses)
        v_rel = bound_vels - v_com[None, :]
        v_squared = jnp.sum(v_rel**2, axis=1)
        sigma_v_squared = jnp.sum(bound_masses * v_squared) / jnp.sum(bound_masses)
        sigma_v = jnp.sqrt(sigma_v_squared)
    else:
        sigma_v = 0.0
    
    # Half-mass radius
    if n_bound > 0:
        bound_positions = positions[bound_mask]
        bound_masses_for_rh = masses[bound_mask]
        r_com = compute_center_of_mass_position(positions, masses, bound_mask)
        r_vec = bound_positions - r_com[None, :]
        distances = jnp.sqrt(jnp.sum(r_vec**2, axis=1))
        sort_indices = jnp.argsort(distances)
        sorted_distances = distances[sort_indices]
        sorted_masses = bound_masses_for_rh[sort_indices]
        cumulative_mass = jnp.cumsum(sorted_masses)
        half_mass = 0.5 * bound_mass
        half_mass_index = jnp.searchsorted(cumulative_mass, half_mass)
        half_mass_index = jnp.minimum(half_mass_index, len(sorted_distances) - 1)
        r_h = sorted_distances[half_mass_index]
    else:
        r_h = 0.0
    
    return {
        'f_bound': float(f_bound),
        'sigma_v': float(sigma_v),
        'r_h': float(r_h)
    }


def sanity_check_summary_stats(stats_dict):
    """
    Verify summary statistics are within expected ranges.
    
    Expected ranges (from project spec):
    - f_bound: 0.3 - 1.0
    - sigma_v: 0.5 - 5 (AU/yr or code units)
    - r_h: 20 - 150 (AU or code units)
    
    Args:
        stats_dict: dictionary with 'f_bound', 'sigma_v', 'r_h'
    
    Returns:
        dict with 'in_range' (bool) and 'warnings' (list of strings)
    """
    warnings = []
    
    f_bound = stats_dict['f_bound']
    sigma_v = stats_dict['sigma_v']
    r_h = stats_dict['r_h']
    
    # Check f_bound
    if not (0.0 <= f_bound <= 1.0):
        warnings.append(f"f_bound = {f_bound:.3f} outside [0, 1] - CRITICAL BUG!")
    elif not (0.3 <= f_bound <= 1.0):
        warnings.append(f"f_bound = {f_bound:.3f} outside expected [0.3, 1.0] - may be extreme case")
    
    # Check sigma_v
    if sigma_v < 0:
        warnings.append(f"sigma_v = {sigma_v:.3f} is negative - CRITICAL BUG!")
    elif not (0.5 <= sigma_v <= 5.0):
        warnings.append(f"sigma_v = {sigma_v:.3f} outside expected [0.5, 5.0] - may be extreme case")
    
    # Check r_h
    if r_h < 0:
        warnings.append(f"r_h = {r_h:.3f} is negative - CRITICAL BUG!")
    elif not (20 <= r_h <= 150):
        warnings.append(f"r_h = {r_h:.3f} outside expected [20, 150] - may be extreme case")
    
    in_range = len([w for w in warnings if "CRITICAL" in w]) == 0
    
    return {
        'in_range': in_range,
        'warnings': warnings if warnings else ['All checks passed!']
    }