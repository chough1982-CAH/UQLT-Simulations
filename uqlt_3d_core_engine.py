import numpy as np

# UQLT Constants (from your descriptions: threshold 6.64e-27 kg, migration 0.05, EMN 0.02, center bias 0.01/dist)
THRESHOLD = 6.64e-27
MIGRATION_FACTOR = 0.05
EMN_PRESSURE_FACTOR = 0.02
CENTER_BIAS_FACTOR = 0.01
MIN_MASS_CLIP = 1e-30

def run_uqlt_3d_grid(grid_size=7, initial_mass=1e-26, max_steps=500, tol=1e-30):
    """
    UQLT 3D grid simulation from uniform start.
    - Valignity: neighbor average migration + center bias
    - EMN: outward pressure step-down
    - ChronoCollapse: halve mass > threshold
    Returns: final masses array, steps taken, history of max mass
    """
    masses = np.full((grid_size, grid_size, grid_size), initial_mass, dtype=np.float64)
    center = np.array([grid_size//2] * 3)
    
    history_max = []
    prev_max = np.inf
    
    for step in range(max_steps):
        new_m = masses.copy()
        
        # Valignity migration (vectorized where possible, but loops for 6-neighbor simplicity)
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    pos = np.array([x, y, z])
                    neighbors = []
                    for d in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                        nx, ny, nz = pos + d
                        if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
                            neighbors.append(masses[nx, ny, nz])
                    if neighbors:
                        avg = np.mean(neighbors)
                        migration = MIGRATION_FACTOR * (avg - masses[x,y,z])
                        dist = np.linalg.norm(pos - center)
                        if dist > 0:
                            migration += CENTER_BIAS_FACTOR / dist
                        new_m[x,y,z] += migration
        
        # EMN pressure outward
        new_m -= EMN_PRESSURE_FACTOR * masses * 0.5
        
        # Clip negatives/infinities
        new_m = np.clip(new_m, MIN_MASS_CLIP, np.inf)
        
        # ChronoCollapse halving
        new_m[new_m > THRESHOLD] /= 2
        
        masses = new_m
        
        current_max = np.max(masses)
        history_max.append(current_max)
        
        # Early stop if stabilized
        if step > 10 and abs(current_max - prev_max) < tol:
            break
        prev_max = current_max
    
    return masses, step + 1, history_max

# Run example
if __name__ == "__main__":
    final_masses, steps, hist_max = run_uqlt_3d_grid(grid_size=7)
    
    print(f"Steps to convergence: {steps}")
    print(f"Final max mass: {np.max(final_masses):.8f}")
    
    # Central slice (z = center)
    center_z = 7 // 2
    central_slice = final_masses[:, :, center_z]
    print("Central slice (xy at z=center, rounded to 8 decimals):")
    print(np.round(central_slice, 8))
    
    print(f"Outer edge mass (example corner): {final_masses[0,0,0]:.2e}")
