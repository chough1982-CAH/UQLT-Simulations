import numpy as np

# UQLT Constants (unchanged from core)
THRESHOLD = 6.64e-27
MIGRATION_FACTOR = 0.05
EMN_PRESSURE_FACTOR = 0.02
CENTER_BIAS_FACTOR = 0.01
MIN_MASS_CLIP = 1e-30

# C-Shift specific constants (from your containment curvature definition)
DEPTH_INTEGRAL_SCALE = 1.0

def compute_c_shift(masses, grid_size):
    """
    C-Shift redshift proxy from converged grid.
    Derived from outer-to-core density fade (EMN step-down) + cumulative depth.
    """
    center = grid_size // 2
    core_mass = masses[center, center, center]
    
    # Outer edge average (corners)
    outer_positions = [
        (0,0,0), (0,0,grid_size-1), (0,grid_size-1,0), (grid_size-1,0,0),
        (0,grid_size-1,grid_size-1), (grid_size-1,0,grid_size-1), (grid_size-1,grid_size-1,0),
        (grid_size-1,grid_size-1,grid_size-1)
    ]
    outer_masses = [masses[x,y,z] for x,y,z in outer_positions]
    outer_avg = np.mean(outer_masses)
    
    weakening_factor = outer_avg / core_mass if core_mass > 0 else 0
    
    # Cumulative depth integral (average slice mass along z)
    depth_integral = 0.0
    for z in range(grid_size):
        slice_mass = np.mean(masses[:, :, z])
        depth_integral += slice_mass * DEPTH_INTEGRAL_SCALE
    
    # C-Shift z proxy
    z_proxy = weakening_factor * (depth_integral / grid_size**3)
    
    return z_proxy, weakening_factor, depth_integral

# Minimal core engine stub (for standalone run; normally import from core file)
def run_uqlt_3d_grid(grid_size=7, initial_mass=1e-26, max_steps=500, tol=1e-30):
    masses = np.full((grid_size, grid_size, grid_size), initial_mass, dtype=np.float64)
    center = np.array([grid_size//2] * 3)
    
    history_max = []
    prev_max = np.inf
    
    for step in range(max_steps):
        new_m = masses.copy()
        
        # Valignity migration (simplified loop)
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
        
        new_m -= EMN_PRESSURE_FACTOR * masses * 0.5
        new_m = np.clip(new_m, MIN_MASS_CLIP, np.inf)
        new_m[new_m > THRESHOLD] /= 2
        masses = new_m
        
        current_max = np.max(masses)
        history_max.append(current_max)
        
        if step > 10 and abs(current_max - prev_max) < tol:
            break
        prev_max = current_max
    
    return masses, step + 1, history_max

if __name__ == "__main__":
    final_masses, steps, hist_max = run_uqlt_3d_grid(grid_size=7)
    z_proxy, weakening, depth_int = compute_c_shift(final_masses, 7)
    
    print(f"Steps: {steps}")
    print(f"Final max mass: {np.max(final_masses):.8f}")
    print(f"C-Shift z proxy: {z_proxy:.4f}")
    print(f"Weakening factor: {weakening:.4f}")
    print(f"Depth integral: {depth_int:.2f}")
