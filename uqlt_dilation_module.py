import numpy as np 

# UQLT Constants from your model
THRESHOLD = 6.64e-27

def compute_time_dilation(layers=10, base_density=1e-26, density_max_factor=2.0):
    """
    Time dilation module from UQLT.
    - τ rate = halving_steps / depth
    - δ dilation = 1 / τ (slower in deeper containment layers)
    Returns printed table of layer, density, τ, δ
    """
    density_gradient = np.linspace(base_density, THRESHOLD * density_max_factor, layers)
    
    halving_steps = np.where(density_gradient > THRESHOLD, 2, 1)
    depth = np.arange(1, layers + 1)
    tau = halving_steps / depth.astype(float)
    delta = 1 / tau
    
    print("Layer | Density (kg) | τ Rate | δ Dilation")
    print("-" * 45)
    for i in range(layers):
        print(f"{i+1:5} | {density_gradient[i]:.2e} | {tau[i]:.4f} | {delta[i]:.4f}")

if __name__ == "__main__":
    print("UQLT Time Dilation from Containment Depth")
    compute_time_dilation(layers=10)
