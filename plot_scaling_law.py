import json
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler
import os

def get_sigmas():
    # Helper to get sigmas for 50 steps
    # We use standard DDIM 50 steps sigmas for SD1.5
    model_id = "runwayml/stable-diffusion-v1-5"
    try:
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    except:
        # Fallback to local approximation or offline
        print("Warning: Could not load scheduler from HF. Using cached/approximate sigmas.")
        # Hardcoded linear beta schedule approximation for 50 steps if offline
        # But let's assume online or cache for submission
        return None 
        
    scheduler.set_timesteps(50)
    timesteps = scheduler.timesteps
    sigmas = []
    for t in timesteps:
        alpha_prod_t = scheduler.alphas_cumprod[t]
        sigma_t = (1 - alpha_prod_t) ** 0.5
        sigmas.append(sigma_t.item())
    return np.array(sigmas)

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            item = json.loads(line)
            # data structure: [[v], [v]...]
            vec = np.array(item['delta_s']).flatten()
            data.append(vec)
    return np.array(data)

def main():
    print("Loading data...")
    if not os.path.exists('data/force_mem_all.jsonl'):
        print("Data files not found. Please ensure force_mem_all.jsonl is in the data directory.")
        return

    mem_data = load_data('data/force_mem_all.jsonl')
    nonmem_data = load_data('data/force_nonmem_all.jsonl')
    
    print(f"Memorized Shape: {mem_data.shape}")
    print(f"Generalized Shape: {nonmem_data.shape}")
    
    sigmas = get_sigmas()
    if sigmas is None: return

    # Data Interpretation:
    # 'delta_s' is ||epsilon||. t-DSM metric is || d s / d t ||.
    # Relationship: s = - epsilon / sigma.
    
    # 1. Compute Mean ||epsilon||
    mem_eps = np.mean(mem_data, axis=0)
    gen_eps = np.mean(nonmem_data, axis=0)
    
    # 2. Recover Score Proxy S = eps / sigma
    mem_s = mem_eps / sigmas
    gen_s = gen_eps / sigmas
    
    # 3. Calculate Time Derivative || dS/dt || ~ |dS/dsigma| * (1/sigma)
    d_mem_s = mem_s[1:] - mem_s[:-1]
    d_gen_s = gen_s[1:] - gen_s[:-1]
    d_sigma = sigmas[1:] - sigmas[:-1]
    
    sigmas_mid = (sigmas[1:] + sigmas[:-1]) / 2
    
    mem_deriv_sigma = np.abs(d_mem_s / d_sigma)
    gen_deriv_sigma = np.abs(d_gen_s / d_sigma)
    
    # t-DSM proxy
    mem_tdsm = mem_deriv_sigma * (1.0 / sigmas_mid)
    gen_tdsm = gen_deriv_sigma * (1.0 / sigmas_mid)
    
    # --- Regression ---
    # User requested late stage. DDIM 50 is sparse: 0.04, 0.14, 0.19, 0.24
    # Extend to 0.25 to get ~4 points.
    reg_mask = (sigmas_mid >= 0.04) & (sigmas_mid <= 0.25)
    if np.sum(reg_mask) > 1:
        reg_x = np.log(sigmas_mid[reg_mask])
        
        slope_mem, _ = np.polyfit(reg_x, np.log(mem_tdsm[reg_mask]), 1)
        slope_gen, _ = np.polyfit(reg_x, np.log(gen_tdsm[reg_mask]), 1)
        print(f"Memorized Slope: {slope_mem:.4f}")
        print(f"Generalized Slope: {slope_gen:.4f}")
    else:
        print("Warning: Not enough points for regression.")
        slope_mem, slope_gen = 0.0, 0.0
    
    # Plot
    # Plot for ICML Paper
    # Use standard style settings
    plt.style.use('default') 
    # Try to use seaborn-paper style if available, else manually set fonts
    try:
        import seaborn as sns
        sns.set_context("paper", font_scale=1.5)
        sns.set_style("ticks")
    except:
        pass

    plt.figure(figsize=(6, 5)) # Compact size for column width
    
    # Plot Data
    plt.loglog(sigmas_mid, mem_tdsm, color='#d62728', linewidth=2.5, label='Memorized') # Professional Red
    plt.loglog(sigmas_mid, gen_tdsm, color='#1f77b4', linestyle='--', linewidth=2.5, label='Generalized') # Professional Blue
    
    # Add reference slopes
    # Position references to avoid clutter
    ref_x = np.linspace(0.06, 0.4, 100)
    
    # Calculate offset to place reference lines cleanly below the curves
    # At x=0.2, Memorized is around 200, Generalized around 20
    # Slope -4 line: y = k * x^-4. at x=0.2, y=625k. Let's make y=200 -> k=0.32
    # Slope -3 line: y = k * x^-3. at x=0.2, y=125k. Let's make y=20 -> k=0.16
    
    # Match scale visually (shifting them slightly for clarity)
    scale_gen = gen_tdsm[np.argmin(np.abs(sigmas_mid - 0.2))]
    scale_mem = mem_tdsm[np.argmin(np.abs(sigmas_mid - 0.2))]

    # Draw references distinct from data
    # Shifted down to be below the generalized curve
    plt.loglog(ref_x, 0.8 * ref_x**-3, 'k:', linewidth=2, label=r'$\propto \sigma^{-3}$')
    plt.loglog(ref_x, 1.2 * ref_x**-4, 'gray', linestyle='-.', linewidth=2, label=r'$\propto \sigma^{-4}$')
    
    plt.legend(fontsize=12, frameon=True, loc='upper left')
    plt.gca().invert_xaxis()
    
    # Labels with LaTeX
    plt.xlabel(r'$\sigma_t$', fontsize=16)
    plt.ylabel(r'$\|\partial_t \mathbf{s}(\mathbf{x}_t, t)\|$', fontsize=16)
    
    # Ticks formatting
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, which="major", ls="-", alpha=0.3)
    plt.grid(True, which="minor", ls=":", alpha=0.1)
    
    # Layout
    plt.tight_layout()
    
    out_path_png = 'results/figures/fig_tdsm_scaling_icml.png'
    out_path_pdf = 'results/figures/fig_tdsm_scaling_icml.pdf'
    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
    
    plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_pdf, bbox_inches='tight') # Vector format for paper
    print(f"Saved {out_path_png} and {out_path_pdf}")

if __name__ == "__main__":
    main()
