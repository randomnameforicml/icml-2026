
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_prompts(filepath, n=4):
    """Load prompts from main experiment output (memorized_results.jsonl or non_memorized_results.jsonl)."""
    prompts = []
    with open(filepath, 'r') as f:
        for line in f:
            if len(prompts) >= n:
                break
            if line.strip():
                item = json.loads(line)
                prompts.append(item['prompt'])
    return prompts

def decode_latents(vae, latents):
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().detach().permute(0, 2, 3, 1).float().numpy()[0]
    return image

@torch.no_grad()
def get_x0_prediction(scheduler, latents, noise_pred, t):
    alpha_prod_t = scheduler.alphas_cumprod[t]
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
    return pred_original_sample

def main():
    parser = argparse.ArgumentParser(description='Visualize first-step predictions')
    parser.add_argument('--mem_results', type=str, default='results/paper_results/memorized_results.jsonl',
                      help='Path to memorized results from detect.py')
    parser.add_argument('--nonmem_results', type=str, default='results/paper_results/non_memorized_results.jsonl',
                      help='Path to non-memorized results from detect.py')
    parser.add_argument('--n_samples', type=int, default=4,
                      help='Number of samples to visualize per category')
    parser.add_argument('--output_dir', type=str, default='results/visualizations',
                      help='Output directory for visualizations')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5',
                      help='Model ID or path')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Prompts from main experiment results
    print(f"\nLoading prompts from main experiment outputs...")
    mem_prompts = load_prompts(args.mem_results, args.n_samples)
    gen_prompts = load_prompts(args.nonmem_results, args.n_samples)
    
    print("\n" + "="*50)
    print("SELECTED PROMPTS FOR VISUALIZATION:")
    print("="*50)
    print(f"Memorized (n={len(mem_prompts)}):")
    for i, p in enumerate(mem_prompts):
        print(f"  {i+1}. {p[:80]}..." if len(p) > 80 else f"  {i+1}. {p}")
        
    print(f"\nNon-Memorized (n={len(gen_prompts)}):")
    for i, p in enumerate(gen_prompts):
        print(f"  {i+1}. {p[:80]}..." if len(p) > 80 else f"  {i+1}. {p}")
    print("="*50 + "\n")

    # 2. Load Model
    print(f"Loading model: {args.model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
        
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    
    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Generate
    generator = torch.Generator(device).manual_seed(42) # Fixed seed for consistency
    pipe.scheduler.set_timesteps(50)
    
    def run_viz(prompts, prefix):
        for i, prompt in enumerate(prompts):
            # Encode Text
            text_input = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]
            uncond_input = pipe.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
            # Init Latents
            latents = torch.randn(
                (1, pipe.unet.config.in_channels, 512 // 8, 512 // 8),
                generator=generator, device=device, dtype=pipe.unet.dtype
            )
            
            # Step 1 (t=981 in DDIM 50? First step)
            t = pipe.scheduler.timesteps[0]
            
            # Predict Noise
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
            # Predict x0
            x0 = get_x0_prediction(pipe.scheduler, latents, noise_pred, t)
            img = decode_latents(pipe.vae, x0)
            
            # Save
            fname = os.path.join(args.output_dir, f"{prefix}_{i+1}.png")
            plt.imsave(fname, img)
            print(f"Saved {fname}")

    print("Generating Memorized...")
    run_viz(mem_prompts, "mem")
    
    print("Generating Generalized...")
    run_viz(gen_prompts, "gen")
    
    print("\nVisualization Complete.")

if __name__ == "__main__":
    main()
