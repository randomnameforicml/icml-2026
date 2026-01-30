"""
This script generates the data required by plot_scaling_law.py.

Usage:
    python collect_wen_timeseries.py \
        --model_id "runwayml/stable-diffusion-v1-5" \
        --mem_dataset "data/memorized_500.jsonl" \
        --nonmem_dataset "data/non_memorized_500.jsonl" \
        --output_dir "data/"
"""

import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

def get_epsilon(prediction_type, noise_pred, t, latents, scheduler):
    """Convert prediction to epsilon."""
    if prediction_type == "v_prediction":
        alpha_prod_t = scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        alpha_t = alpha_prod_t ** 0.5
        sigma_t = beta_prod_t ** 0.5
        return alpha_t * noise_pred + sigma_t * latents
    elif prediction_type == "epsilon":
        return noise_pred
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

def collect_wen_timeseries(pipe, prompt, num_steps=50, seed=42):
    """Collect Wen metric ||s_c - s_u|| across all timesteps."""
    device = pipe.device
    dtype = pipe.unet.dtype
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_steps)
    timesteps = pipe.scheduler.timesteps
    
    # Initialize latents
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        generator=generator,
        device=device,
        dtype=dtype
    )
    
    # Encode text
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_emb = pipe.text_encoder(text_input.input_ids.to(device))[0]
    
    uncond_input = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_emb = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
    
    # Collect ||s_c - s_u|| at each timestep
    wen_values = []
    
    for i, t in enumerate(timesteps):
        with torch.no_grad():
            # Conditional prediction
            pred_c = pipe.unet(latents, t, encoder_hidden_states=text_emb).sample
            eps_c = get_epsilon(pipe.scheduler.config.prediction_type, pred_c, t, latents, pipe.scheduler)
            
            # Unconditional prediction
            pred_u = pipe.unet(latents, t, encoder_hidden_states=uncond_emb).sample
            eps_u = get_epsilon(pipe.scheduler.config.prediction_type, pred_u, t, latents, pipe.scheduler)
            
            # ||s_c - s_u||
            wen = torch.sum((eps_c - eps_u) ** 2).sqrt().item()
            wen_values.append(wen)
            
            # Step forward with CFG
            if i < len(timesteps) - 1:
                guidance_scale = 7.5
                noise_pred = pred_u + guidance_scale * (pred_c - pred_u)
                step_output = pipe.scheduler.step(noise_pred, t, latents)
                latents = step_output.prev_sample
    
    return wen_values

def main():
    parser = argparse.ArgumentParser(description='Collect Wen metric timeseries for scaling law')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--mem_dataset', type=str, required=True)
    parser.add_argument('--nonmem_dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {args.model_id}")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Process memorized
    print("\nProcessing MEMORIZED samples...")
    mem_prompts = []
    with open(args.mem_dataset, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    mem_prompts.append(item.get('prompt', item.get('text', '')))
                except:
                    pass
    
    if args.limit > 0:
        mem_prompts = mem_prompts[:args.limit]
    
    mem_output = []
    for i, prompt in enumerate(tqdm(mem_prompts, desc="Memorized")):
        wen_ts = collect_wen_timeseries(pipe, prompt, args.num_steps, args.seed + i)
        mem_output.append({"delta_s": wen_ts})
    
    # Process non-memorized
    print("\nProcessing NON-MEMORIZED samples...")
    nonmem_prompts = []
    with open(args.nonmem_dataset, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    nonmem_prompts.append(item.get('prompt', item.get('text', '')))
                except:
                    pass
    
    if args.limit > 0:
        nonmem_prompts = nonmem_prompts[:args.limit]
    
    nonmem_output = []
    for i, prompt in enumerate(tqdm(nonmem_prompts, desc="Non-Memorized")):
        wen_ts = collect_wen_timeseries(pipe, prompt, args.num_steps, args.seed + len(mem_prompts) + i)
        nonmem_output.append({"delta_s": wen_ts})
    
    # Save
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    mem_out_path = os.path.join(args.output_dir, "force_mem_all.jsonl")
    nonmem_out_path = os.path.join(args.output_dir, "force_nonmem_all.jsonl")
    
    with open(mem_out_path, 'w') as f:
        for item in mem_output:
            f.write(json.dumps(item) + '\n')
    
    with open(nonmem_out_path, 'w') as f:
        for item in nonmem_output:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✓ Saved to:")
    print(f"  {mem_out_path}")
    print(f"  {nonmem_out_path}")
    print(f"\nNow you can run: python plot_scaling_law.py")

if __name__ == "__main__":
    main()
