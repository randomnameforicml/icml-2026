
import torch
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

def get_epsilon(prediction_type, noise_pred, t, latents, scheduler):
    # Convert prediction to epsilon
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

def main(args):
    print(f"Loading SD 2.0 (Single File): {args.ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    print("DEBUG: Checking ckpt path extension...")
    if args.ckpt_path.endswith(".safetensors") or args.ckpt_path.endswith(".ckpt"):
        print(f"DEBUG: Loading from single file: {args.ckpt_path}")
        pipe = StableDiffusionPipeline.from_single_file(
            args.ckpt_path, 
            torch_dtype=dtype,
            load_safety_checker=False
        )
    else:
        print(f"DEBUG: Assuming HuggingFace Repo ID: {args.ckpt_path}")
        pipe = StableDiffusionPipeline.from_pretrained(
            args.ckpt_path,
            torch_dtype=dtype,
            safety_checker=None
        )
    print("DEBUG: Model loaded successfully.")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    print(f"Prediction Type: {pipe.scheduler.config.prediction_type}")

    # Dataset
    print(f"Loading dataset from {args.dataset}")
    dataset = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    dataset.append(item.get("prompt", item.get("text", "")))
                except: pass
    if args.limit > 0: dataset = dataset[:args.limit]
        
    os.makedirs("det_outputs", exist_ok=True)
    out_path = f"det_outputs/{args.run_name}.jsonl"
    print(f"Output to: {out_path}")
    
    # Timesteps for DSM (need 3 steps for DSM_01 and DSM_12)
    pipe.scheduler.set_timesteps(args.num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    t0 = timesteps[0] # Step 0 (e.g. 981)
    t1 = timesteps[1] # Step 1 (e.g. 961)
    t2 = timesteps[2] # Step 2 (e.g. 941)
    
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    
    for i, prompt in enumerate(tqdm(dataset)):
        # 1. Inputs
        generator = torch.Generator(device=device).manual_seed(args.gen_seed + i)
        latents = torch.randn((1, unet.config.in_channels, 64, 64), generator=generator, device=device, dtype=dtype)
        
        # 2. Embeddings
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_emb = text_encoder(text_input.input_ids.to(device))[0]
        
        uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        uncond_emb = text_encoder(uncond_input.input_ids.to(device))[0]
        
        # --- Step 0 Calculation (t=t0) ---
        latents.requires_grad_(True)
        
        # Conditional Pass
        pred_c = unet(latents, t0, encoder_hidden_states=text_emb).sample
        eps_c = get_epsilon(pipe.scheduler.config.prediction_type, pred_c, t0, latents, pipe.scheduler)
        
        # Unconditional Pass
        pred_u = unet(latents, t0, encoder_hidden_states=uncond_emb).sample
        eps_u = get_epsilon(pipe.scheduler.config.prediction_type, pred_u, t0, latents, pipe.scheduler)
        
        # Metric 1: Wen (Magnitude at Step 0)
        # Correct definition: ||s_c - s_u||^2 (Guidance Magnitude)
        wen_metric_0 = torch.sum((eps_c - eps_u) ** 2).item()
        
        # Gradients for Hessian Metrics
        # Your Metric 2: ||H_c s_c - H_u s_u||^2
        energy_c = 0.5 * torch.sum(eps_c ** 2)
        grad_c = torch.autograd.grad(energy_c, latents, retain_graph=True)[0] # H_c s_c
        
        energy_u = 0.5 * torch.sum(eps_u ** 2)
        grad_u = torch.autograd.grad(energy_u, latents, retain_graph=True)[0] # H_u s_u
        
        metric2_diff = grad_c - grad_u
        metric2_val = torch.sum(metric2_diff ** 2).item()
        
        # Jeon Metric (User Definition): || (H_c - H_u) (s_c - s_u) ||^2
        # Let v_diff = (s_c - s_u)
        # Term = (H_c - H_u) * v_diff = H_c v_diff - H_u v_diff
        # H_c v_diff = grad(s_c . v_diff)
        # H_u v_diff = grad(s_u . v_diff)
        
        s_diff = (eps_c - eps_u).detach() # v_diff
        
        # H_c * v_diff
        dot_c = torch.sum(eps_c * s_diff)
        hvp_c = torch.autograd.grad(dot_c, latents, retain_graph=True)[0]
        
        # H_u * v_diff
        dot_u = torch.sum(eps_u * s_diff)
        hvp_u = torch.autograd.grad(dot_u, latents, retain_graph=True)[0]
        
        jeon_vector = hvp_c - hvp_u
        jeon_metric = torch.sum(jeon_vector ** 2).item()
        
        # --- Step 0 -> Step 1 Transition for DSM ---
        latents.requires_grad_(False)
        
        # Guided Step (Standard Classifier-Free Guidance)
        guidance_scale = 7.5
        noise_pred_0 = pred_u + guidance_scale * (pred_c - pred_u)
        step_output = pipe.scheduler.step(noise_pred_0, t0, latents)
        latents_1 = step_output.prev_sample.detach()
        
        # --- Step 1 Calculation (t=t1) ---
        pred_c_1 = unet(latents_1, t1, encoder_hidden_states=text_emb).sample
        eps_c_1 = get_epsilon(pipe.scheduler.config.prediction_type, pred_c_1, t1, latents_1, pipe.scheduler)

        # Unconditional Pass at Step 1 (for Wen Step 1)
        pred_u_1 = unet(latents_1, t1, encoder_hidden_states=uncond_emb).sample
        eps_u_1 = get_epsilon(pipe.scheduler.config.prediction_type, pred_u_1, t1, latents_1, pipe.scheduler)
        
        # Metric 1 (Step 1): Wen (Magnitude at Step 1)
        wen_metric_1 = torch.sum((eps_c_1 - eps_u_1) ** 2).item()
        
        # Metric 3a: DSM_01 (Score Change from Step 0 to Step 1)
        dsm_diff_01 = eps_c_1 - eps_c.detach()
        dsm_metric_01 = torch.sum(dsm_diff_01 ** 2).item()
        
        # --- Step 1 -> Step 2 Transition for DSM_12 ---
        # Perform another guided step
        noise_pred_1 = pred_u_1 + guidance_scale * (pred_c_1 - pred_u_1)
        step_output_2 = pipe.scheduler.step(noise_pred_1, t1, latents_1)
        latents_2 = step_output_2.prev_sample.detach()
        
        # --- Step 2 Calculation (t=t2) ---
        pred_c_2 = unet(latents_2, t2, encoder_hidden_states=text_emb).sample
        eps_c_2 = get_epsilon(pipe.scheduler.config.prediction_type, pred_c_2, t2, latents_2, pipe.scheduler)
        
        # Metric 3b: DSM_12 (Score Change from Step 1 to Step 2)
        dsm_diff_12 = eps_c_2 - eps_c_1.detach()
        dsm_metric_12 = torch.sum(dsm_diff_12 ** 2).item()
        
        # Save All
        record = {
            "prompt": prompt,
            "wen_sq_0": wen_metric_0,       # Wen Step 0
            "wen_sq_1": wen_metric_1,       # Wen Step 1
            "jeon_sq": jeon_metric,         # Jeon (Modified)
            "dsm_sq_01": dsm_metric_01,     # DSM Step 0→1
            "dsm_sq_12": dsm_metric_12,     # DSM Step 1→2
            "metric2_sq": metric2_val       # Metric 2 (Hessian Diff)
        }
        
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--gen_seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    main(args)
