"""
Main detection script for memorization detection in diffusion models.
Computes all metrics (Metric 2, DSM, Jeon, Wen) on given datasets.

Usage:
    python detect.py --model_id "runwayml/stable-diffusion-v1-5" \
                     --mem_dataset "data/memorized_prompts.jsonl" \
                     --nonmem_dataset "data/non_memorized_prompts.jsonl" \
                     --output_dir "results/"
"""

import argparse
import os
import subprocess
import sys

def run_detection_script(model_id, dataset_path, output_path, limit, num_inference_steps, gen_seed):
    """Run the detection script on a dataset."""
    # Extract run_name from output_path
    run_name = os.path.splitext(os.path.basename(output_path))[0]
    
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "src", "detect_all_metrics.py"),
        "--ckpt_path", model_id,
        "--dataset", dataset_path,
        "--run_name", run_name,
        "--num_inference_steps", str(num_inference_steps),
        "--gen_seed", str(gen_seed)
    ]
    
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    
    # Run the command
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    
    if result.returncode != 0:
        print(f"Error: Detection failed with exit code {result.returncode}")
        sys.exit(1)
    
    # Move output file to the desired location
    src_file = os.path.join("det_outputs", f"{run_name}.jsonl")
    if os.path.exists(src_file) and src_file != output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.rename(src_file, output_path)

def main():
    parser = argparse.ArgumentParser(description='Detect memorization in diffusion models')
    
    # Model arguments
    parser.add_argument('--model_id', type=str, required=True,
                      help='HuggingFace model ID or path to local checkpoint')
    
    # Data arguments
    parser.add_argument('--mem_dataset', type=str, required=True,
                      help='Path to memorized prompts (JSONL format)')
    parser.add_argument('--nonmem_dataset', type=str, required=True,
                      help='Path to non-memorized prompts (JSONL format)')
    parser.add_argument('--limit', type=int, default=-1,
                      help='Limit number of samples to process (-1 for all)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results/',
                      help='Directory to save results')
    
    # Detection parameters
    parser.add_argument('--num_inference_steps', type=int, default=50,
                      help='Number of DDIM steps')
    parser.add_argument('--gen_seed', type=int, default=42,
                      help='Random seed for generation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run detection on memorized samples
    print("="*60)
    print("Processing MEMORIZED samples...")
    print("="*60)
    mem_output = os.path.join(args.output_dir, "memorized_results.jsonl")
    run_detection_script(
        model_id=args.model_id,
        dataset_path=args.mem_dataset,
        output_path=mem_output,
        limit=args.limit,
        num_inference_steps=args.num_inference_steps,
        gen_seed=args.gen_seed
    )
    
    # Run detection on non-memorized samples
    print("\n" + "="*60)
    print("Processing NON-MEMORIZED samples...")
    print("="*60)
    nonmem_output = os.path.join(args.output_dir, "non_memorized_results.jsonl")
    run_detection_script(
        model_id=args.model_id,
        dataset_path=args.nonmem_dataset,
        output_path=nonmem_output,
        limit=args.limit,
        num_inference_steps=args.num_inference_steps,
        gen_seed=args.gen_seed
    )
    
    print("\n" + "="*60)
    print("✓ Detection complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
