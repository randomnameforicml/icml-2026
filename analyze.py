"""
Analysis script for memorization detection results.
Computes AUC, TPR, and generates visualizations.

Usage:
    python analyze.py --results_dir "results/" --plot_roc --plot_distributions
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.analysis_utils import analyze_results, plot_roc_curves, plot_metric_distributions

def main():
    parser = argparse.ArgumentParser(description='Analyze detection results')
    
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing detection results')
    parser.add_argument('--plot_roc', action='store_true',
                      help='Generate ROC curve plots')
    parser.add_argument('--plot_distributions', action='store_true',
                      help='Generate metric distribution plots')
    parser.add_argument('--output_file', type=str, default='summary.txt',
                      help='Output file for text summary')
    
    args = parser.parse_args()
    
    mem_file = os.path.join(args.results_dir, "memorized_results.jsonl")
    nonmem_file = os.path.join(args.results_dir, "non_memorized_results.jsonl")
    
    if not os.path.exists(mem_file) or not os.path.exists(nonmem_file):
        print(f"Error: Results files not found in {args.results_dir}")
        print(f"Expected: memorized_results.jsonl and non_memorized_results.jsonl")
        return
    
    # Run analysis
    print("="*60)
    print("Analyzing detection results...")
    print("="*60)
    
    results = analyze_results(mem_file, nonmem_file)
    
    # Save summary
    output_path = os.path.join(args.results_dir, args.output_file)
    with open(output_path, 'w') as f:
        f.write(results['summary'])
    
    print(results['summary'])
    print(f"\n✓ Summary saved to: {output_path}")
    
    # Generate plots if requested
    if args.plot_roc:
        fig_dir = os.path.join(args.results_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        plot_roc_curves(results, save_path=os.path.join(fig_dir, "roc_curves.png"))
        print(f"✓ ROC curves saved to: {fig_dir}/roc_curves.png")
    
    if args.plot_distributions:
        fig_dir = os.path.join(args.results_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        plot_metric_distributions(results, save_path=os.path.join(fig_dir, "distributions.png"))
        print(f"✓ Distributions saved to: {fig_dir}/distributions.png")

if __name__ == "__main__":
    main()
