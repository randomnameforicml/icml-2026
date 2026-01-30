"""
Analysis utilities for computing AUC, TPR, and generating summary statistics.
"""

import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def load_results(mem_path, nonmem_path):
    """Load memorized and non-memorized results from JSONL files."""
    mem_data = [json.loads(line) for line in open(mem_path, encoding='utf-8')]
    nonmem_data = [json.loads(line) for line in open(nonmem_path, encoding='utf-8')]
    return mem_data, nonmem_data

def compute_metrics(mem_data, nonmem_data):
    """Compute AUC and TPR @ 1% FPR for all metrics."""
    
    metric_names = {
        'wen_sq_0': 'Wen Step 0',
        'wen_sq_1': 'Wen Step 1',
        'jeon_sq': 'Jeon Metric',
        'dsm_sq_01': 'DSM (0→1)',
        'dsm_sq_12': 'DSM (1→2)',
        'metric2_sq': 'Metric 2 (Hessian)'
    }
    
    y_true = [1] * len(mem_data) + [0] * len(nonmem_data)
    
    results = []
    for key, name in metric_names.items():
        mem_scores = [item[key] for item in mem_data]
        nonmem_scores = [item[key] for item in nonmem_data]
        all_scores = mem_scores + nonmem_scores
        
        auc = roc_auc_score(y_true, all_scores)
        
        fpr, tpr, thresholds = roc_curve(y_true, all_scores)
        idx = np.where(fpr <= 0.01)[0]
        tpr_at_1fpr = tpr[idx[-1]] if len(idx) > 0 else 0
        
        results.append({
            'name': name,
            'key': key,
            'auc': auc,
            'tpr_1fpr': tpr_at_1fpr,
            'mem_mean': np.mean(mem_scores),
            'mem_std': np.std(mem_scores),
            'nonmem_mean': np.mean(nonmem_scores),
            'nonmem_std': np.std(nonmem_scores),
            'fpr': fpr,
            'tpr': tpr
        })
    
    results.sort(key=lambda x: x['auc'], reverse=True)
    return results

def format_summary(results, n_mem, n_nonmem):
    """Format results as a text summary."""
    lines = []
    lines.append("="*70)
    lines.append(f"MEMORIZATION DETECTION RESULTS (N={n_mem}+{n_nonmem})")
    lines.append("="*70)
    lines.append("")
    lines.append(f"{'Metric':<25} {'AUC':<8} {'TPR@1%FPR':<12} {'Mem Mean±Std':<20} {'NonMem Mean±Std'}")
    lines.append("-"*90)
    
    for r in results:
        lines.append(f"{r['name']:<25} {r['auc']:.4f}   {r['tpr_1fpr']:.4f}       "
                    f"{r['mem_mean']:>7.2f}±{r['mem_std']:<7.2f}   {r['nonmem_mean']:>7.2f}±{r['nonmem_std']:<7.2f}")
    
    lines.append("")
    lines.append("="*70)
    lines.append("Top 3 Methods:")
    lines.append("="*70)
    for i, r in enumerate(results[:3], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        lines.append(f"{medal} {r['name']:<25} AUC = {r['auc']:.4f}, TPR@1%FPR = {r['tpr_1fpr']:.4f}")
    
    lines.append("")
    lines.append("="*70)
    
    return "\n".join(lines)

def analyze_results(mem_path, nonmem_path):
    """Main analysis function."""
    mem_data, nonmem_data = load_results(mem_path, nonmem_path)
    results = compute_metrics(mem_data, nonmem_data)
    summary = format_summary(results, len(mem_data), len(nonmem_data))
    
    return {
        'results': results,
        'summary': summary,
        'mem_data': mem_data,
        'nonmem_data': nonmem_data
    }

def plot_roc_curves(analysis_results, save_path=None):
    """Generate ROC curve plot for all metrics."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    
    for r in analysis_results['results']:
        plt.plot(r['fpr'], r['tpr'], label=f"{r['name']} (AUC={r['auc']:.4f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for Memorization Detection', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_metric_distributions(analysis_results, save_path=None):
    """Generate distribution plots for all metrics."""
    import matplotlib.pyplot as plt
    
    results = analysis_results['results']
    mem_data = analysis_results['mem_data']
    nonmem_data = analysis_results['nonmem_data']
    
    n_metrics = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, r in enumerate(results):
        ax = axes[i]
        
        mem_scores = [item[r['key']] for item in mem_data]
        nonmem_scores = [item[r['key']] for item in nonmem_data]
        
        ax.hist(nonmem_scores, bins=30, alpha=0.6, label='Non-Memorized', color='blue', density=True)
        ax.hist(mem_scores, bins=30, alpha=0.6, label='Memorized', color='red', density=True)
        
        ax.set_xlabel('Score', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f"{r['name']} (AUC={r['auc']:.4f})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
