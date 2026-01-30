# Early Detection of Memorization in Diffusion Models

Official implementation (ICML 2026).

## 📋 Overview
Our key contributions include:

- **Metric 2 (Hessian Difference)**: A novel curvature-based metric achieving **97.59% AUC** on SD 1.5
- **DSM (Dynamical Singularity Metric)**: Tracks score dynamics across denoising steps
- **Comprehensive Evaluation**: Tested on N=500 memorized + 500 non-memorized prompts

## 🎯 Main Results (N=500+500)

| Method | AUC | TPR @ 1% FPR |
|--------|-----|--------------|
| **Metric 2 (Ours)** | **0.9759** | 82.8% |
| **DSM (Step 1→2)** | **0.9745** | **85.8%** |
| DSM (Step 0→1) | 0.9647 | 83.4% |
| Jeon et al. | 0.9642 | 80.6% |
| Wen et al. (Step 1) | 0.9621 | 79.0% |

**Model**: Stable Diffusion 1.5  
**Evaluation**: 500 memorized + 500 non-memorized prompts

## 🚀 Quick Start

### Installation

git clone https://github.com/anonymous-submission/diffusion-memorization-detection.git
cd diffusion-memorization-detection

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_installation.py
```

### Quick Demo (3 minutes)

Test the code with included sample data (10 samples):

```bash
# 1. Run detection
python detect.py \
    --model_id "runwayml/stable-diffusion-v1-5" \
    --mem_dataset "data/memorized_prompts_sample.jsonl" \
    --nonmem_dataset "data/non_memorized_prompts_sample.jsonl" \
    --output_dir "results/demo/"

# 2. Analyze results
python analyze.py \
    --results_dir "results/demo/" \
    --plot_roc \
    --plot_distributions
```

## 🔄 Standard Workflow

The repository is organized into three stages:

```mermaid
graph TD
    A[Dataset] --> B(detect.py: Main Experiment)
    B --> C(analyze.py: Analysis & Plots)
    B --> D(vis_1step_prediction.py: Prediction Viz)
    
    A --> E(collect_wen_timeseries.py: Collect 50-step Data)
    E --> F(plot_scaling_law.py: Scaling Law Plot)
    
    style B fill:#e1f5fe,stroke:#01579b
    style C fill:#e8f5e9,stroke:#2e7d32
    style E fill:#fff3e0,stroke:#ef6c00
```

### 1️⃣ Main Experiment (Core Detection)
Calculate all detection metrics (Metric 2, DSM, Wen, Jeon).

```bash
python detect.py \
    --model_id "runwayml/stable-diffusion-v1-5" \
    --mem_dataset "path/to/memorized_500.jsonl" \
    --nonmem_dataset "path/to/non_memorized_500.jsonl" \
    --output_dir "results/paper_results/"
```
*Time: ~4.5 hours for 500+500 samples on RTX 3090.*

### 2️⃣ Analysis (Results & Plots)
Generate AUC tables, ROC curves, and score distributions.

```bash
python analyze.py --results_dir "results/paper_results/" --plot_roc --plot_distributions
```

### 3️⃣ Paper Visualization (Optional)
Generate specific figures for the paper.

#### A. Prediction Visualization (Figure 1/2)
Directly uses main experiment output.
```bash
python vis_1step_prediction.py \
    --mem_results "results/paper_results/memorized_results.jsonl" \
    --nonmem_results "results/paper_results/non_memorized_results.jsonl"
```

#### B. Scaling Law Plot (Figure 3)
Requires separate data collection for all 50 timesteps.
```bash
# 1. Collect data (~2.5 hours)
python collect_wen_timeseries.py \
    --mem_dataset "path/to/memorized.jsonl" \
    --nonmem_dataset "path/to/non_memorized.jsonl" \
    --limit 500

# 2. Plot
python plot_scaling_law.py
```

## 📦 Repository Structure

```
diffusion-memorization-detection/
├── detect.py                    # Main detection script
├── analyze.py                   # Result analysis script
├── test_installation.py         # Installation verification
│
├── vis_1step_prediction.py      # Visualization script (Step 1 prediction)
├── collect_wen_timeseries.py    # Data collection for Scaling Law
├── plot_scaling_law.py          # Plotting script for Scaling Law
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── data/
│   ├── memorized_prompts_sample.jsonl     # Example memorized prompts
│   └── non_memorized_prompts_sample.jsonl # Example non-memorized prompts
│
├── src/
│   ├── detect_all_metrics.py    # Core metric implementations (6 metrics)
│   └── analysis_utils.py        # Analysis and plotting utilities
│
└── results/
    └── .gitkeep                 # Output directory
```

## 🔬 Metrics Implemented

### 1. Metric 2 (Hessian Difference)
Measures the difference in curvature between conditional and unconditional score functions:
`Metric2 = ||∇(0.5||s_c||²) - ∇(0.5||s_u||²)||²`

### 2. DSM (Dynamical Singularity Metric)
Tracks the rate of change of conditional scores across denoising steps:
`DSM_{t→t+1} = ||s_c(x_{t+1}) - s_c(x_t)||²`

### 3. Baseline Methods
- **Jeon et al.**: Hessian-vector product metric
- **Wen et al.**: Guidance force magnitude

## 🛠️ Technical Details

### Hardware Requirements
- GPU: NVIDIA GPU with ≥12GB VRAM (tested on RTX 3090)
- RAM: ≥16GB
- Storage: ≥10GB for model weights

### Adjustable Parameters
- `--limit N`: Use only first N samples (e.g., `--limit 100`)
- `--num_inference_steps N`: Change denoising steps (default: 50)
- `--batch_size N`: Currently only batch_size=1 is supported due to gradient computation

## 📧 Contact
For questions or issues, please open an issue on GitHub.
