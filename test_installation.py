"""
Quick test script to verify installation and basic functionality.
Run this after installing dependencies to check everything works.
"""

import sys
import importlib

def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    
    required = [
        'torch',
        'diffusers',
        'transformers',
        'numpy',
        'sklearn',
        'matplotlib',
        'tqdm'
    ]
    
    missing = []
    for package in required:
        try:
            importlib.import_module(package if package != 'sklearn' else 'sklearn')
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True

def check_cuda():
    """Check CUDA availability."""
    import torch
    print("\nChecking CUDA...")
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA is available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("  ⚠️  CUDA not available (will use CPU - much slower)")
        return False

def check_data():
    """Check if sample data exists."""
    import os
    print("\nChecking sample data...")
    
    data_files = [
        'data/memorized_prompts_sample.jsonl',
        'data/non_memorized_prompts_sample.jsonl'
    ]
    
    all_exist = True
    for f in data_files:
        if os.path.exists(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} - MISSING")
            all_exist = False
    
    if all_exist:
        print("\n✅ Sample data ready!")
    else:
        print("\n⚠️  Some sample data missing")
    
    return all_exist

def main():
    print("="*60)
    print("Diffusion Memorization Detection - Installation Test")
    print("="*60)
    print()
    
    deps_ok = check_dependencies()
    cuda_ok = check_cuda()
    data_ok = check_data()
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Dependencies: {'✅ OK' if deps_ok else '❌ FAILED'}")
    print(f"CUDA: {'✅ Available' if cuda_ok else '⚠️  CPU only'}")
    print(f"Sample Data: {'✅ OK' if data_ok else '⚠️  Missing'}")
    print()
    
    if deps_ok and data_ok:
        print("✅ Everything ready! You can now run:")
        print()
        print("  python detect.py --model_id \"runwayml/stable-diffusion-v1-5\" \\")
        print("                   --mem_dataset \"data/memorized_prompts_sample.jsonl\" \\")
        print("                   --nonmem_dataset \"data/non_memorized_prompts_sample.jsonl\" \\")
        print("                   --output_dir \"results/test/\" --limit 2")
        print()
    else:
        print("❌ Please fix the issues above before running the main scripts.")
    
    print("="*60)

if __name__ == "__main__":
    main()
