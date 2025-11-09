# RTX 5090 FLUX Training Documentation - D:\Flux_Trainer Edition
## Complete Updated Documentation Package

---

## 📋 Overview

This package contains **fully updated documentation** for setting up an uncompromising RTX 5090 FLUX LoRA training environment with the training workspace relocated to **D:\Flux_Trainer** as requested.

### Key Changes from Original Documentation:

- ✅ **Training Environment**: Moved from `C:\AI\flux_training` to `D:\Flux_Trainer`
- ✅ **System Files**: Remain on C: drive (Python, CUDA, Visual Studio)
- ✅ **Build Directories**: Remain on C: drive (`C:\build`)
- ✅ **Models & Datasets**: Now stored in `D:\Flux_Trainer\models` and `D:\Flux_Trainer\dataset`
- ✅ **All Paths Updated**: Every script, config, and command updated with correct paths

---

## 📦 Updated Files Included

1. **`RTX5090_CUDA13_QUICK_COMMANDS_UPDATED.md`**
   - Quick command reference with D:\Flux_Trainer paths
   - Essential commands for experienced users
   - Complete directory structure diagram

2. **`AI_AGENT_RTX5090_SETUP_PROMPT_UPDATED.md`**
   - Fully automated setup prompt for AI agents
   - 12-phase execution plan
   - Creates D:\Flux_Trainer environment automatically
   - 4-6 hour unattended installation

3. **`RTX5090_FLUX_TRAINING_SUMMARY_UPDATED.md`**
   - Complete overview of the training system
   - Quick start paths
   - Configuration selector
   - Updated directory structure

4. **`FLUX_LORA_TRAINING_REFERENCE_UPDATED.md`**
   - Detailed training configurations
   - Face, Action, and Style LoRA settings
   - All example scripts use D:\Flux_Trainer paths
   - Dataset preparation for D:\Flux_Trainer\dataset

5. **`verify_flux_trainer_setup.py`**
   - Python verification script
   - Checks all paths and dependencies
   - Creates missing directories
   - Includes performance benchmark
   - Generates convenient launch script

---

## 🚀 Quick Start Guide

### Option 1: Automated Setup (Recommended)

1. **Ensure D: drive has 100GB+ free space**

2. **Copy the entire content from `AI_AGENT_RTX5090_SETUP_PROMPT_UPDATED.md`**

3. **Paste into your AI agent** (Claude, GPT-4, or automation tool)

4. **Let it run** (4-6 hours)

5. **Verify setup** by running:
   ```powershell
   python D:\verify_flux_trainer_setup.py
   ```

### Option 2: Manual Setup

1. **Follow `RTX5090_CUDA13_QUICK_COMMANDS_UPDATED.md`** for step-by-step instructions

2. **Create D:\Flux_Trainer directory structure**:
   ```powershell
   mkdir D:\Flux_Trainer
   mkdir D:\Flux_Trainer\models
   mkdir D:\Flux_Trainer\dataset
   mkdir D:\Flux_Trainer\output
   mkdir D:\Flux_Trainer\samples
   mkdir D:\Flux_Trainer\cuda_kernels
   ```

3. **Install system components on C: drive**:
   - Python 3.11.9 → C:\Python311
   - CUDA 13.0 → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
   - Build tools → C:\build

4. **Setup training environment on D: drive**:
   - SD-Scripts → D:\Flux_Trainer\sd-scripts-cuda13
   - Models → D:\Flux_Trainer\models
   - Custom kernels → D:\Flux_Trainer\cuda_kernels

---

## 📁 Final Directory Structure

```
C:\ (System Drive)
├── Python311\                     # Python 3.11.9
├── Program Files\
│   └── NVIDIA GPU Computing Toolkit\
│       └── CUDA\v13.0\           # CUDA 13.0
└── build\                        # PyTorch/xformers compilation
    ├── pytorch_rtx5090\
    └── xformers\

D:\ (Training Drive)
└── Flux_Trainer\                 # MAIN TRAINING ENVIRONMENT
    ├── models\                   # Flux model files (33GB total)
    │   ├── flux1-dev.safetensors (23GB)
    │   ├── ae.safetensors (335MB)
    │   ├── clip_l.safetensors (246MB)
    │   └── t5xxl_fp16.safetensors (9.5GB)
    ├── dataset\                  # Your training images
    │   └── [dataset_name]\
    │       └── [repeats]_[trigger]\
    │           ├── image.jpg
    │           └── image.txt
    ├── output\                   # Trained LoRA files
    ├── samples\                  # Generated samples during training
    ├── cuda_kernels\            # Custom Blackwell optimizations
    ├── sd-scripts-cuda13\       # Training scripts
    │   ├── venv\                # Python virtual environment
    │   ├── config_face.toml
    │   ├── config_action.toml
    │   └── config_style.toml
    ├── launch_training.bat      # Convenient launcher
    └── verify_setup.py          # Verification script
```

---

## 🎯 Training Workflow

1. **Prepare Dataset**:
   ```powershell
   # Place images in:
   D:\Flux_Trainer\dataset\[name]\[repeats]_[trigger]\
   
   # Example for face training:
   D:\Flux_Trainer\dataset\johnsmith\70_johnsmith\
   ```

2. **Create Captions**:
   ```powershell
   cd D:\Flux_Trainer\dataset\johnsmith\70_johnsmith
   Get-ChildItem *.jpg | ForEach-Object {
       Set-Content ($_.Name -replace '.jpg','.txt') "johnsmith"
   }
   ```

3. **Launch Training**:
   ```powershell
   # Option 1: Use the launcher
   D:\Flux_Trainer\launch_training.bat
   
   # Option 2: Manual launch
   cd D:\Flux_Trainer\sd-scripts-cuda13
   .\venv\Scripts\activate
   python flux_train_network.py --config_file config_face.toml --highvram
   ```

4. **Find Your LoRA**:
   ```powershell
   # Outputs saved to:
   D:\Flux_Trainer\output\flux_lora_*.safetensors
   ```

---

## ⚠️ Important Requirements

### Hardware:
- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **RAM**: 64GB recommended
- **Storage**: 
  - C: drive: 100GB free (system files + builds)
  - D: drive: 100GB free (training environment + models)

### Software:
- **Python**: EXACTLY 3.11.9 (not 3.11.8, not 3.11.10, not 3.12)
- **CUDA**: 13.0 (for native Blackwell sm_120 support)
- **Driver**: 581.57 or newer

---

## 🔧 Troubleshooting

### If D: drive doesn't exist:
You can modify the paths to use any drive, but you'll need to update all occurrences of `D:\Flux_Trainer` in:
- All config files
- All scripts
- Environment variables

### Quick Path Change:
```powershell
# Example: Change to E: drive
$oldPath = "D:/Flux_Trainer"
$newPath = "E:/Flux_Trainer"

Get-ChildItem -Path . -Filter "*.toml","*.py","*.md" -Recurse | ForEach-Object {
    (Get-Content $_.FullName) -replace [regex]::Escape($oldPath), $newPath | 
    Set-Content $_.FullName
}
```

### Verification Failed:
Run the verification script to identify issues:
```powershell
python verify_flux_trainer_setup.py
```

---

## 📝 Configuration Types

| LoRA Type | Config File | Network Dim | Steps | Time |
|-----------|------------|-------------|-------|------|
| Face Identity | config_face.toml | 128 | 1500 | 90 min |
| Action/Pose | config_action.toml | 48 | 1000 | 60 min |
| Style/Object | config_style.toml | 32 | 800 | 45 min |

---

## ✅ Success Checklist

Before training:
- [ ] D: drive has 100GB+ free space
- [ ] Python 3.11.9 installed at C:\Python311
- [ ] CUDA 13.0 configured
- [ ] PyTorch compiled with sm_120 support
- [ ] D:\Flux_Trainer directory created
- [ ] Models downloaded to D:\Flux_Trainer\models
- [ ] Dataset prepared in D:\Flux_Trainer\dataset
- [ ] Virtual environment activated
- [ ] 900+ TFLOPS benchmark passed

---

## 📞 Support

If you encounter issues:

1. **Check the verification script output** first
2. **Ensure all paths are correct** for your system
3. **Verify D: drive** exists and has sufficient space
4. **Review the troubleshooting** section in each document
5. **Ensure exact Python version** (3.11.9)

---

## 🎉 Ready to Train!

Once setup is complete, you'll have:
- **Zero warnings** during training
- **Native RTX 5090 performance** (900+ TFLOPS)
- **Perfect organization** with D:\Flux_Trainer
- **Easy management** of models, datasets, and outputs

Good luck with your FLUX LoRA training!

---

*Documentation Updated: November 2025*
*Training Environment: D:\Flux_Trainer*
*System: RTX 5090 + CUDA 13.0 + Python 3.11.9*
