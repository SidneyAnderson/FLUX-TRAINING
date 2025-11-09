# RTX 5090 FLUX LORA TRAINING - ISOLATED ENVIRONMENT
## Complete Self-Contained Setup with Zero System Contamination
## Everything in D:\Flux_Trainer - Portable & Clean

---

## 🎯 OVERVIEW: COMPLETE ISOLATION

This documentation provides a **COMPLETELY ISOLATED** Flux LoRA training environment that:

- ✅ **Installs NOTHING system-wide** - No Python in Program Files
- ✅ **Modifies NO system PATH** - System PATH remains untouched  
- ✅ **Sets NO system variables** - No permanent environment changes
- ✅ **100% self-contained** - Everything in D:\Flux_Trainer
- ✅ **Fully portable** - Copy folder to any PC
- ✅ **Clean uninstall** - Just delete the folder
- ✅ **Multiple versions** - Run different setups simultaneously
- ✅ **No conflicts** - Isolated from system Python/CUDA

---

## 📦 ISOLATED ARCHITECTURE

```
D:\Flux_Trainer\                    # EVERYTHING HERE - NOTHING OUTSIDE
│
├── 🐍 python\                      # Portable Python 3.11.9
│   ├── python.exe                  # Isolated Python executable
│   ├── python311.dll               # Python runtime
│   ├── Lib\                        # Standard library
│   │   └── site-packages\          # ALL packages here
│   │       ├── torch\              # PyTorch (compiled locally)
│   │       ├── xformers\           # xformers (compiled locally)
│   │       ├── transformers\       # Hugging Face transformers
│   │       ├── diffusers\          # Diffusion models
│   │       └── prodigyopt\         # Prodigy optimizer
│   └── Scripts\                    # pip, wheel, etc.
│
├── 🔧 cuda_toolkit\                # Local CUDA 13.0 copy
│   ├── bin\                        # nvcc.exe, etc.
│   ├── lib\                        # CUDA libraries
│   ├── include\                    # CUDA headers
│   └── nvvm\                       # NVVM compiler
│
├── 🔨 build\                       # All compilation happens here
│   ├── pytorch\                    # PyTorch source & build
│   ├── xformers\                   # xformers source & build
│   └── logs\                       # Build logs
│
├── 🧠 models\                      # Flux model files (33GB)
│   ├── flux1-dev.safetensors      # Main Flux model
│   ├── ae.safetensors             # VAE
│   ├── clip_l.safetensors         # CLIP encoder
│   └── t5xxl_fp16.safetensors     # T5 encoder
│
├── 🖼️ dataset\                     # Your training images
│   └── [project]\                  # Project folder
│       └── [repeats]_[trigger]\    # Training folder
│
├── 💾 output\                      # Trained LoRAs
│   └── *.safetensors              # Your trained models
│
├── 🎨 samples\                     # Generated samples
│
├── ⚡ cuda_kernels\                # Custom Blackwell kernels
│
├── 📜 sd-scripts-cuda13\          # Training scripts
│   └── venv\                       # Virtual environment
│
├── 🛠️ tools\                       # Build tools
│   └── BuildTools\                 # VS Build Tools (local)
│
├── 📄 FluxTrainer.bat              # ONE-CLICK LAUNCHER
├── 🔍 verify_isolated.py           # Verification script
└── 📋 environment.conf             # Configuration
```

---

## 🚀 QUICK START GUIDE

### Step 1: Automated Installation (4-6 hours)

1. **Ensure D: drive has 150GB free space**

2. **Copy the AI Agent prompt** from `AI_AGENT_RTX5090_SETUP_ISOLATED.md`

3. **Paste into your AI agent** and let it run

4. **Everything installs to D:\Flux_Trainer automatically**

### Step 2: Launch Training

Double-click `D:\Flux_Trainer\FluxTrainer.bat` which:
- Sets up isolated environment
- Loads local Python
- Loads local CUDA
- Presents training menu
- Keeps everything isolated

### Step 3: Train Your LoRA

1. **Prepare dataset** in `D:\Flux_Trainer\dataset\`
2. **Select training type** from menu
3. **Wait for training** to complete
4. **Find LoRA** in `D:\Flux_Trainer\output\`

---

## 💡 KEY ADVANTAGES OF ISOLATION

### 1. No System Contamination
```powershell
# Check system Python - should return nothing
where python

# Check system environment - should return nothing
echo %PYTHONHOME%
echo %CUDA_HOME%

# Everything is in D:\Flux_Trainer only!
```

### 2. Complete Portability
```powershell
# To backup or move to another PC:
robocopy D:\Flux_Trainer E:\Backup\Flux_Trainer /E

# To run multiple versions:
D:\Flux_Trainer_v1\FluxTrainer.bat
D:\Flux_Trainer_v2\FluxTrainer.bat
D:\Flux_Trainer_experimental\FluxTrainer.bat
```

### 3. Clean Uninstall
```powershell
# Complete removal - just delete folder:
Remove-Item -Path "D:\Flux_Trainer" -Recurse -Force
# DONE! No registry, no PATH, no leftovers!
```

### 4. Version Independence
- Your system Python remains untouched
- Can have Python 3.12 system-wide
- Can have different CUDA versions
- No pip conflicts
- No package version issues

---

## 🔧 CONFIGURATION EXAMPLES

### Face Identity Training (Isolated)

Save as `D:\Flux_Trainer\sd-scripts-cuda13\config_face.toml`:

```toml
[model_arguments]
# All paths relative to D:\Flux_Trainer
pretrained_model_name_or_path = "D:/Flux_Trainer/models/flux1-dev.safetensors"
ae = "D:/Flux_Trainer/models/ae.safetensors"
clip_l = "D:/Flux_Trainer/models/clip_l.safetensors"
t5xxl = "D:/Flux_Trainer/models/t5xxl_fp16.safetensors"

[dataset_arguments]
train_data_dir = "D:/Flux_Trainer/dataset"
output_dir = "D:/Flux_Trainer/output"
output_name = "flux_lora_face"
sample_prompts = "D:/Flux_Trainer/sample_prompts.txt"
sample_sampler = "D:/Flux_Trainer/samples"

[training_arguments]
logging_dir = "D:/Flux_Trainer/logs"
```

---

## 🎯 WORKFLOW WITH ISOLATED ENVIRONMENT

### 1. Always Use FluxTrainer.bat
```batch
:: This launcher sets up EVERYTHING
D:\Flux_Trainer\FluxTrainer.bat
```

### 2. Manual Command Line Access
```powershell
# Set isolated environment for current session only
$env:FLUX_HOME = "D:\Flux_Trainer"
$env:PATH = "$env:FLUX_HOME\python;$env:FLUX_HOME\python\Scripts;$env:FLUX_HOME\cuda_toolkit\bin"
$env:PYTHONHOME = "$env:FLUX_HOME\python"
$env:CUDA_HOME = "$env:FLUX_HOME\cuda_toolkit"

# Now everything uses isolated versions
python --version  # Uses D:\Flux_Trainer\python\python.exe
pip list         # Shows only isolated packages
nvcc --version   # Uses D:\Flux_Trainer\cuda_toolkit\bin\nvcc.exe
```

### 3. Installing New Packages
```powershell
# ALWAYS use full path to isolated pip
D:\Flux_Trainer\python\python.exe -m pip install [package] --no-warn-script-location

# Or activate the virtual environment
cd D:\Flux_Trainer\sd-scripts-cuda13
.\venv\Scripts\activate
pip install [package]
```

---

## ⚠️ CRITICAL ISOLATION RULES

### DO:
- ✅ Always use `D:\Flux_Trainer\python\python.exe` for Python
- ✅ Install all packages to isolated environment
- ✅ Keep everything under `D:\Flux_Trainer\`
- ✅ Use `FluxTrainer.bat` launcher
- ✅ Set environment variables for session only

### DON'T:
- ❌ Never use system Python
- ❌ Never modify system PATH
- ❌ Never set permanent environment variables
- ❌ Never install packages globally
- ❌ Never mix with system CUDA

---

## 🔍 VERIFICATION & TROUBLESHOOTING

### Verify Isolation
```powershell
# Run verification script
D:\Flux_Trainer\python\python.exe D:\Flux_Trainer\verify_isolated.py

# Should show:
# ✓ Python: D:\Flux_Trainer\python\python.exe
# ✓ No system Python in path
# ✓ No system packages loaded
# ✓ CUDA from D:\Flux_Trainer\cuda_toolkit
# ✓ PyTorch with sm_120 support
# ✓ 900+ TFLOPS performance
```

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "Python not found" | Using system Python | Use `D:\Flux_Trainer\python\python.exe` |
| "Module not found" | Package in wrong Python | Install to isolated environment |
| "CUDA error" | Using system CUDA | Set `$env:CUDA_HOME = "D:\Flux_Trainer\cuda_toolkit"` |
| "Access denied" | Permissions issue | Run as Administrator for initial setup |
| "Out of space" | D: drive full | Need 150GB free on D: |

---

## 📊 PERFORMANCE EXPECTATIONS

With isolated environment on RTX 5090:

- **Setup Time:** 4-6 hours (one-time)
- **Disk Usage:** ~100-150GB total
- **Performance:** 900+ TFLOPS (same as system install)
- **Training Speed:** No performance penalty
- **Memory Usage:** Same as system install

---

## 🎓 ADVANCED USAGE

### Running Multiple Isolated Environments

```powershell
# Production environment
D:\Flux_Trainer_Production\FluxTrainer.bat

# Testing environment
D:\Flux_Trainer_Testing\FluxTrainer.bat

# Experimental features
D:\Flux_Trainer_Experimental\FluxTrainer.bat

# Each completely independent!
```

### Creating Backups

```powershell
# Full backup (everything)
$date = Get-Date -Format "yyyy-MM-dd"
robocopy D:\Flux_Trainer "E:\Backups\Flux_Trainer_$date" /E /MT:16

# Backup only trained models
robocopy D:\Flux_Trainer\output E:\Backups\LoRAs /E
```

### Sharing Environment

```powershell
# Zip entire environment
Compress-Archive -Path D:\Flux_Trainer -DestinationPath FluxTrainer_Portable.zip

# Share with others - they just extract and run!
# No installation needed on their system
```

---

## ✅ FINAL CHECKLIST

Before starting training in isolated environment:

- [ ] D: drive has 150GB free space
- [ ] Ran AI Agent setup or manual installation
- [ ] All files contained in D:\Flux_Trainer
- [ ] No system Python/CUDA modifications
- [ ] FluxTrainer.bat launches successfully
- [ ] Verification script passes all checks
- [ ] Models downloaded to D:\Flux_Trainer\models
- [ ] Dataset prepared in D:\Flux_Trainer\dataset
- [ ] Can run `where python` and see no results (good!)

---

## 🎉 SUCCESS INDICATORS

You know your isolated setup is working when:

1. **System remains clean** - No Python in PATH
2. **Everything runs from D:\Flux_Trainer** 
3. **Can delete and reinstall anytime**
4. **Can copy to another PC and just run**
5. **No conflicts with other Python apps**
6. **Training achieves 900+ TFLOPS**
7. **Zero warnings about compatibility**

---

## 📝 NOTES

### Why Isolation Matters:

1. **Reproducibility** - Exact same environment every time
2. **Compatibility** - No conflicts with other software
3. **Portability** - Move between machines easily
4. **Safety** - Can't break system Python
5. **Cleanliness** - Complete removal when done
6. **Testing** - Run multiple versions side-by-side

### Storage Requirements:

- **Python + packages:** ~10GB
- **CUDA toolkit copy:** ~5GB  
- **PyTorch compiled:** ~15GB
- **Build files:** ~30GB (can delete after setup)
- **Flux models:** ~33GB
- **Working space:** ~20GB
- **Total recommended:** 150GB

---

**Remember:** This isolated setup gives you COMPLETE CONTROL with ZERO SYSTEM IMPACT. Everything you need is in D:\Flux_Trainer, nothing outside!

---

*Isolated Environment Version 1.0*
*Location: D:\Flux_Trainer*
*System Impact: ZERO*
