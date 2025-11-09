# RTX 5090 + CUDA 13.0 - QUICK COMMAND REFERENCE
## Fully Isolated Self-Contained Environment
## Everything in D:\Flux_Trainer - No System Contamination

**⚠️ PREREQUISITE:** This creates a COMPLETELY ISOLATED environment in D:\Flux_Trainer

---

## 1. CREATE ISOLATED ENVIRONMENT (Run as Admin)
```powershell
# Create complete directory structure
mkdir D:\Flux_Trainer
mkdir D:\Flux_Trainer\python
mkdir D:\Flux_Trainer\build
mkdir D:\Flux_Trainer\cuda_toolkit
mkdir D:\Flux_Trainer\models
mkdir D:\Flux_Trainer\dataset
mkdir D:\Flux_Trainer\output
mkdir D:\Flux_Trainer\samples
mkdir D:\Flux_Trainer\cuda_kernels
mkdir D:\Flux_Trainer\temp

# Set environment variable for this session
$env:FLUX_HOME = "D:\Flux_Trainer"
cd D:\Flux_Trainer
```

## 2. INSTALL PORTABLE PYTHON 3.11.9
```powershell
# Download Python embeddable package (no system installation)
cd D:\Flux_Trainer\temp
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" -OutFile "python.zip"

# Extract to isolated location
Expand-Archive -Path "python.zip" -DestinationPath "D:\Flux_Trainer\python" -Force

# Download get-pip
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "D:\Flux_Trainer\python\get-pip.py"

# Modify python311._pth to enable pip and site-packages
$pthContent = @"
python311.zip
.
Lib
Lib\site-packages
import site
"@
Set-Content -Path "D:\Flux_Trainer\python\python311._pth" -Value $pthContent

# Install pip in isolated environment
D:\Flux_Trainer\python\python.exe D:\Flux_Trainer\python\get-pip.py --no-warn-script-location

# Verify isolated Python
D:\Flux_Trainer\python\python.exe --version  # MUST show 3.11.9
```

## 3. SETUP ISOLATED CUDA 13.0 ENVIRONMENT
```powershell
# Copy CUDA files to isolated location (if CUDA 13.0 installed)
robocopy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" "D:\Flux_Trainer\cuda_toolkit" /E

# Set isolated CUDA environment
$env:CUDA_HOME = "D:\Flux_Trainer\cuda_toolkit"
$env:CUDA_PATH = "D:\Flux_Trainer\cuda_toolkit"
$env:PATH = "D:\Flux_Trainer\cuda_toolkit\bin;D:\Flux_Trainer\python;D:\Flux_Trainer\python\Scripts;$env:PATH"

# Verify CUDA in isolated environment
D:\Flux_Trainer\cuda_toolkit\bin\nvcc.exe --version  # Must show 13.0
```

## 4. BUILD PYTORCH WITH SM_120 (Isolated)
```powershell
cd D:\Flux_Trainer\build
mkdir pytorch_rtx5090
cd pytorch_rtx5090

# Use isolated Python for all operations
D:\Flux_Trainer\python\python.exe -m pip install numpy==1.26.4 pyyaml==6.0.1 ninja==1.11.1.1 cmake==3.28.1 --no-warn-script-location

# Clone and patch
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v2.5.1

# Apply patches (see main guide for patch content)
git apply rtx5090_cuda13.patch

# Configure with isolated paths
$env:TORCH_CUDA_ARCH_LIST = "8.9;9.0;12.0"
$env:USE_CUDA = "1"
$env:CMAKE_GENERATOR = "Ninja"
$env:CUDA_HOME = "D:\Flux_Trainer\cuda_toolkit"
$env:CMAKE_PREFIX_PATH = "D:\Flux_Trainer\python"

# Build with isolated Python (2-4 hours)
D:\Flux_Trainer\python\python.exe setup.py install --prefix="D:\Flux_Trainer\python"

# Verify
D:\Flux_Trainer\python\python.exe -c "import torch; print('sm_120' in str(torch.cuda.get_arch_list()))"
```

## 5. BUILD XFORMERS (Isolated)
```powershell
cd D:\Flux_Trainer\build
git clone https://github.com/facebookresearch/xformers.git
cd xformers

$env:CUDA_HOME = "D:\Flux_Trainer\cuda_toolkit"
$env:TORCH_CUDA_ARCH_LIST = "8.9;9.0;12.0"

D:\Flux_Trainer\python\python.exe -m pip install -r requirements.txt --no-warn-script-location
D:\Flux_Trainer\python\python.exe setup.py install --prefix="D:\Flux_Trainer\python"
```

## 6. BUILD BLACKWELL KERNELS (Isolated)
```powershell
cd D:\Flux_Trainer\cuda_kernels

# Create blackwell_kernels.cu and setup.py from main guide
D:\Flux_Trainer\python\python.exe setup.py install --prefix="D:\Flux_Trainer\python"
D:\Flux_Trainer\python\python.exe -c "import blackwell_flux_kernels; print('✓ Kernels loaded')"
```

## 7. SETUP SD-SCRIPTS (Isolated)
```powershell
cd D:\Flux_Trainer
git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts-cuda13
cd sd-scripts-cuda13

# Create virtual environment using isolated Python
D:\Flux_Trainer\python\python.exe -m venv venv --system-site-packages

# Activate isolated venv
.\venv\Scripts\activate

# Install dependencies in isolated environment
pip install accelerate==0.34.2 transformers==4.46.2 diffusers==0.31.0 --no-warn-script-location
pip install prodigyopt==1.0 --no-warn-script-location

# Apply patches from main guide
```

## 8. DOWNLOAD FLUX MODELS
```powershell
cd D:\Flux_Trainer\models
# Download Flux models (use wget or browser)
# flux1-dev.safetensors (~23GB)
# ae.safetensors (~335MB)  
# clip_l.safetensors (~246MB)
# t5xxl_fp16.safetensors (~9.5GB)
```

## 9. PREPARE DATASET
```powershell
# Create dataset structure
cd D:\Flux_Trainer\dataset
mkdir t4r4woman
mkdir t4r4woman\70_t4r4woman

# Add images and create captions
Get-ChildItem "D:\Flux_Trainer\dataset\t4r4woman\70_t4r4woman\*.txt" | ForEach-Object {
    Set-Content $_.FullName "t4r4woman"
}
```

## 10. CREATE ISOLATED CONFIG
Save as `D:\Flux_Trainer\sd-scripts-cuda13\config.toml`:
```toml
[model_arguments]
pretrained_model_name_or_path = "D:/Flux_Trainer/models/flux1-dev.safetensors"
ae = "D:/Flux_Trainer/models/ae.safetensors"
clip_l = "D:/Flux_Trainer/models/clip_l.safetensors"  
t5xxl = "D:/Flux_Trainer/models/t5xxl_fp16.safetensors"

[dataset_arguments]
train_data_dir = "D:/Flux_Trainer/dataset"
output_dir = "D:/Flux_Trainer/output"
output_name = "flux_lora_t4r4woman"

[network_arguments]
network_module = "networks.lora_flux"
network_dim = 128
network_alpha = 64

[optimizer_arguments]
optimizer_type = "prodigyopt"
learning_rate = 1.0

[training_arguments]
max_train_steps = 1500
mixed_precision = "bf16"
save_every_n_steps = 100
sample_every_n_steps = 100
```

## 11. LAUNCH WITH ISOLATED ENVIRONMENT
```powershell
cd D:\Flux_Trainer\sd-scripts-cuda13

# Create launch script
@'
@echo off
set FLUX_HOME=D:\Flux_Trainer
set PATH=%FLUX_HOME%\python;%FLUX_HOME%\python\Scripts;%FLUX_HOME%\cuda_toolkit\bin;%PATH%
set CUDA_HOME=%FLUX_HOME%\cuda_toolkit
set PYTHONHOME=%FLUX_HOME%\python
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /d D:\Flux_Trainer\sd-scripts-cuda13
call venv\Scripts\activate
python flux_train_network.py --config_file config.toml --highvram
'@ | Out-File -FilePath "D:\Flux_Trainer\launch_training.bat" -Encoding ASCII

# Run training
D:\Flux_Trainer\launch_training.bat
```

## 12. VERIFY ISOLATED SETUP
```powershell
# Run verification using isolated Python
D:\Flux_Trainer\python\python.exe -c @"
import sys, os
print('Python location:', sys.executable)
print('Python version:', sys.version)
print('Site packages:', [p for p in sys.path if 'site-packages' in p])

# Verify isolation
assert 'Flux_Trainer' in sys.executable, 'Not using isolated Python!'
assert not any('C:\\Python' in p for p in sys.path), 'System Python detected!'

import torch
print('PyTorch location:', torch.__file__)
print('CUDA available:', torch.cuda.is_available())
print('sm_120 support:', 'sm_120' in str(torch.cuda.get_arch_list()))
"@
```

---

## ⚠️ CRITICAL ADVANTAGES OF ISOLATION

1. **No System Contamination** - System Python remains untouched
2. **Portable** - Entire D:\Flux_Trainer folder can be moved/backed up
3. **Multiple Versions** - Can have different setups in different folders
4. **Clean Uninstall** - Just delete D:\Flux_Trainer folder
5. **No PATH Conflicts** - Isolated from system PATH
6. **Reproducible** - Exact same environment every time

---

## 🚨 TROUBLESHOOTING ISOLATED SETUP

| Issue | Solution |
|-------|----------|
| "Python not found" | Always use `D:\Flux_Trainer\python\python.exe` |
| "Module not found" | Install to isolated pip: `D:\Flux_Trainer\python\python.exe -m pip install` |
| "CUDA not found" | Set `$env:CUDA_HOME = "D:\Flux_Trainer\cuda_toolkit"` |
| "Wrong Python used" | Check with `where python` - should show D:\Flux_Trainer first |
| "Can't import torch" | Reinstall to isolated environment |

---

## 📁 ISOLATED DIRECTORY STRUCTURE

```
D:\Flux_Trainer\              # COMPLETELY SELF-CONTAINED
├── python\                   # Isolated Python 3.11.9
│   ├── python.exe
│   ├── Lib\
│   │   └── site-packages\    # All Python packages here
│   └── Scripts\
├── cuda_toolkit\             # Isolated CUDA 13.0
│   ├── bin\
│   ├── lib\
│   └── include\
├── build\                    # All compilation here
│   ├── pytorch_rtx5090\
│   └── xformers\
├── models\                   # Flux models
├── dataset\                  # Training data
├── output\                   # LoRA outputs
├── samples\                  # Generated samples
├── cuda_kernels\            # Custom kernels
├── sd-scripts-cuda13\       # Training scripts
│   └── venv\               # Virtual environment
├── temp\                    # Temporary files
└── launch_training.bat      # One-click launcher

NO FILES OUTSIDE D:\Flux_Trainer!
```

---

## 🔒 ENVIRONMENT VARIABLES (Session Only)

```powershell
# Set these for current session only (not system-wide)
$env:FLUX_HOME = "D:\Flux_Trainer"
$env:PYTHONHOME = "D:\Flux_Trainer\python"
$env:CUDA_HOME = "D:\Flux_Trainer\cuda_toolkit"
$env:PATH = "D:\Flux_Trainer\python;D:\Flux_Trainer\python\Scripts;D:\Flux_Trainer\cuda_toolkit\bin;$env:PATH"
```

---

**Full Guide:** UNCOMPROMISING_RTX5090_GUIDE_ISOLATED.md
**Time Required:** 4-6 hours setup, 1-1.5 hours training
**Result:** Completely isolated, portable, zero-contamination environment
