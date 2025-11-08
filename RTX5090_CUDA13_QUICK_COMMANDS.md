# RTX 5090 + CUDA 13.0 - QUICK COMMAND REFERENCE
## Uncompromising Setup - Essential Commands Only

**⚠️ PREREQUISITE:** Follow full guide for context. This is just command reference.

---

## 1. INSTALL PYTHON 3.11.9 (Run as Admin)
```powershell
# Save and run the install_python_3.11.9.ps1 script from main guide
# Or quick version:
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile "$env:TEMP\python-3.11.9.exe"
Start-Process -FilePath "$env:TEMP\python-3.11.9.exe" -ArgumentList "/quiet", "InstallAllUsers=1", "TargetDir=C:\Python311", "PrependPath=1" -Wait
python --version  # MUST show 3.11.9
```

## 2. SET CUDA 13.0 ENVIRONMENT
```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"
nvcc --version  # Must show 13.0
```

## 3. BUILD PYTORCH WITH SM_120
```powershell
mkdir C:\build\pytorch_rtx5090
cd C:\build\pytorch_rtx5090

# Install build deps
pip install numpy==1.26.4 pyyaml==6.0.1 ninja==1.11.1.1 cmake==3.28.1

# Clone and patch
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v2.5.1

# Apply patches (see main guide for patch content)
git apply rtx5090_cuda13.patch

# Configure
$env:TORCH_CUDA_ARCH_LIST = "8.9;9.0;12.0"
$env:USE_CUDA = "1"
$env:CMAKE_GENERATOR = "Ninja"

# Build (2-4 hours)
python setup.py install

# Verify
python -c "import torch; print('sm_120' in str(torch.cuda.get_arch_list()))"
```

## 4. BUILD XFORMERS
```powershell
cd C:\build
git clone https://github.com/facebookresearch/xformers.git
cd xformers

$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:TORCH_CUDA_ARCH_LIST = "8.9;9.0;12.0"

pip install -r requirements.txt
python setup.py install
```

## 5. BUILD BLACKWELL KERNELS
```powershell
cd C:\AI\flux_training\cuda_kernels
# Create blackwell_kernels.cu and setup.py from main guide
python setup.py install
python -c "import blackwell_flux_kernels; print('✓ Kernels loaded')"
```

## 6. SETUP SD-SCRIPTS
```powershell
cd C:\AI\flux_training
git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts-cuda13
cd sd-scripts-cuda13

C:\Python311\python.exe -m venv venv
.\venv\Scripts\activate

# Link compiled PyTorch
$site_packages = python -c "import site; print(site.getsitepackages()[0])"
New-Item -ItemType Junction -Path "$site_packages\torch" -Target "C:\build\pytorch-install\lib\python3.11\site-packages\torch"

# Install deps
pip install accelerate==0.34.2 transformers==4.46.2 diffusers==0.31.0
pip install prodigyopt==1.0  # CRITICAL

# Apply patches from main guide
```

## 7. PREPARE DATASET
```powershell
# 17 images in dataset/t4r4woman/70_t4r4woman/
# All captions = just "t4r4woman"
Get-ChildItem "dataset\t4r4woman\70_t4r4woman\*.txt" | ForEach-Object {
    Set-Content $_.FullName "t4r4woman"
}
```

## 8. CREATE CONFIG (Save as config.toml)
```toml
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
timestep_sampling = "shift"
discrete_flow_shift = 3.1582
```

## 9. LAUNCH TRAINING
```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python flux_train_network.py --config_file config.toml --highvram
```

## 10. VERIFY SUCCESS
```python
# Run verification script
python verify_setup.py

# Expected output:
# ✅ Python 3.11.9
# ✅ CUDA 13.0
# ✅ PyTorch with native sm_120
# ✅ RTX 5090 detected
# ✅ Blackwell custom kernels
# ✅ Performance: 900+ TFLOPS
# 🎉 ALL CHECKS PASSED
```

---

## ⚠️ CRITICAL REMINDERS

1. **Python MUST be 3.11.9** - Not 3.11.8, not 3.11.10, not 3.12
2. **CUDA 13.0 paths** - Use v13.0, not v12.4
3. **Compile from source** - No pre-built packages
4. **Prodigy optimizer** - Essential for face training
5. **Simple captions** - Just trigger word
6. **Judge by samples** - Not loss values

---

## 🚨 TROUBLESHOOTING

| Issue | Solution |
|-------|----------|
| "Python not 3.11.9" | Run clean install script |
| "sm_120 not found" | Rebuild PyTorch with patches |
| "CUDA error" | Verify CUDA 13.0 paths |
| "Import error" | Check virtual env activation |
| "Low TFLOPS" | Verify compilation flags |

---

**Full Guide:** UNCOMPROMISING_RTX5090_GUIDE.md
**Time Required:** 4-6 hours setup, 1-1.5 hours training
**Result:** Native RTX 5090 performance with zero compromises
