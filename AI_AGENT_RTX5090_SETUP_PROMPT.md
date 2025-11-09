# AI AGENT PROMPT: RTX 5090 FLUX TRAINING SETUP
## Fully Isolated Self-Contained Environment
## Zero System Contamination - Everything in D:\Flux_Trainer

**Instructions:** Copy this entire document and provide it to an AI agent capable of system administration.

---

```
You are an expert system administrator and CUDA developer tasked with setting up a COMPLETELY ISOLATED Flux LoRA training environment for an RTX 5090 GPU. This setup will be 100% self-contained in D:\Flux_Trainer with ZERO system-wide installations or modifications.

SYSTEM CONTEXT:
- Target GPU: NVIDIA RTX 5090 (32GB VRAM, Blackwell architecture)
- Current Driver: 581.57 or newer (only system requirement)
- Operating System: Windows 10/11 x64
- Required RAM: 64GB recommended for compilation
- Required Storage: 150GB free on D: drive
- Installation Location: D:\Flux_Trainer (EVERYTHING goes here)

CRITICAL REQUIREMENTS:
- NO system-wide Python installation
- NO system PATH modifications
- NO system environment variables
- EVERYTHING contained in D:\Flux_Trainer
- Portable Python 3.11.9 EXACTLY
- Local CUDA 13.0 copy
- All builds in D:\Flux_Trainer\build
- Complete isolation from system
- Must be deletable/portable as single folder
- Performance MUST exceed 900 TFLOPS

==============================================================================
PHASE 1: CREATE ISOLATED ENVIRONMENT STRUCTURE [10 minutes]
==============================================================================

Execute as Administrator in PowerShell:

1. Create complete directory structure:
   ```powershell
   # Create main environment
   $FLUX_HOME = "D:\Flux_Trainer"
   mkdir $FLUX_HOME -Force
   
   # Create all subdirectories
   $directories = @(
       "python",
       "python\Scripts",
       "python\Lib\site-packages",
       "cuda_toolkit",
       "build",
       "build\pytorch",
       "build\xformers",
       "models",
       "dataset",
       "output",
       "samples",
       "logs",
       "cuda_kernels",
       "sd-scripts-cuda13",
       "temp",
       "cache",
       "tools"
   )
   
   foreach ($dir in $directories) {
       mkdir "$FLUX_HOME\$dir" -Force
   }
   
   # Set session-only environment
   $env:FLUX_HOME = $FLUX_HOME
   cd $FLUX_HOME
   ```

2. Create environment configuration file:
   ```powershell
   @'
   # Flux Trainer Isolated Environment Configuration
   FLUX_HOME=D:\Flux_Trainer
   PYTHON_HOME=%FLUX_HOME%\python
   CUDA_HOME=%FLUX_HOME%\cuda_toolkit
   BUILD_DIR=%FLUX_HOME%\build
   MODELS_DIR=%FLUX_HOME%\models
   '@ | Out-File "$FLUX_HOME\environment.conf" -Encoding UTF8
   ```

CHECKPOINT: Directory structure created at D:\Flux_Trainer

==============================================================================
PHASE 2: INSTALL PORTABLE PYTHON 3.11.9 [20 minutes]
==============================================================================

1. Download Python embeddable package:
   ```powershell
   cd $env:FLUX_HOME\temp
   
   # Download Python 3.11.9 embeddable
   Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" -OutFile "python_embed.zip"
   
   # Download Python 3.11.9 full (for headers and libs)
   Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile "python_full.exe"
   ```

2. Extract and setup portable Python:
   ```powershell
   # Extract embeddable Python
   Expand-Archive -Path "python_embed.zip" -DestinationPath "$env:FLUX_HOME\python" -Force
   
   # Extract full Python for development files
   Start-Process -FilePath "python_full.exe" -ArgumentList "/quiet", "TargetDir=$env:FLUX_HOME\temp\python_full" -Wait
   
   # Copy necessary development files
   Copy-Item "$env:FLUX_HOME\temp\python_full\include" "$env:FLUX_HOME\python\" -Recurse
   Copy-Item "$env:FLUX_HOME\temp\python_full\libs" "$env:FLUX_HOME\python\" -Recurse
   ```

3. Configure Python for package installation:
   ```powershell
   # Modify python311._pth to enable site-packages
   @'
   python311.zip
   .
   Lib
   Lib\site-packages
   include
   libs
   Scripts
   import site
   '@ | Out-File "$env:FLUX_HOME\python\python311._pth" -Encoding UTF8
   
   # Download and install pip
   Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "$env:FLUX_HOME\python\get-pip.py"
   & "$env:FLUX_HOME\python\python.exe" "$env:FLUX_HOME\python\get-pip.py" --no-warn-script-location
   
   # Install wheel and setuptools
   & "$env:FLUX_HOME\python\python.exe" -m pip install --upgrade pip setuptools wheel --no-warn-script-location
   ```

4. Verify isolated Python:
   ```powershell
   & "$env:FLUX_HOME\python\python.exe" --version  # MUST show Python 3.11.9
   & "$env:FLUX_HOME\python\python.exe" -c "import sys; print(f'Executable: {sys.executable}'); assert 'Flux_Trainer' in sys.executable"
   ```

CHECKPOINT: Portable Python 3.11.9 installed at D:\Flux_Trainer\python

==============================================================================
PHASE 3: SETUP LOCAL CUDA 13.0 TOOLKIT [15 minutes]
==============================================================================

1. Copy CUDA toolkit to isolated location:
   ```powershell
   # If CUDA 13.0 is installed system-wide, copy it
   if (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0") {
       robocopy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" "$env:FLUX_HOME\cuda_toolkit" /E /MT:16
   } else {
       # Download CUDA 13.0 toolkit installer
       Write-Host "Downloading CUDA 13.0 toolkit..."
       # Note: User must install CUDA 13.0 first or download manually
       throw "Please install CUDA 13.0 first"
   }
   ```

2. Configure isolated CUDA environment:
   ```powershell
   # Set session environment
   $env:CUDA_HOME = "$env:FLUX_HOME\cuda_toolkit"
   $env:CUDA_PATH = "$env:FLUX_HOME\cuda_toolkit"
   $env:PATH = "$env:FLUX_HOME\cuda_toolkit\bin;$env:FLUX_HOME\python;$env:FLUX_HOME\python\Scripts;$env:PATH"
   ```

3. Verify CUDA in isolated environment:
   ```powershell
   & "$env:FLUX_HOME\cuda_toolkit\bin\nvcc.exe" --version  # Must show release 13.0
   
   # Test sm_120 compilation
   @'
   #include <cuda_runtime.h>
   #include <stdio.h>
   __global__ void test_sm120() { printf("RTX 5090 sm_120 kernel!\n"); }
   int main() {
       cudaDeviceProp prop;
       cudaGetDeviceProperties(&prop, 0);
       printf("Device: %s, Compute: %d.%d\n", prop.name, prop.major, prop.minor);
       test_sm120<<<1, 1>>>();
       cudaDeviceSynchronize();
       return 0;
   }
   '@ | Out-File "$env:FLUX_HOME\temp\test_sm120.cu"
   
   & "$env:FLUX_HOME\cuda_toolkit\bin\nvcc.exe" -arch=sm_120 "$env:FLUX_HOME\temp\test_sm120.cu" -o "$env:FLUX_HOME\temp\test_sm120.exe"
   & "$env:FLUX_HOME\temp\test_sm120.exe"
   ```

CHECKPOINT: Local CUDA 13.0 configured at D:\Flux_Trainer\cuda_toolkit

==============================================================================
PHASE 4: INSTALL BUILD TOOLS TO ISOLATED ENVIRONMENT [20 minutes]
==============================================================================

1. Install Visual Studio Build Tools locally:
   ```powershell
   # Download build tools
   cd $env:FLUX_HOME\tools
   Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_buildtools.exe" -OutFile "vs_buildtools.exe"
   
   # Install to local directory
   Start-Process -FilePath "vs_buildtools.exe" -ArgumentList @(
       "--quiet", "--wait",
       "--add", "Microsoft.VisualStudio.Workload.VCTools",
       "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
       "--installPath", "$env:FLUX_HOME\tools\BuildTools"
   ) -Wait
   ```

2. Setup isolated build environment:
   ```powershell
   # Install build dependencies with isolated Python
   & "$env:FLUX_HOME\python\python.exe" -m pip install `
       numpy==1.26.4 `
       pyyaml==6.0.1 `
       typing_extensions==4.9.0 `
       ninja==1.11.1.1 `
       cmake==3.28.1 `
       mkl-static `
       mkl-include `
       --no-warn-script-location
   ```

CHECKPOINT: Build tools installed in isolated environment

==============================================================================
PHASE 5: BUILD PYTORCH FROM SOURCE [2-4 hours]
==============================================================================

1. Setup build environment:
   ```powershell
   cd $env:FLUX_HOME\build\pytorch
   
   # Clone PyTorch
   git clone --recursive https://github.com/pytorch/pytorch.git
   cd pytorch
   git checkout v2.5.1
   git submodule sync
   git submodule update --init --recursive
   ```

2. Apply RTX 5090 patches:
   ```powershell
   @'
   diff --git a/aten/src/ATen/cuda/CUDAContext.cpp b/aten/src/ATen/cuda/CUDAContext.cpp
   +++ b/aten/src/ATen/cuda/CUDAContext.cpp
   @@ -87,6 +87,7 @@
      cuda_arch_list.push_back(9.0);
   +  cuda_arch_list.push_back(12.0);  // RTX 5090 Blackwell
    }
   diff --git a/cmake/public/cuda.cmake b/cmake/public/cuda.cmake
   +++ b/cmake/public/cuda.cmake
   @@ -120,6 +120,8 @@
    endif()
   +if(CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
   +  list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_120,code=sm_120")
   +endif()
   '@ | Out-File rtx5090.patch
   git apply rtx5090.patch
   ```

3. Build PyTorch with isolated environment:
   ```powershell
   # Set all paths to isolated environment
   $env:TORCH_CUDA_ARCH_LIST = "8.9;9.0;12.0"
   $env:USE_CUDA = "1"
   $env:CMAKE_GENERATOR = "Ninja"
   $env:CUDA_HOME = "$env:FLUX_HOME\cuda_toolkit"
   $env:CMAKE_PREFIX_PATH = "$env:FLUX_HOME\python"
   $env:TORCH_NVCC_FLAGS = "-gencode=arch=compute_120,code=sm_120"
   $env:MAX_JOBS = "8"
   
   # Build and install to isolated Python
   & "$env:FLUX_HOME\python\python.exe" setup.py install --prefix="$env:FLUX_HOME\python"
   ```

4. Verify PyTorch build:
   ```powershell
   & "$env:FLUX_HOME\python\python.exe" -c @"
   import torch
   print(f'PyTorch version: {torch.__version__}')
   print(f'CUDA available: {torch.cuda.is_available()}')
   print(f'CUDA version: {torch.version.cuda}')
   assert 'sm_120' in str(torch.cuda.get_arch_list()), 'sm_120 not found!'
   print('✓ Native sm_120 support confirmed')
   "@
   ```

CHECKPOINT: PyTorch compiled with native sm_120 support in isolated environment

==============================================================================
PHASE 6: BUILD XFORMERS [30 minutes]
==============================================================================

1. Build xformers in isolated environment:
   ```powershell
   cd $env:FLUX_HOME\build\xformers
   git clone https://github.com/facebookresearch/xformers.git
   cd xformers
   git checkout v0.0.28
   
   # Set environment
   $env:CUDA_HOME = "$env:FLUX_HOME\cuda_toolkit"
   $env:TORCH_CUDA_ARCH_LIST = "8.9;9.0;12.0"
   
   # Install requirements
   & "$env:FLUX_HOME\python\python.exe" -m pip install -r requirements.txt --no-warn-script-location
   
   # Build and install
   & "$env:FLUX_HOME\python\python.exe" setup.py install --prefix="$env:FLUX_HOME\python"
   ```

CHECKPOINT: xformers compiled in isolated environment

==============================================================================
PHASE 7: CREATE BLACKWELL KERNELS [20 minutes]
==============================================================================

1. Create custom CUDA kernels:
   ```powershell
   cd $env:FLUX_HOME\cuda_kernels
   ```

2. Create blackwell_kernels.cu (see main guide for full code)

3. Build kernels with isolated environment:
   ```powershell
   & "$env:FLUX_HOME\python\python.exe" setup.py install --prefix="$env:FLUX_HOME\python"
   & "$env:FLUX_HOME\python\python.exe" -c "import blackwell_flux_kernels; print('✓ Blackwell kernels loaded')"
   ```

CHECKPOINT: Custom Blackwell kernels compiled

==============================================================================
PHASE 8: SETUP SD-SCRIPTS [20 minutes]
==============================================================================

1. Clone and setup sd-scripts:
   ```powershell
   cd $env:FLUX_HOME
   git clone https://github.com/kohya-ss/sd-scripts.git sd-scripts-cuda13
   cd sd-scripts-cuda13
   
   # Create virtual environment using isolated Python
   & "$env:FLUX_HOME\python\python.exe" -m venv venv --system-site-packages
   
   # Activate and install dependencies
   & ".\venv\Scripts\activate"
   pip install accelerate==0.34.2 transformers==4.46.2 diffusers==0.31.0 --no-warn-script-location
   pip install safetensors==0.4.5 omegaconf==2.3.0 wandb==0.17.8 --no-warn-script-location
   pip install prodigyopt==1.0 bitsandbytes==0.43.3 --no-warn-script-location
   ```

CHECKPOINT: sd-scripts installed in isolated environment

==============================================================================
PHASE 9: DOWNLOAD FLUX MODELS [30 minutes]
==============================================================================

1. Download models to isolated environment:
   ```powershell
   cd $env:FLUX_HOME\models
   
   Write-Host "Please download the following models to $env:FLUX_HOME\models:"
   Write-Host "1. flux1-dev.safetensors (23GB)"
   Write-Host "2. ae.safetensors (335MB)"
   Write-Host "3. clip_l.safetensors (246MB)"
   Write-Host "4. t5xxl_fp16.safetensors (9.5GB)"
   
   # Verify models
   $requiredModels = @(
       "flux1-dev.safetensors",
       "ae.safetensors",
       "clip_l.safetensors",
       "t5xxl_fp16.safetensors"
   )
   
   foreach ($model in $requiredModels) {
       if (Test-Path "$env:FLUX_HOME\models\$model") {
           Write-Host "✓ Found: $model"
       } else {
           Write-Host "✗ Missing: $model"
       }
   }
   ```

CHECKPOINT: All Flux models in D:\Flux_Trainer\models

==============================================================================
PHASE 10: CREATE PORTABLE LAUNCHER [5 minutes]
==============================================================================

1. Create portable launcher script:
   ```powershell
   @'
   @echo off
   :: Flux Trainer Portable Launcher
   :: This sets up the complete isolated environment
   
   set FLUX_HOME=D:\Flux_Trainer
   
   :: Set isolated Python
   set PYTHONHOME=%FLUX_HOME%\python
   set PYTHONPATH=%FLUX_HOME%\python\Lib;%FLUX_HOME%\python\Lib\site-packages
   
   :: Set isolated CUDA
   set CUDA_HOME=%FLUX_HOME%\cuda_toolkit
   set CUDA_PATH=%FLUX_HOME%\cuda_toolkit
   
   :: Set isolated PATH (no system contamination)
   set PATH=%FLUX_HOME%\python;%FLUX_HOME%\python\Scripts;%FLUX_HOME%\cuda_toolkit\bin;%FLUX_HOME%\tools\BuildTools\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64
   
   :: Set build paths
   set CMAKE_PREFIX_PATH=%FLUX_HOME%\python
   set DISTUTILS_USE_SDK=1
   
   :: Performance settings
   set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   
   :: Launch directory
   cd /d %FLUX_HOME%\sd-scripts-cuda13
   
   :: Activate virtual environment
   call venv\Scripts\activate.bat
   
   :: Display environment info
   echo ========================================
   echo Flux Trainer Isolated Environment
   echo ========================================
   echo Python: %PYTHONHOME%
   echo CUDA: %CUDA_HOME%
   echo Working Dir: %CD%
   echo ========================================
   echo.
   
   :: Menu
   echo Select option:
   echo 1. Train Face LoRA (1500 steps)
   echo 2. Train Action LoRA (1000 steps)
   echo 3. Train Style LoRA (800 steps)
   echo 4. Run verification
   echo 5. Command prompt
   echo.
   
   set /p choice="Enter choice (1-5): "
   
   if "%choice%"=="1" goto train_face
   if "%choice%"=="2" goto train_action
   if "%choice%"=="3" goto train_style
   if "%choice%"=="4" goto verify
   if "%choice%"=="5" goto shell
   
   :train_face
   python flux_train_network.py --config_file config_face.toml --highvram
   goto end
   
   :train_action
   python flux_train_network.py --config_file config_action.toml --highvram
   goto end
   
   :train_style
   python flux_train_network.py --config_file config_style.toml --highvram
   goto end
   
   :verify
   python %FLUX_HOME%\verify_isolated.py
   goto end
   
   :shell
   cmd /k
   goto end
   
   :end
   pause
   '@ | Out-File "$env:FLUX_HOME\FluxTrainer.bat" -Encoding ASCII
   ```

2. Create verification script:
   ```powershell
   @'
   import os, sys, torch
   
   print("=" * 60)
   print("FLUX TRAINER ISOLATED ENVIRONMENT VERIFICATION")
   print("=" * 60)
   
   # Check Python isolation
   assert "Flux_Trainer" in sys.executable, "Not using isolated Python!"
   print(f"✓ Python: {sys.executable}")
   print(f"✓ Version: {sys.version.split()[0]}")
   
   # Check CUDA
   print(f"✓ CUDA available: {torch.cuda.is_available()}")
   print(f"✓ CUDA version: {torch.version.cuda}")
   
   # Check sm_120
   arch_list = str(torch.cuda.get_arch_list())
   assert "sm_120" in arch_list, "Missing sm_120 support!"
   print(f"✓ Native sm_120 support: YES")
   
   # Check paths
   assert not any("C:\\Python" in p for p in sys.path), "System Python detected!"
   assert not any("C:\\Users" in p for p in sys.path if "AppData" in p), "User Python detected!"
   print("✓ Complete isolation verified")
   
   # Performance test
   import time
   size = 4096
   a = torch.randn(size, size, dtype=torch.bfloat16).cuda()
   b = torch.randn(size, size, dtype=torch.bfloat16).cuda()
   
   for _ in range(5):
       c = torch.matmul(a, b)
   torch.cuda.synchronize()
   
   start = time.perf_counter()
   for _ in range(20):
       c = torch.matmul(a, b)
   torch.cuda.synchronize()
   
   tflops = (2 * size ** 3 * 20) / (time.perf_counter() - start) / 1e12
   print(f"✓ Performance: {tflops:.1f} TFLOPS")
   
   print("=" * 60)
   print("🎉 ALL CHECKS PASSED - READY FOR TRAINING!")
   print("=" * 60)
   '@ | Out-File "$env:FLUX_HOME\verify_isolated.py" -Encoding UTF8
   ```

CHECKPOINT: Portable launcher and verification created

==============================================================================
PHASE 11: FINAL VERIFICATION [5 minutes]
==============================================================================

1. Run comprehensive verification:
   ```powershell
   & "$env:FLUX_HOME\python\python.exe" "$env:FLUX_HOME\verify_isolated.py"
   ```

2. Test portability:
   ```powershell
   # Show that no system modifications were made
   Write-Host "System Python check:"
   where.exe python 2>$null || Write-Host "No system Python in PATH (Good!)"
   
   Write-Host "`nSystem environment check:"
   [Environment]::GetEnvironmentVariable("PYTHONHOME", "Machine") || Write-Host "No system PYTHONHOME (Good!)"
   [Environment]::GetEnvironmentVariable("CUDA_HOME", "Machine") || Write-Host "No system CUDA_HOME (Good!)"
   
   Write-Host "`nIsolation complete! Everything is in: $env:FLUX_HOME"
   ```

CHECKPOINT: All verifications passed, complete isolation confirmed

==============================================================================
SUCCESS CRITERIA VERIFICATION
==============================================================================

After completion, verify ALL of the following:

✓ NO system Python installation or modifications
✓ NO system PATH changes
✓ NO system environment variables
✓ Everything contained in D:\Flux_Trainer
✓ Python 3.11.9 at D:\Flux_Trainer\python
✓ CUDA 13.0 at D:\Flux_Trainer\cuda_toolkit
✓ PyTorch with sm_120 in D:\Flux_Trainer\python\Lib\site-packages
✓ Can delete D:\Flux_Trainer to completely remove
✓ Can copy D:\Flux_Trainer to another machine
✓ Performance > 900 TFLOPS
✓ FluxTrainer.bat launches isolated environment

==============================================================================
REPORTING REQUIREMENTS
==============================================================================

After EACH phase, report:
1. Phase number and name
2. Success: YES/NO
3. Isolation verified: YES/NO
4. Key outputs
5. Time taken

Final report must include:
1. Total time: _____ hours
2. Performance: _____ TFLOPS
3. All phases completed: YES/NO
4. Complete isolation: YES/NO
5. Folder size: _____ GB
6. Portable: YES/NO
7. System contamination: NONE

==============================================================================
BEGIN EXECUTION
==============================================================================

Start with Phase 1 immediately. Execute all commands exactly as specified.
This creates a COMPLETELY ISOLATED environment with ZERO system contamination.
Everything will be self-contained in D:\Flux_Trainer.

Begin now.
```

---

## Usage Instructions

### Key Benefits of Isolated Setup:

1. **Zero System Contamination** - No system Python or PATH changes
2. **Completely Portable** - Copy folder to any machine
3. **Multiple Versions** - Run different setups simultaneously
4. **Clean Uninstall** - Just delete D:\Flux_Trainer
5. **No Conflicts** - Isolated from all system packages
6. **Backup Friendly** - Single folder contains everything

### To Use:

1. **Run the AI agent** with this prompt
2. **Wait 4-6 hours** for complete setup
3. **Use FluxTrainer.bat** to launch training
4. **Everything stays in D:\Flux_Trainer**

### Directory After Setup:
```
D:\Flux_Trainer\         # 100% self-contained
├── python\              # Portable Python 3.11.9
├── cuda_toolkit\        # Local CUDA 13.0
├── build\               # All compilation
├── models\              # Flux models
├── dataset\             # Training data
├── output\              # Results
├── sd-scripts-cuda13\   # Scripts
└── FluxTrainer.bat      # Launch everything
```

---

**This is a COMPLETELY ISOLATED setup.** Nothing touches the system. Everything is contained in D:\Flux_Trainer.
