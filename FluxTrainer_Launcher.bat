@echo off
:: ============================================================================
:: FLUX TRAINER PORTABLE LAUNCHER v2.0
:: Completely Isolated Environment - Zero System Contamination
:: Everything contained in D:\Flux_Trainer
:: ============================================================================

setlocal enabledelayedexpansion

:: Check if running from correct location
if not exist "D:\Flux_Trainer\python\python.exe" (
    color 0C
    echo ERROR: Flux Trainer environment not found at D:\Flux_Trainer
    echo.
    echo Please ensure the isolated environment is installed at:
    echo D:\Flux_Trainer\
    echo.
    echo Run the setup script first or check the installation path.
    pause
    exit /b 1
)

:: ============================================================================
:: ISOLATED ENVIRONMENT CONFIGURATION
:: ============================================================================

:: Base directory (everything is here)
set FLUX_HOME=D:\Flux_Trainer

:: Python configuration (isolated)
set PYTHONHOME=%FLUX_HOME%\python
set PYTHONPATH=%FLUX_HOME%\python\Lib;%FLUX_HOME%\python\Lib\site-packages
set PYTHON_EXE=%FLUX_HOME%\python\python.exe

:: CUDA configuration (isolated)
set CUDA_HOME=%FLUX_HOME%\cuda_toolkit
set CUDA_PATH=%FLUX_HOME%\cuda_toolkit
set CUDNN_PATH=%FLUX_HOME%\cuda_toolkit

:: Build tools (if needed)
set DISTUTILS_USE_SDK=1
set VS_PATH=%FLUX_HOME%\tools\BuildTools\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64

:: Isolated PATH (no system contamination)
set PATH=%PYTHONHOME%;%PYTHONHOME%\Scripts;%CUDA_HOME%\bin;%VS_PATH%

:: PyTorch configuration
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set TORCH_CUDA_ARCH_LIST=8.9;9.0;12.0

:: Model and data paths
set MODEL_DIR=%FLUX_HOME%\models
set DATASET_DIR=%FLUX_HOME%\dataset
set OUTPUT_DIR=%FLUX_HOME%\output
set SAMPLE_DIR=%FLUX_HOME%\samples

:: ============================================================================
:: STARTUP CHECKS
:: ============================================================================

cls
color 0A
echo ============================================================================
echo                    FLUX TRAINER ISOLATED ENVIRONMENT
echo                         RTX 5090 - CUDA 13.0
echo                    Everything in: %FLUX_HOME%
echo ============================================================================
echo.

:: Check Python
echo [1/4] Checking Python...
%PYTHON_EXE% --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo ERROR: Python not found at %PYTHON_EXE%
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('%PYTHON_EXE% --version 2^>^&1') do set PYTHON_VER=%%i
echo       Found: Python %PYTHON_VER% [Isolated]

:: Check CUDA
echo [2/4] Checking CUDA...
if exist "%CUDA_HOME%\bin\nvcc.exe" (
    for /f "tokens=5" %%i in ('"%CUDA_HOME%\bin\nvcc.exe" --version ^| findstr "release"') do set CUDA_VER=%%i
    echo       Found: CUDA !CUDA_VER! [Isolated]
) else (
    echo       WARNING: CUDA toolkit not found in isolated environment
)

:: Check PyTorch
echo [3/4] Checking PyTorch...
%PYTHON_EXE% -c "import torch; print(f'      Found: PyTorch {torch.__version__} [sm_120: {\"sm_120\" in str(torch.cuda.get_arch_list())}]')" 2>nul
if %errorlevel% neq 0 (
    echo       WARNING: PyTorch not installed or not working
)

:: Check Models
echo [4/4] Checking Models...
set MODEL_COUNT=0
if exist "%MODEL_DIR%\flux1-dev.safetensors" set /a MODEL_COUNT+=1
if exist "%MODEL_DIR%\ae.safetensors" set /a MODEL_COUNT+=1
if exist "%MODEL_DIR%\clip_l.safetensors" set /a MODEL_COUNT+=1
if exist "%MODEL_DIR%\t5xxl_fp16.safetensors" set /a MODEL_COUNT+=1
echo       Found: %MODEL_COUNT%/4 models in %MODEL_DIR%

echo.
echo ============================================================================
echo                            MAIN MENU
echo ============================================================================
echo.
echo   TRAINING OPTIONS:
echo   [1] Train Face Identity LoRA    (128 dim, 1500 steps, ~90 min)
echo   [2] Train Action/Pose LoRA      (48 dim, 1000 steps, ~60 min)
echo   [3] Train Style/Object LoRA     (32 dim, 800 steps, ~45 min)
echo   [4] Resume Previous Training
echo   [5] Custom Configuration
echo.
echo   TOOLS & UTILITIES:
echo   [6] Prepare Dataset             (Resize, caption, organize)
echo   [7] Test LoRA                   (Generate samples)
echo   [8] Merge LoRAs                 (Combine multiple LoRAs)
echo   [9] Environment Info            (Detailed system check)
echo.
echo   MANAGEMENT:
echo   [V] Verify Installation         (Run all checks)
echo   [C] Command Prompt              (Isolated shell)
echo   [I] Install Package             (pip install in isolated env)
echo   [B] Backup Configuration        (Save settings)
echo   [X] Exit
echo.
echo ============================================================================
echo.

set /p choice="Select option: "

:: Convert to uppercase
set choice=%choice:~0,1%
if /i "%choice%"=="1" goto train_face
if /i "%choice%"=="2" goto train_action
if /i "%choice%"=="3" goto train_style
if /i "%choice%"=="4" goto resume_training
if /i "%choice%"=="5" goto custom_config
if /i "%choice%"=="6" goto prepare_dataset
if /i "%choice%"=="7" goto test_lora
if /i "%choice%"=="8" goto merge_loras
if /i "%choice%"=="9" goto env_info
if /i "%choice%"=="V" goto verify
if /i "%choice%"=="C" goto shell
if /i "%choice%"=="I" goto install_package
if /i "%choice%"=="B" goto backup
if /i "%choice%"=="X" goto exit_script

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto startup_checks

:: ============================================================================
:: TRAINING FUNCTIONS
:: ============================================================================

:train_face
cls
echo ============================================================================
echo                    FACE IDENTITY LORA TRAINING
echo ============================================================================
echo.
echo Configuration:
echo - Network Dimension: 128 (high for facial details)
echo - Alpha: 64
echo - Optimizer: Prodigy (self-adjusting LR)
echo - Steps: 1500
echo - Expected Time: ~90 minutes
echo.
echo Dataset Requirements:
echo - Location: %DATASET_DIR%\[name]\[repeats]_[trigger]
echo - Images: 15-25 photos of the same person
echo - Resolution: 1024x1024
echo - Captions: Just the trigger word (e.g., "johndoe")
echo.

set /p dataset_name="Enter dataset folder name (or press Enter to cancel): "
if "%dataset_name%"=="" goto startup_checks

if not exist "%DATASET_DIR%\%dataset_name%" (
    echo ERROR: Dataset not found at %DATASET_DIR%\%dataset_name%
    pause
    goto startup_checks
)

set /p trigger="Enter trigger word: "
if "%trigger%"=="" goto startup_checks

echo.
echo Starting training with:
echo - Dataset: %dataset_name%
echo - Trigger: %trigger%
echo - Output: %OUTPUT_DIR%\flux_lora_face_%trigger%.safetensors
echo.
pause

cd /d %FLUX_HOME%\sd-scripts-cuda13
call venv\Scripts\activate
python flux_train_network.py --config_file config_face.toml ^
    --train_data_dir="%DATASET_DIR%\%dataset_name%" ^
    --output_name="flux_lora_face_%trigger%" ^
    --output_dir="%OUTPUT_DIR%" ^
    --sample_prompts="%trigger%, portrait" ^
    --highvram
pause
goto startup_checks

:train_action
cls
echo ============================================================================
echo                    ACTION/POSE LORA TRAINING
echo ============================================================================
echo.
echo Configuration:
echo - Network Dimension: 48 (medium for flexibility)
echo - Alpha: 24
echo - Optimizer: Prodigy with cosine scheduler
echo - Steps: 1000
echo - Expected Time: ~60 minutes
echo.

cd /d %FLUX_HOME%\sd-scripts-cuda13
call venv\Scripts\activate
python flux_train_network.py --config_file config_action.toml --highvram
pause
goto startup_checks

:train_style
cls
echo ============================================================================
echo                    STYLE/OBJECT LORA TRAINING
echo ============================================================================
echo.
echo Configuration:
echo - Network Dimension: 32 (lower for styles)
echo - Alpha: 16
echo - Optimizer: AdamW with cosine restarts
echo - Steps: 800
echo - Expected Time: ~45 minutes
echo.

cd /d %FLUX_HOME%\sd-scripts-cuda13
call venv\Scripts\activate
python flux_train_network.py --config_file config_style.toml --highvram
pause
goto startup_checks

:resume_training
cls
echo ============================================================================
echo                    RESUME TRAINING
echo ============================================================================
echo.
echo Available checkpoints:
dir /b "%OUTPUT_DIR%\*.safetensors" 2>nul
echo.
set /p checkpoint="Enter checkpoint filename to resume from: "
if exist "%OUTPUT_DIR%\%checkpoint%" (
    cd /d %FLUX_HOME%\sd-scripts-cuda13
    call venv\Scripts\activate
    python flux_train_network.py --resume="%OUTPUT_DIR%\%checkpoint%" --highvram
) else (
    echo Checkpoint not found!
)
pause
goto startup_checks

:custom_config
cls
echo ============================================================================
echo                    CUSTOM CONFIGURATION
echo ============================================================================
echo.
echo Available configs:
dir /b "%FLUX_HOME%\sd-scripts-cuda13\*.toml" 2>nul
echo.
set /p config="Enter config filename: "
if exist "%FLUX_HOME%\sd-scripts-cuda13\%config%" (
    cd /d %FLUX_HOME%\sd-scripts-cuda13
    call venv\Scripts\activate
    python flux_train_network.py --config_file "%config%" --highvram
) else (
    echo Config not found!
)
pause
goto startup_checks

:: ============================================================================
:: UTILITY FUNCTIONS
:: ============================================================================

:prepare_dataset
cls
echo ============================================================================
echo                    DATASET PREPARATION
echo ============================================================================
%PYTHON_EXE% %FLUX_HOME%\prepare_dataset.py
pause
goto startup_checks

:test_lora
cls
echo ============================================================================
echo                    TEST LORA
echo ============================================================================
echo.
echo Available LoRAs:
dir /b "%OUTPUT_DIR%\*.safetensors" 2>nul
echo.
set /p lora="Enter LoRA filename: "
set /p prompt="Enter test prompt: "

cd /d %FLUX_HOME%\sd-scripts-cuda13
call venv\Scripts\activate
python flux_minimal_inference.py ^
    --ckpt "%MODEL_DIR%\flux1-dev.safetensors" ^
    --clip_l "%MODEL_DIR%\clip_l.safetensors" ^
    --t5xxl "%MODEL_DIR%\t5xxl_fp16.safetensors" ^
    --ae "%MODEL_DIR%\ae.safetensors" ^
    --lora "%OUTPUT_DIR%\%lora%" ^
    --prompt "%prompt%" ^
    --output "%SAMPLE_DIR%\test_output.png" ^
    --seed 42 ^
    --steps 20 ^
    --guidance 3.5

if exist "%SAMPLE_DIR%\test_output.png" (
    echo.
    echo Image saved to: %SAMPLE_DIR%\test_output.png
    start "" "%SAMPLE_DIR%\test_output.png"
)
pause
goto startup_checks

:merge_loras
cls
echo ============================================================================
echo                    MERGE LORAS
echo ============================================================================
echo Coming soon...
pause
goto startup_checks

:env_info
cls
echo ============================================================================
echo                    ENVIRONMENT INFORMATION
echo ============================================================================
echo.
echo Paths:
echo   FLUX_HOME:     %FLUX_HOME%
echo   Python:        %PYTHONHOME%
echo   CUDA:          %CUDA_HOME%
echo   Models:        %MODEL_DIR%
echo   Dataset:       %DATASET_DIR%
echo   Output:        %OUTPUT_DIR%
echo.
echo Python Packages:
%PYTHON_EXE% -m pip list | findstr /i "torch xformers transformers diffusers"
echo.
echo GPU Information:
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
echo.
echo Disk Usage:
for /f "tokens=3" %%a in ('dir %FLUX_HOME% /s /-c ^| findstr "File(s)"') do set TOTAL_SIZE=%%a
set /a TOTAL_GB=%TOTAL_SIZE:~0,-9%/1
echo   Total Size: ~%TOTAL_GB% GB
echo.
pause
goto startup_checks

:verify
cls
echo ============================================================================
echo                    VERIFICATION
echo ============================================================================
echo.
%PYTHON_EXE% %FLUX_HOME%\verify_isolated.py
pause
goto startup_checks

:shell
cls
echo ============================================================================
echo                    ISOLATED COMMAND PROMPT
echo ============================================================================
echo.
echo You are now in the isolated environment.
echo - Python: %PYTHON_EXE%
echo - pip: %PYTHONHOME%\Scripts\pip.exe
echo - CUDA: %CUDA_HOME%
echo.
echo Type 'exit' to return to menu.
echo.
cmd /k
goto startup_checks

:install_package
cls
echo ============================================================================
echo                    INSTALL PACKAGE
echo ============================================================================
echo.
set /p package="Enter package name to install: "
%PYTHON_EXE% -m pip install %package% --no-warn-script-location
pause
goto startup_checks

:backup
cls
echo ============================================================================
echo                    BACKUP CONFIGURATION
echo ============================================================================
echo.
echo Creating backup...
set backup_date=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%
set backup_date=%backup_date: =0%
mkdir "%FLUX_HOME%\backups\%backup_date%" 2>nul

echo Backing up configurations...
xcopy "%FLUX_HOME%\sd-scripts-cuda13\*.toml" "%FLUX_HOME%\backups\%backup_date%\" /Y >nul
xcopy "%FLUX_HOME%\*.conf" "%FLUX_HOME%\backups\%backup_date%\" /Y >nul
xcopy "%FLUX_HOME%\*.bat" "%FLUX_HOME%\backups\%backup_date%\" /Y >nul

echo Backup created: %FLUX_HOME%\backups\%backup_date%
pause
goto startup_checks

:exit_script
cls
echo.
echo Thank you for using Flux Trainer Isolated Environment!
echo.
echo Remember: Everything is self-contained in %FLUX_HOME%
echo No system modifications were made.
echo.
timeout /t 3 >nul
exit /b 0
