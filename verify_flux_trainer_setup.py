"""
RTX 5090 FLUX Training Environment Verification Script
Verifies all paths and setup for D:\Flux_Trainer environment
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """
    Comprehensive environment verification for D:\Flux_Trainer setup
    """
    
    print("=" * 60)
    print("RTX 5090 FLUX TRAINING ENVIRONMENT VERIFICATION")
    print("Training Environment: D:\\Flux_Trainer")
    print("=" * 60)
    
    checks = {
        "passed": [],
        "failed": []
    }
    
    # 1. Check Python version
    print("\n1. Checking Python installation...")
    python_version = sys.version_info
    if python_version[:3] == (3, 11, 9):
        checks["passed"].append("✅ Python 3.11.9 installed at C:\\Python311")
        print("   ✅ Python 3.11.9 detected")
    else:
        checks["failed"].append("❌ Python version is not 3.11.9")
        print(f"   ❌ Wrong Python version: {sys.version}")
    
    # 2. Check CUDA installation
    print("\n2. Checking CUDA 13.0...")
    cuda_path = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")
    if cuda_path.exists():
        checks["passed"].append("✅ CUDA 13.0 directory found")
        print("   ✅ CUDA 13.0 directory found")
        
        # Check nvcc
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            if "release 13.0" in result.stdout:
                checks["passed"].append("✅ CUDA 13.0 nvcc verified")
                print("   ✅ CUDA 13.0 nvcc verified")
            else:
                checks["failed"].append("❌ CUDA version mismatch")
                print("   ❌ CUDA version mismatch")
        except:
            checks["failed"].append("❌ nvcc not accessible")
            print("   ❌ nvcc not accessible")
    else:
        checks["failed"].append("❌ CUDA 13.0 not found")
        print("   ❌ CUDA 13.0 not found at expected location")
    
    # 3. Check D: drive and training environment
    print("\n3. Checking D:\\Flux_Trainer environment...")
    
    required_dirs = {
        "D:/Flux_Trainer": "Main training directory",
        "D:/Flux_Trainer/models": "Model storage",
        "D:/Flux_Trainer/dataset": "Dataset storage",
        "D:/Flux_Trainer/output": "Output directory",
        "D:/Flux_Trainer/samples": "Sample outputs",
        "D:/Flux_Trainer/cuda_kernels": "Custom kernels",
        "D:/Flux_Trainer/sd-scripts-cuda13": "Training scripts"
    }
    
    for dir_path, description in required_dirs.items():
        path = Path(dir_path)
        if path.exists():
            checks["passed"].append(f"✅ {description}: {dir_path}")
            print(f"   ✅ {description} exists")
        else:
            checks["failed"].append(f"❌ Missing: {dir_path}")
            print(f"   ❌ Missing: {description}")
            # Try to create missing directories
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"      📁 Created: {dir_path}")
            except Exception as e:
                print(f"      ⚠️ Could not create: {e}")
    
    # 4. Check PyTorch installation
    print("\n4. Checking PyTorch with sm_120 support...")
    try:
        import torch
        if torch.cuda.is_available():
            if 'sm_120' in str(torch.cuda.get_arch_list()):
                checks["passed"].append("✅ PyTorch with native sm_120 support")
                print("   ✅ PyTorch with native sm_120 support")
            else:
                checks["failed"].append("❌ PyTorch missing sm_120 support")
                print("   ❌ PyTorch missing sm_120 support (needs recompilation)")
        else:
            checks["failed"].append("❌ CUDA not available in PyTorch")
            print("   ❌ CUDA not available in PyTorch")
    except ImportError:
        checks["failed"].append("❌ PyTorch not installed")
        print("   ❌ PyTorch not installed")
    
    # 5. Check model files
    print("\n5. Checking Flux model files...")
    model_files = {
        "flux1-dev.safetensors": 23 * 1024**3,  # 23GB
        "ae.safetensors": 335 * 1024**2,  # 335MB
        "clip_l.safetensors": 246 * 1024**2,  # 246MB
        "t5xxl_fp16.safetensors": 9.5 * 1024**3  # 9.5GB
    }
    
    models_dir = Path("D:/Flux_Trainer/models")
    if models_dir.exists():
        for model_file, expected_size in model_files.items():
            model_path = models_dir / model_file
            if model_path.exists():
                actual_size = model_path.stat().st_size
                size_gb = actual_size / (1024**3)
                checks["passed"].append(f"✅ Model: {model_file} ({size_gb:.1f}GB)")
                print(f"   ✅ {model_file} found ({size_gb:.1f}GB)")
            else:
                checks["failed"].append(f"❌ Missing model: {model_file}")
                print(f"   ❌ Missing: {model_file}")
                print(f"      Download from Hugging Face to D:\\Flux_Trainer\\models")
    
    # 6. Check virtual environment
    print("\n6. Checking virtual environment...")
    venv_path = Path("D:/Flux_Trainer/sd-scripts-cuda13/venv")
    if venv_path.exists():
        checks["passed"].append("✅ Virtual environment exists")
        print("   ✅ Virtual environment exists")
    else:
        checks["failed"].append("❌ Virtual environment not created")
        print("   ❌ Virtual environment not created")
        print("      Run: python -m venv D:\\Flux_Trainer\\sd-scripts-cuda13\\venv")
    
    # 7. Check critical dependencies
    print("\n7. Checking critical dependencies...")
    try:
        import prodigyopt
        checks["passed"].append("✅ Prodigy optimizer installed")
        print("   ✅ Prodigy optimizer installed")
    except ImportError:
        checks["failed"].append("❌ Prodigy optimizer not installed")
        print("   ❌ Prodigy optimizer not installed (pip install prodigyopt)")
    
    # 8. Performance benchmark (optional)
    print("\n8. Running performance benchmark...")
    try:
        import torch
        import time
        
        if torch.cuda.is_available():
            # Quick TFLOPS test
            size = 4096
            a = torch.randn(size, size, dtype=torch.bfloat16).cuda()
            b = torch.randn(size, size, dtype=torch.bfloat16).cuda()
            
            # Warmup
            for _ in range(5):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(20):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            tflops = (2 * size ** 3 * 20) / (time.perf_counter() - start) / 1e12
            
            if tflops > 900:
                checks["passed"].append(f"✅ Performance: {tflops:.1f} TFLOPS")
                print(f"   ✅ Performance: {tflops:.1f} TFLOPS")
            else:
                checks["failed"].append(f"⚠️ Performance: {tflops:.1f} TFLOPS (expected >900)")
                print(f"   ⚠️ Performance: {tflops:.1f} TFLOPS (expected >900)")
    except Exception as e:
        print(f"   ⚠️ Could not run benchmark: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ Passed checks: {len(checks['passed'])}")
    for check in checks["passed"][:5]:  # Show first 5
        print(f"   {check}")
    if len(checks["passed"]) > 5:
        print(f"   ... and {len(checks['passed']) - 5} more")
    
    if checks["failed"]:
        print(f"\n❌ Failed checks: {len(checks['failed'])}")
        for check in checks["failed"]:
            print(f"   {check}")
        
        print("\n⚠️ ACTIONS REQUIRED:")
        print("1. Review failed checks above")
        print("2. Follow the setup guide to resolve issues")
        print("3. Ensure D: drive has 100GB+ free space")
        print("4. Run this script again after fixes")
    else:
        print("\n🎉 ALL CHECKS PASSED!")
        print("Your D:\\Flux_Trainer environment is ready for training!")
        print("\nNext steps:")
        print("1. Prepare your dataset in D:\\Flux_Trainer\\dataset")
        print("2. Choose a config (face/action/style)")
        print("3. Start training from D:\\Flux_Trainer\\sd-scripts-cuda13")
    
    return len(checks["failed"]) == 0

def create_launch_script():
    """
    Create a convenient launch script for training
    """
    
    launch_script = '''@echo off
echo ========================================
echo RTX 5090 FLUX LoRA Training Launcher
echo Training Environment: D:\\Flux_Trainer
echo ========================================
echo.

:: Check if D: drive exists
if not exist D:\\ (
    echo ERROR: D: drive not found!
    echo Please ensure D: drive is available.
    pause
    exit /b 1
)

:: Navigate to training directory
cd /d D:\\Flux_Trainer\\sd-scripts-cuda13

:: Activate virtual environment
call venv\\Scripts\\activate.bat

:: Set CUDA environment
set CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

:: Display options
echo Select training type:
echo 1. Face Identity (1500 steps, dim=128)
echo 2. Action/Pose (1000 steps, dim=48)
echo 3. Style/Object (800 steps, dim=32)
echo 4. Custom config
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    set config=config_face.toml
    echo Starting Face Identity training...
) else if "%choice%"=="2" (
    set config=config_action.toml
    echo Starting Action/Pose training...
) else if "%choice%"=="3" (
    set config=config_style.toml
    echo Starting Style/Object training...
) else if "%choice%"=="4" (
    set /p config="Enter config filename: "
    echo Using custom config: %config%
) else (
    echo Invalid choice!
    pause
    exit /b 1
)

:: Start training
echo.
echo Starting training with %config%...
echo Output will be saved to: D:\\Flux_Trainer\\output
echo Samples will be saved to: D:\\Flux_Trainer\\samples
echo.

python flux_train_network.py --config_file %config% --highvram

echo.
echo Training complete!
echo Check D:\\Flux_Trainer\\output for your LoRA file.
pause
'''
    
    # Save launch script
    script_path = Path("D:/Flux_Trainer/launch_training.bat")
    try:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(launch_script)
        print(f"\n✅ Launch script created: {script_path}")
        print("   You can now double-click launch_training.bat to start training!")
    except Exception as e:
        print(f"\n⚠️ Could not create launch script: {e}")

if __name__ == "__main__":
    print("Starting environment verification...\n")
    
    # Run verification
    success = check_environment()
    
    # Create launch script if successful
    if success:
        create_launch_script()
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    
    # Keep window open
    input("\nPress Enter to exit...")
