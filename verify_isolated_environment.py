"""
Flux Trainer Isolated Environment Verification Script
Complete system checks for D:\\Flux_Trainer setup
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for Windows console
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{text}{Colors.RESET}")
    print("=" * 70)

def print_success(text: str):
    """Print success message in green"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message in yellow"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message in red"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_info(text: str):
    """Print info message in blue"""
    print(f"{Colors.BLUE}→ {text}{Colors.RESET}")

class IsolatedEnvironmentVerifier:
    def __init__(self, flux_home: str = "D:\\Flux_Trainer"):
        self.flux_home = Path(flux_home)
        self.results = {
            "passed": [],
            "warnings": [],
            "errors": []
        }
        
    def verify_all(self) -> bool:
        """Run all verification checks"""
        print_header("FLUX TRAINER ISOLATED ENVIRONMENT VERIFICATION")
        print(f"Environment Location: {self.flux_home}")
        
        # Run all checks
        self.check_directory_structure()
        self.check_python_isolation()
        self.check_cuda_isolation()
        self.check_pytorch_installation()
        self.check_dependencies()
        self.check_models()
        self.check_no_system_contamination()
        self.run_performance_benchmark()
        self.check_disk_usage()
        
        # Print summary
        self.print_summary()
        
        return len(self.results["errors"]) == 0
    
    def check_directory_structure(self):
        """Verify all required directories exist"""
        print_header("1. Directory Structure Check")
        
        required_dirs = {
            "python": "Python installation",
            "python/Lib": "Python libraries",
            "python/Lib/site-packages": "Python packages",
            "python/Scripts": "Python scripts",
            "cuda_toolkit": "CUDA toolkit",
            "cuda_toolkit/bin": "CUDA binaries",
            "models": "Flux models",
            "dataset": "Training datasets",
            "output": "Output directory",
            "samples": "Sample outputs",
            "cuda_kernels": "Custom kernels",
            "sd-scripts-cuda13": "Training scripts",
            "build": "Build directory",
            "tools": "Build tools",
            "temp": "Temporary files",
            "logs": "Log files"
        }
        
        for dir_path, description in required_dirs.items():
            full_path = self.flux_home / dir_path
            if full_path.exists():
                print_success(f"{description}: {dir_path}")
                self.results["passed"].append(f"Directory: {dir_path}")
            else:
                print_warning(f"Missing {description}: {dir_path}")
                self.results["warnings"].append(f"Missing directory: {dir_path}")
                # Try to create missing directory
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    print_info(f"Created: {dir_path}")
                except Exception as e:
                    print_error(f"Could not create {dir_path}: {e}")
    
    def check_python_isolation(self):
        """Verify Python is isolated and correct version"""
        print_header("2. Python Isolation Check")
        
        python_exe = self.flux_home / "python" / "python.exe"
        
        if not python_exe.exists():
            print_error(f"Python not found at {python_exe}")
            self.results["errors"].append("Python not installed in isolated environment")
            return
        
        # Check Python version
        try:
            result = subprocess.run(
                [str(python_exe), "--version"],
                capture_output=True,
                text=True
            )
            version = result.stdout.strip()
            
            if "3.11.9" in version:
                print_success(f"Python version: {version}")
                self.results["passed"].append("Python 3.11.9")
            else:
                print_warning(f"Python version mismatch: {version} (expected 3.11.9)")
                self.results["warnings"].append(f"Python version: {version}")
                
        except Exception as e:
            print_error(f"Could not check Python version: {e}")
            self.results["errors"].append("Python version check failed")
        
        # Verify isolation
        try:
            result = subprocess.run(
                [str(python_exe), "-c", 
                 "import sys; print(sys.executable); print('|'.join(sys.path))"],
                capture_output=True,
                text=True
            )
            
            lines = result.stdout.strip().split('\n')
            exe_path = lines[0] if lines else ""
            sys_paths = lines[1].split('|') if len(lines) > 1 else []
            
            # Check executable is in isolated environment
            if str(self.flux_home) in exe_path:
                print_success(f"Python executable: {exe_path}")
                self.results["passed"].append("Python is isolated")
            else:
                print_error(f"Python not isolated: {exe_path}")
                self.results["errors"].append("Python not isolated")
            
            # Check for system Python in paths
            system_paths = [p for p in sys_paths if "C:\\Python" in p or "C:\\Users" in p 
                           and "Flux_Trainer" not in p]
            
            if not system_paths:
                print_success("No system Python in path")
                self.results["passed"].append("Clean Python path")
            else:
                print_warning(f"System paths detected: {system_paths}")
                self.results["warnings"].append("System Python in path")
                
        except Exception as e:
            print_error(f"Could not verify Python isolation: {e}")
            self.results["errors"].append("Python isolation check failed")
    
    def check_cuda_isolation(self):
        """Verify CUDA toolkit is isolated"""
        print_header("3. CUDA Isolation Check")
        
        cuda_dir = self.flux_home / "cuda_toolkit"
        nvcc_exe = cuda_dir / "bin" / "nvcc.exe"
        
        if not cuda_dir.exists():
            print_warning("CUDA toolkit not found in isolated environment")
            print_info("You may need to copy CUDA 13.0 to the isolated environment")
            self.results["warnings"].append("CUDA not isolated")
            return
        
        if nvcc_exe.exists():
            try:
                result = subprocess.run(
                    [str(nvcc_exe), "--version"],
                    capture_output=True,
                    text=True
                )
                
                if "release 13.0" in result.stdout:
                    print_success("CUDA 13.0 found in isolated environment")
                    self.results["passed"].append("CUDA 13.0 isolated")
                else:
                    version = "unknown"
                    for line in result.stdout.split('\n'):
                        if "release" in line:
                            version = line.strip()
                            break
                    print_warning(f"CUDA version: {version} (expected 13.0)")
                    self.results["warnings"].append(f"CUDA version: {version}")
                    
            except Exception as e:
                print_error(f"Could not check CUDA version: {e}")
                self.results["errors"].append("CUDA check failed")
        else:
            print_warning(f"nvcc.exe not found at {nvcc_exe}")
            self.results["warnings"].append("nvcc not found")
    
    def check_pytorch_installation(self):
        """Verify PyTorch with sm_120 support"""
        print_header("4. PyTorch Installation Check")
        
        python_exe = self.flux_home / "python" / "python.exe"
        
        if not python_exe.exists():
            print_error("Cannot check PyTorch - Python not found")
            return
        
        try:
            # Check PyTorch installation
            result = subprocess.run(
                [str(python_exe), "-c", 
                 "import torch; print(f'version:{torch.__version__}|cuda:{torch.cuda.is_available()}|sm120:{\"sm_120\" in str(torch.cuda.get_arch_list())}')"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                parts = {k: v for k, v in [p.split(':') for p in output.split('|')]}
                
                print_success(f"PyTorch version: {parts.get('version', 'unknown')}")
                
                if parts.get('cuda') == 'True':
                    print_success("CUDA available in PyTorch")
                    self.results["passed"].append("PyTorch CUDA support")
                else:
                    print_error("CUDA not available in PyTorch")
                    self.results["errors"].append("No CUDA in PyTorch")
                
                if parts.get('sm120') == 'True':
                    print_success("Native sm_120 support confirmed")
                    self.results["passed"].append("sm_120 support")
                else:
                    print_error("Missing sm_120 support (needs recompilation)")
                    self.results["errors"].append("No sm_120 support")
                    
            else:
                print_error("PyTorch not installed or not working")
                self.results["errors"].append("PyTorch not installed")
                
        except subprocess.TimeoutExpired:
            print_error("PyTorch check timed out")
            self.results["errors"].append("PyTorch check timeout")
        except Exception as e:
            print_error(f"Could not check PyTorch: {e}")
            self.results["errors"].append("PyTorch check failed")
    
    def check_dependencies(self):
        """Check critical dependencies"""
        print_header("5. Dependencies Check")
        
        python_exe = self.flux_home / "python" / "python.exe"
        
        if not python_exe.exists():
            print_error("Cannot check dependencies - Python not found")
            return
        
        critical_packages = [
            "torch",
            "xformers",
            "transformers",
            "diffusers",
            "accelerate",
            "safetensors",
            "prodigyopt"
        ]
        
        for package in critical_packages:
            try:
                result = subprocess.run(
                    [str(python_exe), "-c", f"import {package}; print({package}.__version__)"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    version = result.stdout.strip()
                    print_success(f"{package}: {version}")
                    self.results["passed"].append(f"{package} installed")
                else:
                    print_warning(f"{package}: not installed")
                    self.results["warnings"].append(f"{package} missing")
                    
            except subprocess.TimeoutExpired:
                print_warning(f"{package}: check timed out")
            except Exception:
                print_warning(f"{package}: not installed")
                self.results["warnings"].append(f"{package} missing")
    
    def check_models(self):
        """Check if Flux models are present"""
        print_header("6. Model Files Check")
        
        models_dir = self.flux_home / "models"
        
        required_models = {
            "flux1-dev.safetensors": 23 * 1024**3,  # 23GB
            "ae.safetensors": 335 * 1024**2,  # 335MB
            "clip_l.safetensors": 246 * 1024**2,  # 246MB
            "t5xxl_fp16.safetensors": 9.5 * 1024**3  # 9.5GB
        }
        
        models_found = 0
        total_size = 0
        
        for model_name, expected_size in required_models.items():
            model_path = models_dir / model_name
            
            if model_path.exists():
                actual_size = model_path.stat().st_size
                total_size += actual_size
                size_gb = actual_size / (1024**3)
                
                if abs(actual_size - expected_size) / expected_size < 0.1:  # 10% tolerance
                    print_success(f"{model_name}: {size_gb:.1f}GB")
                    models_found += 1
                else:
                    print_warning(f"{model_name}: {size_gb:.1f}GB (size mismatch)")
                    models_found += 1
            else:
                print_warning(f"{model_name}: not found")
                print_info(f"Download from Hugging Face to {models_dir}")
        
        if models_found == len(required_models):
            self.results["passed"].append("All models present")
        elif models_found > 0:
            self.results["warnings"].append(f"{models_found}/{len(required_models)} models found")
        else:
            self.results["warnings"].append("No models found")
        
        if total_size > 0:
            print_info(f"Total model size: {total_size / (1024**3):.1f}GB")
    
    def check_no_system_contamination(self):
        """Verify no system modifications were made"""
        print_header("7. System Contamination Check")
        
        contamination_found = False
        
        # Check system PATH for Python
        try:
            result = subprocess.run(
                ["where", "python"],
                capture_output=True,
                text=True,
                shell=True
            )
            
            if result.returncode == 0 and result.stdout:
                system_pythons = [p for p in result.stdout.strip().split('\n') 
                                 if "Flux_Trainer" not in p]
                if system_pythons:
                    print_info(f"System Python found: {system_pythons[0]} (OK - not our concern)")
                else:
                    print_success("No system Python in PATH (completely isolated)")
            else:
                print_success("No Python in system PATH (perfect isolation)")
                
        except Exception:
            print_info("Could not check system PATH")
        
        # Check system environment variables
        env_vars = ["PYTHONHOME", "PYTHONPATH", "CUDA_HOME", "CUDA_PATH"]
        
        for var in env_vars:
            value = os.environ.get(var, "")
            if value and "Flux_Trainer" not in value:
                print_info(f"System {var}: {value} (not from our setup)")
            elif value and "Flux_Trainer" in value:
                print_warning(f"{var} points to Flux_Trainer (should be session-only)")
                contamination_found = True
            else:
                print_success(f"No system {var} set")
        
        if not contamination_found:
            print_success("No system contamination detected")
            self.results["passed"].append("Clean system")
        else:
            self.results["warnings"].append("Possible system contamination")
    
    def run_performance_benchmark(self):
        """Run a quick performance benchmark"""
        print_header("8. Performance Benchmark")
        
        python_exe = self.flux_home / "python" / "python.exe"
        
        if not python_exe.exists():
            print_error("Cannot run benchmark - Python not found")
            return
        
        benchmark_code = """
import torch
import time

if not torch.cuda.is_available():
    print("error:CUDA not available")
else:
    # Quick TFLOPS test
    size = 4096
    device = torch.cuda.current_device()
    
    # Get GPU name
    gpu_name = torch.cuda.get_device_name(device)
    print(f"gpu:{gpu_name}")
    
    # Create tensors
    a = torch.randn(size, size, dtype=torch.bfloat16).cuda()
    b = torch.randn(size, size, dtype=torch.bfloat16).cuda()
    
    # Warmup
    for _ in range(5):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    tflops = (2 * size ** 3 * iterations) / elapsed / 1e12
    
    print(f"tflops:{tflops:.1f}")
    
    # Memory info
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"memory:{allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
"""
        
        try:
            result = subprocess.run(
                [str(python_exe), "-c", benchmark_code],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith("gpu:"):
                        gpu = line.split(':')[1]
                        print_info(f"GPU: {gpu}")
                        
                    elif line.startswith("tflops:"):
                        tflops = float(line.split(':')[1])
                        if tflops > 900:
                            print_success(f"Performance: {tflops:.1f} TFLOPS (Excellent!)")
                            self.results["passed"].append(f"{tflops:.1f} TFLOPS")
                        elif tflops > 500:
                            print_warning(f"Performance: {tflops:.1f} TFLOPS (Below expected)")
                            self.results["warnings"].append(f"{tflops:.1f} TFLOPS")
                        else:
                            print_error(f"Performance: {tflops:.1f} TFLOPS (Too low)")
                            self.results["errors"].append(f"Low performance: {tflops:.1f} TFLOPS")
                            
                    elif line.startswith("memory:"):
                        print_info(f"GPU Memory: {line.split(':')[1]}")
                        
                    elif line.startswith("error:"):
                        print_error(line.split(':')[1])
                        self.results["errors"].append("Benchmark failed")
                        
            else:
                print_error("Benchmark failed to run")
                self.results["errors"].append("Benchmark error")
                
        except subprocess.TimeoutExpired:
            print_error("Benchmark timed out")
            self.results["errors"].append("Benchmark timeout")
        except Exception as e:
            print_error(f"Could not run benchmark: {e}")
            self.results["errors"].append("Benchmark failed")
    
    def check_disk_usage(self):
        """Check disk space usage"""
        print_header("9. Disk Usage Check")
        
        if not self.flux_home.exists():
            print_error(f"Environment directory not found: {self.flux_home}")
            return
        
        try:
            total_size = 0
            file_count = 0
            
            # Calculate size
            for item in self.flux_home.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
            
            size_gb = total_size / (1024**3)
            
            print_info(f"Total files: {file_count:,}")
            print_info(f"Total size: {size_gb:.1f}GB")
            
            # Check available space on D: drive
            import shutil
            total, used, free = shutil.disk_usage("D:\\")
            free_gb = free / (1024**3)
            
            if free_gb > 50:
                print_success(f"Free space on D: drive: {free_gb:.1f}GB")
            elif free_gb > 20:
                print_warning(f"Low free space on D: drive: {free_gb:.1f}GB")
                self.results["warnings"].append(f"Low disk space: {free_gb:.1f}GB")
            else:
                print_error(f"Critical: Only {free_gb:.1f}GB free on D: drive")
                self.results["errors"].append(f"Insufficient disk space: {free_gb:.1f}GB")
                
        except Exception as e:
            print_error(f"Could not check disk usage: {e}")
    
    def print_summary(self):
        """Print final summary"""
        print_header("VERIFICATION SUMMARY")
        
        passed = len(self.results["passed"])
        warnings = len(self.results["warnings"])
        errors = len(self.results["errors"])
        
        print(f"\n{Colors.GREEN}✓ Passed: {passed}{Colors.RESET}")
        if passed > 0:
            for item in self.results["passed"][:5]:
                print(f"  • {item}")
            if passed > 5:
                print(f"  • ... and {passed - 5} more")
        
        if warnings > 0:
            print(f"\n{Colors.YELLOW}⚠ Warnings: {warnings}{Colors.RESET}")
            for item in self.results["warnings"]:
                print(f"  • {item}")
        
        if errors > 0:
            print(f"\n{Colors.RED}✗ Errors: {errors}{Colors.RESET}")
            for item in self.results["errors"]:
                print(f"  • {item}")
        
        print("\n" + "=" * 70)
        
        if errors == 0 and warnings == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}🎉 PERFECT! Your isolated environment is ready!{Colors.RESET}")
            print(f"{Colors.GREEN}Location: {self.flux_home}{Colors.RESET}")
            print(f"{Colors.GREEN}No system contamination detected.{Colors.RESET}")
        elif errors == 0:
            print(f"{Colors.YELLOW}{Colors.BOLD}✓ Environment is functional with minor issues.{Colors.RESET}")
            print(f"{Colors.YELLOW}Review warnings above for optimal setup.{Colors.RESET}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ Environment needs attention!{Colors.RESET}")
            print(f"{Colors.RED}Please fix errors before training.{Colors.RESET}")
        
        print("=" * 70)
        
        # Save results to file
        self.save_results()
    
    def save_results(self):
        """Save verification results to JSON file"""
        results_file = self.flux_home / "verification_results.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\nResults saved to: {results_file}")
        except Exception as e:
            print(f"\nCould not save results: {e}")


def main():
    """Main verification entry point"""
    print(f"{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     FLUX TRAINER ISOLATED ENVIRONMENT VERIFICATION          ║")
    print("║                  Version 2.0 - Zero System Impact           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    
    # Check if custom path provided
    import sys
    flux_home = sys.argv[1] if len(sys.argv) > 1 else "D:\\Flux_Trainer"
    
    # Run verification
    verifier = IsolatedEnvironmentVerifier(flux_home)
    success = verifier.verify_all()
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
