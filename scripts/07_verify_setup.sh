#!/bin/bash
###############################################################################
# Complete Setup Verification
# Ensures all components are correctly installed for RTX 5090
###############################################################################

set -e

echo "============================================================"
echo "RTX 5090 + CUDA 13.0 SETUP VERIFICATION"
echo "============================================================"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"

checks_passed=0
checks_failed=0
checks_warning=0

# Activate venv if available
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo ""
echo -e "${BLUE}=== System Information ===${NC}"

# Python version
echo -n "Python version: "
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" == 3.11.* ]]; then
    echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"
    ((checks_passed++))
else
    echo -e "${YELLOW}⚠ $PYTHON_VERSION (3.11.x recommended)${NC}"
    ((checks_warning++))
fi

# CUDA version
echo -n "CUDA version: "
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')
    if [[ "$CUDA_VERSION" == "13.0" ]]; then
        echo -e "${GREEN}✓ $CUDA_VERSION${NC}"
        ((checks_passed++))
    else
        echo -e "${YELLOW}⚠ $CUDA_VERSION (13.0 recommended)${NC}"
        ((checks_warning++))
    fi
else
    echo -e "${RED}✗ NVCC not found${NC}"
    ((checks_failed++))
fi

# GPU detection
echo ""
echo -e "${BLUE}=== GPU Information ===${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -n1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)

    echo "GPU: $GPU_NAME"
    echo "VRAM: $GPU_MEMORY"
    echo "Driver: $DRIVER_VERSION"

    if [[ "$GPU_NAME" == *"5090"* ]]; then
        echo -e "${GREEN}✓ RTX 5090 detected${NC}"
        ((checks_passed++))

        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
        if [[ "$COMPUTE_CAP" == "12.0" ]]; then
            echo -e "${GREEN}✓ Compute Capability: sm_120${NC}"
            ((checks_passed++))
        else
            echo -e "${RED}✗ Wrong compute capability: $COMPUTE_CAP${NC}"
            ((checks_failed++))
        fi
    else
        echo -e "${YELLOW}⚠ Not RTX 5090: $GPU_NAME${NC}"
        ((checks_warning++))
    fi
else
    echo -e "${RED}✗ nvidia-smi not found${NC}"
    ((checks_failed++))
fi

# PyTorch verification
echo ""
echo -e "${BLUE}=== PyTorch Verification ===${NC}"

python << 'EOFPY'
import sys
import torch

checks_passed = 0
checks_failed = 0

GREEN = '\033[0;32m'
RED = '\033[0;31m'
NC = '\033[0m'

# PyTorch version
print(f"PyTorch version: {torch.__version__}")

# CUDA availability
if torch.cuda.is_available():
    print(f"{GREEN}✓ CUDA available{NC}")
    checks_passed += 1

    # CUDA version
    print(f"CUDA version: {torch.version.cuda}")

    # Architecture list
    arch_list = torch.cuda.get_arch_list()
    print(f"Architecture list: {arch_list}")

    if 'sm_120' in str(arch_list):
        print(f"{GREEN}✓ Native sm_120 support confirmed{NC}")
        checks_passed += 1
    else:
        print(f"{RED}✗ sm_120 not in architecture list{NC}")
        checks_failed += 1

    # GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {cap[0]}.{cap[1]}")

    if cap == (12, 0):
        print(f"{GREEN}✓ sm_120 GPU detected{NC}")
        checks_passed += 1
    else:
        print(f"{RED}✗ Not sm_120 GPU: sm_{cap[0]}{cap[1]}{NC}")
        checks_failed += 1

    # BF16 test
    try:
        a = torch.randn(1000, 1000, dtype=torch.bfloat16, device='cuda')
        b = torch.randn(1000, 1000, dtype=torch.bfloat16, device='cuda')
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print(f"{GREEN}✓ BF16 computation working{NC}")
        checks_passed += 1
    except Exception as e:
        print(f"{RED}✗ BF16 computation failed: {e}{NC}")
        checks_failed += 1

else:
    print(f"{RED}✗ CUDA not available in PyTorch{NC}")
    checks_failed += 1

sys.exit(checks_failed)
EOFPY

if [ $? -eq 0 ]; then
    ((checks_passed+=4))
else
    ((checks_failed+=4))
fi

# xformers verification
echo ""
echo -e "${BLUE}=== xformers Verification ===${NC}"
if python -c "import xformers; print(f'✓ xformers {xformers.__version__}')" 2>/dev/null; then
    echo -e "${GREEN}✓ xformers installed${NC}"
    ((checks_passed++))
else
    echo -e "${RED}✗ xformers not installed${NC}"
    ((checks_failed++))
fi

# Custom kernels verification
echo ""
echo -e "${BLUE}=== Custom Blackwell Kernels ===${NC}"
if python -c "import blackwell_flux_kernels; print('✓ Blackwell kernels loaded')" 2>/dev/null; then
    echo -e "${GREEN}✓ Custom kernels available${NC}"
    ((checks_passed++))
else
    echo -e "${YELLOW}⚠ Custom kernels not available${NC}"
    ((checks_warning++))
fi

# sd-scripts verification
echo ""
echo -e "${BLUE}=== sd-scripts Verification ===${NC}"
SD_SCRIPTS_DIR="$PROJECT_ROOT/sd-scripts"
if [ -d "$SD_SCRIPTS_DIR" ]; then
    echo -e "${GREEN}✓ sd-scripts directory exists${NC}"
    ((checks_passed++))

    if [ -f "$SD_SCRIPTS_DIR/flux_train_network.py" ]; then
        echo -e "${GREEN}✓ flux_train_network.py found${NC}"
        ((checks_passed++))
    else
        echo -e "${RED}✗ flux_train_network.py not found${NC}"
        ((checks_failed++))
    fi

    # Check dependencies
    python -c "import accelerate, transformers, diffusers, prodigyopt" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training dependencies installed${NC}"
        ((checks_passed++))
    else
        echo -e "${RED}✗ Some training dependencies missing${NC}"
        ((checks_failed++))
    fi
else
    echo -e "${RED}✗ sd-scripts not found${NC}"
    ((checks_failed++))
fi

# Performance benchmark
echo ""
echo -e "${BLUE}=== Performance Benchmark ===${NC}"
echo "Running BF16 matrix multiplication benchmark..."

python << 'EOFPY'
import torch
import time

if not torch.cuda.is_available():
    print("⚠ CUDA not available, skipping benchmark")
    exit(0)

# Benchmark parameters
size = 8192
iterations = 100

print(f"Matrix size: {size}x{size}")
print(f"Iterations: {iterations}")

# Create test data
a = torch.randn(size, size, dtype=torch.bfloat16, device='cuda')
b = torch.randn(size, size, dtype=torch.bfloat16, device='cuda')

# Warmup
for _ in range(10):
    c = torch.matmul(a, b)
torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(iterations):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
end = time.perf_counter()

# Calculate TFLOPS
flops = 2 * size ** 3 * iterations
duration = end - start
tflops = flops / duration / 1e12

print(f"\nPerformance: {tflops:.1f} TFLOPS")

# RTX 5090 theoretical: ~1320 TFLOPS for BF16
# Achievable: ~900-1000 TFLOPS (70-75%)
expected_min = 500  # Minimum acceptable

GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'

if tflops >= expected_min:
    print(f"{GREEN}✓ Performance acceptable ({tflops:.1f} TFLOPS){NC}")
    exit(0)
else:
    print(f"{YELLOW}⚠ Performance below expected ({tflops:.1f} TFLOPS){NC}")
    exit(1)
EOFPY

if [ $? -eq 0 ]; then
    ((checks_passed++))
else
    ((checks_warning++))
fi

# Summary
echo ""
echo "============================================================"
echo "VERIFICATION SUMMARY"
echo "============================================================"
echo -e "${GREEN}Passed: $checks_passed${NC}"
echo -e "${YELLOW}Warnings: $checks_warning${NC}"
echo -e "${RED}Failed: $checks_failed${NC}"

if [ $checks_failed -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ ALL CRITICAL CHECKS PASSED!${NC}"
    if [ $checks_warning -gt 0 ]; then
        echo -e "${YELLOW}Some non-critical warnings present${NC}"
    fi
    echo ""
    echo "Your system is ready for RTX 5090 FLUX training!"
    echo ""
    echo "Next steps:"
    echo "  1. Prepare your dataset in ./dataset/"
    echo "  2. Download FLUX models to ./models/"
    echo "  3. Edit config/rtx5090_native.toml"
    echo "  4. Run: ./scripts/08_start_training.sh"
    exit 0
else
    echo ""
    echo -e "${RED}❌ SOME CHECKS FAILED${NC}"
    echo ""
    echo "Please fix the failed checks before training."
    echo "Re-run the setup scripts as needed."
    exit 1
fi
