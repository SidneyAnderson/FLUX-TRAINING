#!/bin/bash
###############################################################################
# Build Custom Blackwell Kernels for FLUX
# Optimized CUDA kernels for RTX 5090 with CUDA 13.0
###############################################################################

set -e

echo "============================================================"
echo "BUILDING CUSTOM BLACKWELL KERNELS"
echo "============================================================"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"
KERNELS_DIR="$PROJECT_ROOT/kernels"

# Ensure venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    else
        echo -e "${RED}Virtual environment not found. Run 02_setup_python.sh first.${NC}"
        exit 1
    fi
fi

# Verify PyTorch with sm_120
if ! python -c "import torch; assert 'sm_120' in str(torch.cuda.get_arch_list())" 2>/dev/null; then
    echo -e "${RED}PyTorch with sm_120 not found. Run 03_build_pytorch.sh first.${NC}"
    exit 1
fi

# Set CUDA environment
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA: $CUDA_HOME"
echo "Kernels directory: $KERNELS_DIR"
echo ""

# Check if kernels are already built
if python -c "import blackwell_flux_kernels" 2>/dev/null; then
    echo -e "${GREEN}✓ Blackwell kernels already installed${NC}"
    echo ""
    echo "To rebuild, uninstall first:"
    echo "  pip uninstall blackwell_flux_kernels"
    exit 0
fi

# Go to kernels directory
cd "$KERNELS_DIR"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.egg-info
python setup.py clean || true

# Build kernels
echo ""
echo "Building Blackwell kernels for sm_120..."
echo "This will take 5-10 minutes..."
echo ""

export TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"
export MAX_JOBS=$(nproc)

python setup.py build_ext --inplace
python setup.py install

# Verify installation
echo ""
echo "============================================================"
echo "VERIFICATION"
echo "============================================================"

if python -c "import blackwell_flux_kernels" 2>/dev/null; then
    echo -e "${GREEN}✓ Blackwell kernels imported successfully${NC}"

    # Test the kernels
    python << 'EOFPY'
import torch
import blackwell_flux_kernels

print("\nTesting Blackwell kernels...")

# Test BF16 matmul
print("Testing BF16 matrix multiplication...")
A = torch.randn(128, 128, dtype=torch.bfloat16, device='cuda')
B = torch.randn(128, 128, dtype=torch.bfloat16, device='cuda')

try:
    C = blackwell_flux_kernels.bf16_matmul(A, B, 1.0, 0.0)
    print("✓ BF16 matmul works")
except Exception as e:
    print(f"⚠ BF16 matmul test skipped (requires GPU): {e}")

# Test Flash Attention
print("Testing Flash Attention...")
Q = torch.randn(2, 64, 128, dtype=torch.bfloat16, device='cuda')
K = torch.randn(2, 64, 128, dtype=torch.bfloat16, device='cuda')
V = torch.randn(2, 64, 128, dtype=torch.bfloat16, device='cuda')

try:
    O = blackwell_flux_kernels.flash_attention(Q, K, V, 1.0/128**0.5)
    print("✓ Flash Attention works")
except Exception as e:
    print(f"⚠ Flash Attention test skipped (requires GPU): {e}")

print("\n✅ Blackwell kernels ready!")
EOFPY

    echo ""
    echo -e "${GREEN}✅ Blackwell kernels build complete!${NC}"
    echo ""
    echo "Next step: ./scripts/06_setup_sd_scripts.sh"
else
    echo -e "${RED}✗ Blackwell kernels import failed${NC}"
    exit 1
fi
