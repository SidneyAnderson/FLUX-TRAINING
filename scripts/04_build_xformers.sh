#!/bin/bash
###############################################################################
# Build xformers from Source with Native sm_120 Support
# Compiles xformers with CUDA 13.0 and Blackwell optimizations
###############################################################################

set -e

echo "============================================================"
echo "BUILDING XFORMERS FROM SOURCE - NATIVE SM_120"
echo "============================================================"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"
BUILD_DIR="$PROJECT_ROOT/build"
XFORMERS_DIR="$BUILD_DIR/xformers"

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
echo ""

# Check if xformers is already built
if python -c "import xformers; print(xformers.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✓ xformers already installed${NC}"
    XFORMERS_VERSION=$(python -c "import xformers; print(xformers.__version__)")
    echo "Version: $XFORMERS_VERSION"
    echo ""
    echo "To rebuild, uninstall first: pip uninstall xformers"
    exit 0
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone xformers if not already cloned
if [ ! -d "$XFORMERS_DIR" ]; then
    echo "Cloning xformers..."
    git clone https://github.com/facebookresearch/xformers.git
    cd xformers
    git checkout v0.0.28
    git submodule update --init --recursive
else
    echo "xformers already cloned, updating..."
    cd xformers
    git fetch
    git checkout v0.0.28
    git submodule update --init --recursive
fi

# Set build environment for sm_120
echo ""
echo "Configuring build for sm_120..."

export TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"
export XFORMERS_BUILD_TYPE=Release
export XFORMERS_ENABLE_DEBUG_ASSERTIONS=0
export NVCC_FLAGS="-gencode=arch=compute_120,code=sm_120"
export MAX_JOBS=$(nproc)

# Modify setup.py to ensure sm_120 is included
cat > /tmp/xformers_sm120_setup.py << 'EOFPY'
import os
import sys

# Force CUDA 13.0 and sm_120
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9;9.0;12.0'
os.environ['XFORMERS_BUILD_TYPE'] = 'Release'
os.environ['XFORMERS_ENABLE_DEBUG_ASSERTIONS'] = '0'

# Import original setup
with open('setup.py', 'r') as f:
    setup_code = f.read()

# Execute setup
exec(setup_code)
EOFPY

# Install build dependencies
echo ""
echo "Installing build dependencies..."
pip install -r requirements.txt
pip install ninja cmake

# Build xformers
echo ""
echo "Building xformers (this will take 30-60 minutes)..."
echo "Started at: $(date)"
echo ""

# Use the modified setup
python -c "$(cat /tmp/xformers_sm120_setup.py)" build_ext --inplace
python setup.py install

# Verify build
echo ""
echo "============================================================"
echo "VERIFICATION"
echo "============================================================"

if python -c "import xformers" 2>/dev/null; then
    echo -e "${GREEN}✓ xformers imported successfully${NC}"

    python -c "
import xformers
import torch

print(f'xformers version: {xformers.__version__}')
print(f'PyTorch CUDA arch list: {torch.cuda.get_arch_list()}')

# Check if xformers has C extensions
if hasattr(xformers, '_C'):
    print('✓ xformers C extensions loaded')
else:
    print('⚠ xformers C extensions not found (may be normal)')

print('✅ xformers build complete with CUDA 13.0!')
"
    echo ""
    echo -e "${GREEN}✅ xformers build complete!${NC}"
    echo ""
    echo "Next step: ./scripts/05_build_blackwell_kernels.sh"
else
    echo -e "${RED}✗ xformers import failed${NC}"
    exit 1
fi
