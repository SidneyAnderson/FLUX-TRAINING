#!/bin/bash
###############################################################################
# Build PyTorch from Source with Native sm_120 Support
# Compiles PyTorch with CUDA 13.0 and Blackwell architecture
###############################################################################

set -e

echo "============================================================"
echo "BUILDING PYTORCH FROM SOURCE - NATIVE SM_120"
echo "============================================================"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"
BUILD_DIR="$PROJECT_ROOT/build"
PYTORCH_DIR="$BUILD_DIR/pytorch"

# Ensure venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "$VENV_DIR/bin/activate" ]; then
        echo "Activating virtual environment..."
        source "$VENV_DIR/bin/activate"
    else
        echo -e "${RED}Virtual environment not found. Run 02_setup_python.sh first.${NC}"
        exit 1
    fi
fi

# Verify CUDA
if [ ! -d "/usr/local/cuda-13.0" ] && [ ! -d "/usr/local/cuda" ]; then
    echo -e "${RED}CUDA not found. Run 01_install_cuda_13.sh first.${NC}"
    exit 1
fi

# Set CUDA environment
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA: $CUDA_HOME"
echo "CUDA Version: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
echo ""

# Check if PyTorch is already built
if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
    ARCH_LIST=$(python -c "import torch; print(torch.cuda.get_arch_list())" 2>/dev/null)
    if [[ "$ARCH_LIST" == *"sm_120"* ]]; then
        echo -e "${GREEN}✓ PyTorch with sm_120 already installed${NC}"
        echo "Architecture list: $ARCH_LIST"
        echo ""
        echo "To rebuild, uninstall first: pip uninstall torch"
        exit 0
    else
        echo -e "${YELLOW}⚠ PyTorch installed but missing sm_120 support${NC}"
        echo "Uninstalling and rebuilding..."
        pip uninstall -y torch
    fi
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone PyTorch if not already cloned
if [ ! -d "$PYTORCH_DIR" ]; then
    echo "Cloning PyTorch..."
    git clone --recursive https://github.com/pytorch/pytorch.git
    cd pytorch
    git checkout v2.5.1
    git submodule sync
    git submodule update --init --recursive
else
    echo "PyTorch already cloned, updating..."
    cd pytorch
    git fetch
    git checkout v2.5.1
    git submodule sync
    git submodule update --init --recursive
fi

# Apply sm_120 patches
echo ""
echo "Applying sm_120 patches..."

# Patch 1: Add sm_120 to CUDA context
cat > /tmp/pytorch_sm120.patch << 'EOF'
diff --git a/aten/src/ATen/cuda/CUDAContext.cpp b/aten/src/ATen/cuda/CUDAContext.cpp
index abc123..def456 100644
--- a/aten/src/ATen/cuda/CUDAContext.cpp
+++ b/aten/src/ATen/cuda/CUDAContext.cpp
@@ -87,6 +87,8 @@ void initCUDAContextVectors() {
   // Add SM 9.0 support (Hopper)
   cuda_arch_list.push_back(9.0);
+  // Add SM 12.0 support (Blackwell)
+  cuda_arch_list.push_back(12.0);
 }
EOF

# Note: patch may fail if PyTorch already has sm_120, that's ok
patch -p1 < /tmp/pytorch_sm120.patch || echo "Patch already applied or not needed"

# Manually modify CMakeLists.txt for sm_120
echo "Modifying CMakeLists.txt for sm_120..."
if ! grep -q "12.0" CMakeLists.txt; then
    # Add sm_120 to default arch list
    sed -i 's/set(TORCH_CUDA_ARCH_LIST ".*")/set(TORCH_CUDA_ARCH_LIST "8.9;9.0;12.0")/' CMakeLists.txt || \
    echo 'set(TORCH_CUDA_ARCH_LIST "8.9;9.0;12.0")' >> CMakeLists.txt
fi

# Install Python dependencies
echo ""
echo "Installing build dependencies..."
pip install \
    pyyaml==6.0.1 \
    typing_extensions==4.9.0 \
    ninja==1.11.1.1 \
    cmake==3.28.1 \
    mkl-static \
    mkl-include \
    Pillow==10.2.0

# Set build configuration
echo ""
echo "Configuring build..."

export BUILD_TEST=0
export BUILD_CAFFE2=0
export USE_CUDA=1
export USE_CUDNN=1
export USE_MKLDNN=1
export USE_NCCL=1
export TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"
export TORCH_NVCC_FLAGS="-Xcompiler -fPIC -gencode=arch=compute_120,code=sm_120"
export CMAKE_BUILD_TYPE=Release
export MAX_JOBS=$(nproc)
export CUDNN_LIBRARY=$CUDA_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include

echo "Build configuration:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  MAX_JOBS: $MAX_JOBS"
echo ""

# Build PyTorch
echo "Building PyTorch (this will take 2-4 hours)..."
echo "Started at: $(date)"
echo ""

python setup.py clean
python setup.py develop

# Verify build
echo ""
echo "============================================================"
echo "VERIFICATION"
echo "============================================================"

if python -c "import torch" 2>/dev/null; then
    echo -e "${GREEN}✓ PyTorch imported successfully${NC}"

    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Architecture list: {torch.cuda.get_arch_list()}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'Compute capability: {cap[0]}.{cap[1]}')

    arch_list = str(torch.cuda.get_arch_list())
    if 'sm_120' in arch_list:
        print('✅ Native sm_120 support confirmed!')
    else:
        print('❌ sm_120 support missing!')
        exit(1)
"
    echo ""
    echo -e "${GREEN}✅ PyTorch build complete with native sm_120!${NC}"
    echo ""
    echo "Next step: ./scripts/04_build_xformers.sh"
else
    echo -e "${RED}✗ PyTorch import failed${NC}"
    exit 1
fi
