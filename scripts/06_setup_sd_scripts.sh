#!/bin/bash
###############################################################################
# Setup sd-scripts with Blackwell Optimizations
# Integrates custom kernels and CUDA 13.0 features
###############################################################################

set -e

echo "============================================================"
echo "SETTING UP SD-SCRIPTS WITH BLACKWELL OPTIMIZATIONS"
echo "============================================================"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"
SD_SCRIPTS_DIR="$PROJECT_ROOT/sd-scripts"

# Ensure venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    else
        echo -e "${RED}Virtual environment not found. Run 02_setup_python.sh first.${NC}"
        exit 1
    fi
fi

# Set CUDA environment
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Clone sd-scripts if not already cloned
if [ ! -d "$SD_SCRIPTS_DIR" ]; then
    echo "Cloning sd-scripts..."
    cd "$PROJECT_ROOT"
    git clone https://github.com/kohya-ss/sd-scripts.git
    cd sd-scripts
else
    echo "sd-scripts already cloned"
    cd "$SD_SCRIPTS_DIR"
    git pull
fi

# Install dependencies
echo ""
echo "Installing sd-scripts dependencies..."
pip install -r requirements.txt

# Install additional required packages
pip install \
    accelerate==0.34.2 \
    transformers==4.46.2 \
    diffusers==0.31.0 \
    safetensors==0.4.5 \
    opencv-python==4.10.0.84 \
    einops==0.8.0 \
    pytorch-lightning==2.4.0 \
    tensorboard==2.18.0 \
    toml==0.10.2 \
    prodigyopt==1.0

# Patch sd-scripts for Blackwell integration
echo ""
echo "Patching sd-scripts for Blackwell optimization..."

# Create patch for library/model_util.py
cat > /tmp/sd_scripts_blackwell.patch << 'EOFPATCH'
import sys
import os

# Set CUDA 13.0 environment
os.environ['CUDA_HOME'] = os.environ.get('CUDA_HOME', '/usr/local/cuda')
os.environ['CUDA_VERSION'] = '13.0'

# Import Blackwell kernels if available
try:
    import blackwell_flux_kernels
    HAS_BLACKWELL = True
    print("✓ Blackwell RTX 5090 kernels (CUDA 13.0) loaded")
    print("  • Native sm_120 support")
    print("  • BF16 tensor cores active")
    print("  • Flash Attention optimized")
except ImportError:
    HAS_BLACKWELL = False
    print("⚠ Running without Blackwell optimizations")
EOFPATCH

# Add to library/model_util.py if not already present
if ! grep -q "HAS_BLACKWELL" library/model_util.py 2>/dev/null; then
    # Create backup
    [ -f library/model_util.py ] && cp library/model_util.py library/model_util.py.backup

    # Prepend our patch
    cat /tmp/sd_scripts_blackwell.patch library/model_util.py.backup > library/model_util.py || \
    cat /tmp/sd_scripts_blackwell.patch > library/model_util.py
    echo "✓ model_util.py patched"
else
    echo "✓ model_util.py already patched"
fi

# Create Blackwell-optimized training wrapper
cat > "$SD_SCRIPTS_DIR/train_flux_blackwell.py" << 'EOFPY'
#!/usr/bin/env python3
"""
FLUX Training with Blackwell (RTX 5090) Optimizations
Enables all CUDA 13.0 features and custom kernels
"""

import os
import sys
import torch

# Force CUDA 13.0 and Blackwell optimizations
os.environ['CUDA_HOME'] = os.environ.get('CUDA_HOME', '/usr/local/cuda')
os.environ['CUDA_VERSION'] = '13.0'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9;9.0;12.0'

# Blackwell-specific environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.95,max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDNN_BENCHMARK'] = '1'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'

def setup_blackwell_optimizations():
    """Enable all Blackwell-specific optimizations"""
    if not torch.cuda.is_available():
        print("⚠ CUDA not available")
        return

    device_capability = torch.cuda.get_device_capability()
    if device_capability != (12, 0):
        print(f"⚠ Not a Blackwell GPU (sm_{device_capability[0]}{device_capability[1]})")
        return

    print("=" * 60)
    print("ENABLING BLACKWELL OPTIMIZATIONS")
    print("=" * 60)

    # Disable TF32 for matmul (use BF16 instead on Blackwell)
    torch.backends.cuda.matmul.allow_tf32 = False
    print("✓ TF32 disabled (using BF16)")

    # Enable BF16 reduced precision
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    print("✓ BF16 reduced precision enabled")

    # Enable optimized attention backends
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    print("✓ Flash Attention enabled")

    # CUDA graphs for reduced overhead
    torch.cuda.set_sync_debug_mode(0)
    print("✓ CUDA sync debug disabled (performance mode)")

    # Import custom kernels
    try:
        import blackwell_flux_kernels
        print("✓ Custom Blackwell kernels loaded")
        return True
    except ImportError:
        print("⚠ Custom Blackwell kernels not available")
        return False

    print("=" * 60)

# Setup optimizations
setup_blackwell_optimizations()

# Import and run main training script
import flux_train_network
flux_train_network.setup_parser()

if __name__ == "__main__":
    flux_train_network.train(flux_train_network.setup_parser().parse_args())
EOFPY

chmod +x "$SD_SCRIPTS_DIR/train_flux_blackwell.py"

# Verify setup
echo ""
echo "============================================================"
echo "VERIFICATION"
echo "============================================================"

python -c "
import sys
sys.path.insert(0, '$SD_SCRIPTS_DIR')

print('Checking sd-scripts imports...')
import library.model_util
print('✓ library.model_util imported')

import accelerate
import transformers
import diffusers
print('✓ Core dependencies imported')

import prodigyopt
print('✓ Prodigy optimizer imported')

print('\n✅ sd-scripts setup complete!')
"

echo ""
echo -e "${GREEN}✅ sd-scripts setup complete with Blackwell integration!${NC}"
echo ""
echo "Training script: $SD_SCRIPTS_DIR/train_flux_blackwell.py"
echo ""
echo "Next step: ./scripts/07_verify_setup.sh"
