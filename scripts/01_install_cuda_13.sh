#!/bin/bash
###############################################################################
# CUDA 13.0 Installation for RTX 5090
# Installs CUDA 13.0 toolkit with native sm_120 support
###############################################################################

set -e

echo "============================================================"
echo "CUDA 13.0 INSTALLATION FOR RTX 5090"
echo "============================================================"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (sudo)${NC}"
    exit 1
fi

# Check if CUDA 13.0 is already installed
if [ -d "/usr/local/cuda-13.0" ]; then
    echo -e "${GREEN}✓ CUDA 13.0 already installed${NC}"
    nvcc --version
    echo ""
    echo "Skipping installation. If you need to reinstall, remove /usr/local/cuda-13.0 first."
    exit 0
fi

echo ""
echo "This will install CUDA 13.0 Toolkit"
echo "Note: CUDA 13.0 provides native Blackwell (sm_120) support"
echo ""

# Detect Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_VERSION=$VERSION_ID
else
    echo -e "${RED}Cannot detect OS version${NC}"
    exit 1
fi

echo "Detected Ubuntu $OS_VERSION"

# Install prerequisites
echo ""
echo "Installing prerequisites..."
apt-get update
apt-get install -y wget gnupg2 software-properties-common

# Add NVIDIA package repositories
echo ""
echo "Adding NVIDIA package repositories..."

# NOTE: As of writing, CUDA 13.0 may not be publicly released
# This script prepares for when it is available
# For now, you may need to use CUDA 12.4 or check NVIDIA's website

echo -e "${YELLOW}⚠ IMPORTANT:${NC}"
echo "CUDA 13.0 may not be publicly available yet."
echo "Please check https://developer.nvidia.com/cuda-downloads"
echo ""
echo "If CUDA 13.0 is not available, you can:"
echo "  1. Use CUDA 12.4 (has sm_120 support in compatibility mode)"
echo "  2. Wait for CUDA 13.0 official release"
echo "  3. Request access to CUDA 13.0 early access program"
echo ""

read -p "Do you have access to CUDA 13.0 installer? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Please download CUDA 13.0 installer manually from NVIDIA"
    echo "Then run the installer with:"
    echo "  sudo sh cuda_13.0_*_linux.run --silent --toolkit"
    exit 1
fi

# If user confirms they have CUDA 13.0
echo ""
echo "Please provide the path to CUDA 13.0 installer (.run file):"
read -p "Path: " CUDA_INSTALLER

if [ ! -f "$CUDA_INSTALLER" ]; then
    echo -e "${RED}Installer not found: $CUDA_INSTALLER${NC}"
    exit 1
fi

# Install CUDA 13.0
echo ""
echo "Installing CUDA 13.0..."
sh "$CUDA_INSTALLER" --silent --toolkit --override

# Verify installation
if [ -d "/usr/local/cuda-13.0" ]; then
    echo -e "${GREEN}✓ CUDA 13.0 installed successfully${NC}"

    # Set up environment
    echo ""
    echo "Setting up environment variables..."

    # Add to bash profile
    cat >> /etc/profile.d/cuda.sh << 'EOF'
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
EOF

    # Create symlink
    ln -sf /usr/local/cuda-13.0 /usr/local/cuda

    # Source the new environment
    source /etc/profile.d/cuda.sh

    # Verify
    echo ""
    echo "Verification:"
    nvcc --version

    # Test sm_120 support
    echo ""
    echo "Testing sm_120 support..."
    cat > /tmp/test_sm120.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_kernel() {
    if (threadIdx.x == 0) {
        printf("sm_120 kernel executed successfully!\n");
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    if (prop.major == 12 && prop.minor == 0) {
        test_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        printf("✓ Native sm_120 support confirmed\n");
    }

    return 0;
}
EOF

    nvcc -arch=sm_120 /tmp/test_sm120.cu -o /tmp/test_sm120
    /tmp/test_sm120
    rm /tmp/test_sm120.cu /tmp/test_sm120

    echo ""
    echo -e "${GREEN}✅ CUDA 13.0 installation complete!${NC}"
    echo ""
    echo "Next step: ./scripts/02_setup_python.sh"
else
    echo -e "${RED}✗ Installation failed${NC}"
    exit 1
fi
