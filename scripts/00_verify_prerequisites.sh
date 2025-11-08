#!/bin/bash
###############################################################################
# RTX 5090 Prerequisites Verification
# Checks system requirements before starting setup
###############################################################################

set -e

echo "============================================================"
echo "RTX 5090 FLUX TRAINING - PREREQUISITES VERIFICATION"
echo "============================================================"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

checks_passed=0
checks_failed=0

# Function to check command existence
check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found: $(command -v $1)"
        ((checks_passed++))
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        ((checks_failed++))
        return 1
    fi
}

# Function to check GPU
check_gpu() {
    echo ""
    echo "Checking GPU..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        if [[ "$GPU_NAME" == *"5090"* ]]; then
            echo -e "${GREEN}✓${NC} RTX 5090 detected: $GPU_NAME"
            ((checks_passed++))

            # Check compute capability
            COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
            if [[ "$COMPUTE_CAP" == "12.0" ]]; then
                echo -e "${GREEN}✓${NC} Compute Capability: $COMPUTE_CAP (sm_120)"
                ((checks_passed++))
            else
                echo -e "${RED}✗${NC} Wrong compute capability: $COMPUTE_CAP (expected 12.0)"
                ((checks_failed++))
            fi
        else
            echo -e "${RED}✗${NC} RTX 5090 not detected. Found: $GPU_NAME"
            ((checks_failed++))
        fi
    else
        echo -e "${RED}✗${NC} nvidia-smi not found - GPU drivers not installed"
        ((checks_failed++))
    fi
}

# Function to check disk space
check_disk_space() {
    echo ""
    echo "Checking disk space..."
    AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_GB" -ge 200 ]; then
        echo -e "${GREEN}✓${NC} Available space: ${AVAILABLE_GB}GB (>= 200GB required)"
        ((checks_passed++))
    else
        echo -e "${RED}✗${NC} Insufficient disk space: ${AVAILABLE_GB}GB (200GB required)"
        ((checks_failed++))
    fi
}

# Function to check RAM
check_ram() {
    echo ""
    echo "Checking RAM..."
    TOTAL_RAM_GB=$(free -g | grep Mem | awk '{print $2}')
    if [ "$TOTAL_RAM_GB" -ge 60 ]; then
        echo -e "${GREEN}✓${NC} Total RAM: ${TOTAL_RAM_GB}GB (>= 64GB recommended)"
        ((checks_passed++))
    else
        echo -e "${YELLOW}⚠${NC} RAM: ${TOTAL_RAM_GB}GB (64GB recommended, may work with less)"
        ((checks_passed++))
    fi
}

# Function to check OS
check_os() {
    echo ""
    echo "Checking OS..."
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "OS: $NAME $VERSION"
        if [[ "$ID" == "ubuntu" ]] && ([[ "$VERSION_ID" == "22.04" ]] || [[ "$VERSION_ID" == "24.04" ]]); then
            echo -e "${GREEN}✓${NC} Ubuntu 22.04/24.04 LTS detected"
            ((checks_passed++))
        else
            echo -e "${YELLOW}⚠${NC} Not Ubuntu 22.04/24.04 - may work but untested"
            ((checks_passed++))
        fi
    fi
}

echo ""
echo "=== System Information ==="
check_os
check_ram
check_disk_space

echo ""
echo "=== GPU Check ==="
check_gpu

echo ""
echo "=== Build Tools ==="
check_command "gcc"
check_command "g++"
check_command "cmake"
check_command "ninja" || echo -e "${YELLOW}⚠${NC} ninja will be installed during setup"
check_command "git"
check_command "wget"
check_command "curl"

echo ""
echo "=== Python ==="
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Python found: $PYTHON_VERSION"
    if [[ "$PYTHON_VERSION" == 3.11.* ]]; then
        echo -e "${GREEN}✓${NC} Python 3.11.x detected (compatible)"
        ((checks_passed++))
    else
        echo -e "${YELLOW}⚠${NC} Python $PYTHON_VERSION (3.11.9 will be set up in venv)"
        ((checks_passed++))
    fi
else
    echo -e "${RED}✗${NC} Python3 not found"
    ((checks_failed++))
fi

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo -e "Checks passed: ${GREEN}$checks_passed${NC}"
echo -e "Checks failed: ${RED}$checks_failed${NC}"

if [ $checks_failed -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All prerequisites met! Ready to proceed.${NC}"
    echo ""
    echo "Next step: ./scripts/01_install_cuda_13.sh"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Some prerequisites are missing. Please install required components.${NC}"
    echo ""
    echo "Required actions:"
    echo "  1. Install NVIDIA drivers (if GPU check failed)"
    echo "  2. Install build tools: sudo apt install build-essential cmake git wget curl"
    echo "  3. Ensure sufficient disk space (200GB)"
    exit 1
fi
