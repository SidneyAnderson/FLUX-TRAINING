#!/bin/bash
###############################################################################
# Python 3.11.9 Environment Setup
# Creates isolated Python environment with exact version
###############################################################################

set -e

echo "============================================================"
echo "PYTHON 3.11.9 ENVIRONMENT SETUP"
echo "============================================================"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"
PYTHON_VERSION="3.11.9"

echo "Project root: $PROJECT_ROOT"
echo "Virtual environment: $VENV_DIR"
echo ""

# Check if venv already exists
if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
    source "$VENV_DIR/bin/activate"
    CURRENT_VERSION=$(python --version | awk '{print $2}')
    echo "Current Python: $CURRENT_VERSION"

    if [[ "$CURRENT_VERSION" == "3.11"* ]]; then
        echo -e "${GREEN}✓ Python 3.11.x already set up${NC}"
        echo ""
        echo "If you want to recreate the environment, delete the venv directory:"
        echo "  rm -rf $VENV_DIR"
        exit 0
    fi
fi

# Install pyenv if not present
if ! command -v pyenv &> /dev/null; then
    echo "Installing pyenv..."
    curl https://pyenv.run | bash

    # Add to bash profile
    cat >> ~/.bashrc << 'EOF'

# Pyenv configuration
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
EOF

    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
fi

# Install Python build dependencies
echo ""
echo "Installing Python build dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev
fi

# Install Python 3.11.9
echo ""
echo "Installing Python $PYTHON_VERSION..."
pyenv install -s $PYTHON_VERSION

# Create virtual environment
echo ""
echo "Creating virtual environment..."
pyenv virtualenv $PYTHON_VERSION flux-training-env
pyenv local flux-training-env

# Alternative: use venv directly
mkdir -p "$VENV_DIR"
~/.pyenv/versions/$PYTHON_VERSION/bin/python -m venv "$VENV_DIR"

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install essential build tools
echo ""
echo "Installing build dependencies..."
pip install \
    setuptools==69.0.3 \
    wheel==0.42.0 \
    ninja==1.11.1.1 \
    cmake==3.28.1 \
    numpy==1.26.4

# Verify
echo ""
echo "============================================================"
echo "VERIFICATION"
echo "============================================================"
python --version
pip --version
echo ""
echo -e "${GREEN}✅ Python environment ready!${NC}"
echo ""
echo "To activate this environment in the future:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Next step: ./scripts/03_build_pytorch.sh"
