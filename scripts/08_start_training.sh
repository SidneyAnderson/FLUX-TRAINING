#!/bin/bash
###############################################################################
# Start FLUX Training with Blackwell Optimizations
# Launches training with all RTX 5090 + CUDA 13.0 features enabled
###############################################################################

set -e

echo "============================================================"
echo "STARTING RTX 5090 NATIVE FLUX TRAINING"
echo "============================================================"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"
SD_SCRIPTS_DIR="$PROJECT_ROOT/sd-scripts"
CONFIG_FILE="$PROJECT_ROOT/config/rtx5090_native.toml"

# Ensure venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "$VENV_DIR/bin/activate" ]; then
        echo "Activating virtual environment..."
        source "$VENV_DIR/bin/activate"
    else
        echo -e "${RED}Virtual environment not found. Run setup scripts first.${NC}"
        exit 1
    fi
fi

# Pre-flight checks
echo ""
echo -e "${BLUE}=== Pre-Flight Checks ===${NC}"

# Check config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Config file found${NC}"

# Check sd-scripts directory
if [ ! -d "$SD_SCRIPTS_DIR" ]; then
    echo -e "${RED}✗ sd-scripts not found. Run 06_setup_sd_scripts.sh first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ sd-scripts directory found${NC}"

# Check dataset directory
DATASET_DIR="$PROJECT_ROOT/dataset"
if [ ! -d "$DATASET_DIR" ]; then
    echo -e "${RED}✗ Dataset directory not found: $DATASET_DIR${NC}"
    echo ""
    echo "Please create dataset directory and add your training images:"
    echo "  mkdir -p $DATASET_DIR"
    echo "  # Add images and .txt captions"
    exit 1
fi

# Count images
IMAGE_COUNT=$(find "$DATASET_DIR" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo -e "${RED}✗ No images found in dataset directory${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dataset directory found ($IMAGE_COUNT images)${NC}"

# Check models directory
MODELS_DIR="$PROJECT_ROOT/models"
if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${RED}✗ Models directory not found: $MODELS_DIR${NC}"
    echo ""
    echo "Please create models directory and download FLUX models:"
    echo "  mkdir -p $MODELS_DIR"
    echo "  # Download flux1-dev.safetensors, clip_l.safetensors, etc."
    exit 1
fi

# Check for required models
REQUIRED_MODELS=("flux1-dev.safetensors" "clip_l.safetensors" "t5xxl_fp16.safetensors" "ae.safetensors")
MISSING_MODELS=()

for model in "${REQUIRED_MODELS[@]}"; do
    if [ ! -f "$MODELS_DIR/$model" ]; then
        MISSING_MODELS+=("$model")
    fi
done

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo -e "${RED}✗ Missing required models:${NC}"
    for model in "${MISSING_MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "Download from HuggingFace and place in $MODELS_DIR"
    exit 1
fi
echo -e "${GREEN}✓ All required models found${NC}"

# Verify PyTorch with sm_120
if ! python -c "import torch; assert 'sm_120' in str(torch.cuda.get_arch_list())" 2>/dev/null; then
    echo -e "${RED}✗ PyTorch with sm_120 not detected${NC}"
    echo "Run 07_verify_setup.sh to diagnose issues"
    exit 1
fi
echo -e "${GREEN}✓ PyTorch with sm_120 verified${NC}"

# Set CUDA environment
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Blackwell-specific environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.95,max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"

# Display GPU information
echo ""
echo -e "${BLUE}=== GPU Information ===${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader
else
    echo -e "${YELLOW}⚠ nvidia-smi not available${NC}"
fi

# Display training configuration
echo ""
echo -e "${BLUE}=== Training Configuration ===${NC}"
echo "Config file: $CONFIG_FILE"
echo "Dataset: $DATASET_DIR ($IMAGE_COUNT images)"
echo "Models: $MODELS_DIR"
echo "Output: $PROJECT_ROOT/output"
echo "Logs: $PROJECT_ROOT/logs"

# Ask for confirmation
echo ""
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Create output and logs directories
mkdir -p "$PROJECT_ROOT/output"
mkdir -p "$PROJECT_ROOT/logs"

# Change to sd-scripts directory
cd "$SD_SCRIPTS_DIR"

echo ""
echo "============================================================"
echo "LAUNCHING TRAINING"
echo "============================================================"
echo "Started at: $(date)"
echo ""

# Launch training with accelerate
if [ -f "train_flux_blackwell.py" ]; then
    # Use our custom Blackwell-optimized wrapper
    echo "Using Blackwell-optimized training script..."
    python train_flux_blackwell.py \
        --config_file "$CONFIG_FILE" \
        --sample_at_first \
        --highvram
else
    # Fallback to standard flux_train_network.py
    echo "Using standard training script..."
    accelerate launch \
        --num_cpu_threads_per_process 8 \
        flux_train_network.py \
        --config_file "$CONFIG_FILE" \
        --sample_at_first \
        --highvram
fi

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "TRAINING COMPLETED"
echo "============================================================"
echo "Finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo ""
    echo "Output models: $PROJECT_ROOT/output"
    echo "TensorBoard logs: $PROJECT_ROOT/logs"
    echo ""
    echo "To view training logs:"
    echo "  tensorboard --logdir=$PROJECT_ROOT/logs"
else
    echo -e "${RED}✗ Training failed with exit code $EXIT_CODE${NC}"
    echo "Check logs for details."
    exit $EXIT_CODE
fi
