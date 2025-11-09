# FLUX LORA TRAINING REFERENCE GUIDE
## Optimized Configurations for Face, Action, and Style LoRAs on RTX 5090
## Updated for D:\Flux_Trainer Training Environment

---

## 📚 TABLE OF CONTENTS

1. [Understanding LoRA Types](#understanding-lora-types)
2. [Face Identity LoRAs (99% Accuracy)](#face-identity-loras)
3. [Action LoRAs (Movement & Poses)](#action-loras)
4. [Style LoRAs (Clothing & Accessories)](#style-loras)
5. [Dataset Preparation Guidelines](#dataset-preparation-guidelines)
6. [Training Best Practices](#training-best-practices)
7. [Testing & Validation](#testing-validation)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## 🎯 UNDERSTANDING LORA TYPES

### Why Different Settings Matter

LoRA (Low-Rank Adaptation) training requires different approaches based on what you're teaching the model:

| LoRA Type | Learning Focus | Key Challenge | Critical Settings | Output Location |
|-----------|---------------|---------------|-------------------|-----------------|
| **Face Identity** | Exact facial features | Preventing averaging | High rank (64-128), minimal captions | D:\Flux_Trainer\output |
| **Action/Pose** | Movement patterns | Capturing dynamics | Medium rank (32-64), descriptive captions | D:\Flux_Trainer\output |
| **Style/Object** | Visual characteristics | Maintaining flexibility | Lower rank (16-32), detailed captions | D:\Flux_Trainer\output |

### Core Principle: Information Density

- **High Information Density** (Faces): Need more parameters (higher rank) to capture subtle details
- **Medium Density** (Actions): Balance between capturing movement and maintaining flexibility
- **Variable Density** (Styles): Can work with fewer parameters but benefit from richer descriptions

---

## 👤 FACE IDENTITY LORAS

### Goal: 99% Facial Accuracy

**Use Case:** Creating a LoRA that consistently generates the exact same person's face across different prompts.

### Optimal Configuration

Save as `D:\Flux_Trainer\sd-scripts-cuda13\config_face.toml`:

```toml
[Face_Identity_Configuration]

[model_arguments]
pretrained_model_name_or_path = "D:/Flux_Trainer/models/flux1-dev.safetensors"
ae = "D:/Flux_Trainer/models/ae.safetensors"
clip_l = "D:/Flux_Trainer/models/clip_l.safetensors"
t5xxl = "D:/Flux_Trainer/models/t5xxl_fp16.safetensors"

[dataset_arguments]
train_data_dir = "D:/Flux_Trainer/dataset"
output_dir = "D:/Flux_Trainer/output"
output_name = "flux_lora_face"
sample_prompts = "D:/Flux_Trainer/sample_prompts.txt"
sample_every_n_steps = 100

[network_arguments]
network_module = "networks.lora_flux"
network_dim = 128               # HIGH: Captures micro-expressions
network_alpha = 64              # Half of dim for stability
network_train_unet_only = true
network_dropout = 0.05          # Minimal: Preserve all features

[optimizer_arguments]
optimizer_type = "prodigyopt"   # CRITICAL: Self-adjusting LR
optimizer_args = [
    "decouple=True",
    "weight_decay=0.01",        # Low: Preserve detail
    "betas=[0.9,0.999]",
    "d_coef=2.0",              # Higher: Faster convergence
    "growth_rate=1.02"
]
learning_rate = 1.0
lr_scheduler = "constant"       # Prodigy handles scheduling

[training_arguments]
max_train_steps = 1500          # Longer: Faces need time
train_batch_size = 1
gradient_accumulation_steps = 4  # Effective batch = 4
mixed_precision = "bf16"
timestep_sampling = "shift"
discrete_flow_shift = 3.1582
save_every_n_steps = 100
```

### Dataset Requirements

**Location:** `D:\Flux_Trainer\dataset\[person_name]\70_[trigger]`
**Quantity:** 15-25 images (odd numbers work better)
**Resolution:** 1024x1024 EXACTLY
**Consistency Requirements:**
- Same hairstyle in all images
- Similar lighting conditions
- No extreme expressions
- No accessories that change shape perception

### Caption Strategy

```powershell
# Create captions in D:\Flux_Trainer\dataset
cd D:\Flux_Trainer\dataset\johnsmith\70_johnsmith
Get-ChildItem *.jpg | ForEach-Object {
    $txtFile = $_.FullName -replace '\.jpg$', '.txt'
    Set-Content $txtFile "johnsmith"
}
```

**WHY:** Forces the model to associate ALL visual information with the trigger, preventing feature dilution

### Training Milestones

| Steps | Expected Behavior | Action | Sample Location |
|-------|------------------|---------|-----------------|
| 0-200 | Generic face, loss ~0.4 | Continue | D:\Flux_Trainer\samples |
| 200-500 | Features emerging, loss plateaus ~0.31 | **NORMAL - KEEP GOING** | D:\Flux_Trainer\samples |
| 500-800 | Identity solidifying | Check samples | D:\Flux_Trainer\samples |
| 800-1200 | Fine details locking | Monitor overfitting | D:\Flux_Trainer\samples |
| 1200-1500 | Perfection or overfit | Test flexibility | D:\Flux_Trainer\output |

### Testing Protocol

```python
# Test script to run from D:\Flux_Trainer
import os
os.chdir("D:\\Flux_Trainer")

# Test increasing complexity
test_prompts = [
    "johnsmith",                          # Must work
    "johnsmith, portrait",                 # Should work
    "johnsmith wearing a suit",           # Tests flexibility
    "johnsmith as a superhero",           # Tests extreme transfer
    "johnsmith, oil painting style"       # Tests style transfer
]

# Success metrics:
# - First 3 prompts: 95%+ facial match
# - Last 2 prompts: 80%+ facial match with style adaptation
```

---

## 🏃 ACTION LORAS

### Goal: Consistent Action Reproduction

**Use Case:** Training specific movements, poses, or actions that can be applied to different subjects.

### Optimal Configuration

Save as `D:\Flux_Trainer\sd-scripts-cuda13\config_action.toml`:

```toml
[Action_Pose_Configuration]

[model_arguments]
pretrained_model_name_or_path = "D:/Flux_Trainer/models/flux1-dev.safetensors"
ae = "D:/Flux_Trainer/models/ae.safetensors"
clip_l = "D:/Flux_Trainer/models/clip_l.safetensors"
t5xxl = "D:/Flux_Trainer/models/t5xxl_fp16.safetensors"

[dataset_arguments]
train_data_dir = "D:/Flux_Trainer/dataset"
output_dir = "D:/Flux_Trainer/output"
output_name = "flux_lora_action"

[network_arguments]
network_module = "networks.lora_flux"
network_dim = 48                # MEDIUM: Balance detail and flexibility
network_alpha = 24               # Half of dim
network_train_unet_only = true
network_dropout = 0.1            # Higher: Prevent rigid memorization

[optimizer_arguments]
optimizer_type = "prodigyopt"
optimizer_args = [
    "decouple=True",
    "weight_decay=0.05",         # Higher: Encourage generalization
    "betas=[0.9,0.99]",
    "d_coef=1.5",               # Moderate convergence
    "growth_rate=1.01"
]
learning_rate = 0.8              # Slightly lower multiplier
lr_scheduler = "cosine"          # Helps with motion smoothness

[training_arguments]
max_train_steps = 1000           # Less than faces
train_batch_size = 2             # Can handle more
gradient_accumulation_steps = 2   # Effective batch = 4
mixed_precision = "bf16"
min_snr_gamma = 5.0              # Improves motion clarity
noise_offset = 0.1               # Helps with dynamic range
```

### Dataset Requirements

**Location:** `D:\Flux_Trainer\dataset\[action_name]\40_[trigger]`
**Quantity:** 20-40 images
**Resolution:** 1024x1024 or 768x1280 (portrait actions)
**Diversity Requirements:**
- Multiple angles of the same action
- Different subjects performing the action (if generalizing)
- Clear motion indication
- Consistent action phase captured

### Caption Strategy

```powershell
# Create descriptive captions in D:\Flux_Trainer\dataset
cd D:\Flux_Trainer\dataset\running\40_humanaction

# Example caption creation
@"
humanaction running, full body, side view
"@ | Out-File "image001.txt" -Encoding UTF8

@"
humanaction running, mid-stride, dynamic pose
"@ | Out-File "image002.txt" -Encoding UTF8
```

---

## 👔 STYLE LORAS

### Goal: Transferable Visual Styles

**Use Case:** Training specific clothing, accessories, art styles, or visual treatments.

### Optimal Configuration

Save as `D:\Flux_Trainer\sd-scripts-cuda13\config_style.toml`:

```toml
[Style_Object_Configuration]

[model_arguments]
pretrained_model_name_or_path = "D:/Flux_Trainer/models/flux1-dev.safetensors"
ae = "D:/Flux_Trainer/models/ae.safetensors"
clip_l = "D:/Flux_Trainer/models/clip_l.safetensors"
t5xxl = "D:/Flux_Trainer/models/t5xxl_fp16.safetensors"

[dataset_arguments]
train_data_dir = "D:/Flux_Trainer/dataset"
output_dir = "D:/Flux_Trainer/output"
output_name = "flux_lora_style"

[network_arguments]
network_module = "networks.lora_flux"
network_dim = 32                 # LOWER: Styles are simpler patterns
network_alpha = 16                # Half of dim
network_train_unet_only = true
network_dropout = 0.15            # Highest: Maximum flexibility

[optimizer_arguments]
optimizer_type = "adamw"          # Better for styles than prodigy
optimizer_args = [
    "weight_decay=0.1",
    "betas=[0.9,0.999]"
]
learning_rate = 0.0002            # Fixed rate works better
lr_scheduler = "cosine_with_restarts"
lr_scheduler_args = [
    "T_0=200",                   # Restart every 200 steps
    "T_mult=1.5",
    "eta_min=0.00002"
]

[training_arguments]
max_train_steps = 800            # Shortest training
train_batch_size = 4             # Can handle larger batches
gradient_accumulation_steps = 1
mixed_precision = "bf16"
```

---

## 📁 DATASET PREPARATION GUIDELINES

### Directory Structure

```
D:\Flux_Trainer\dataset\
├── face_dataset\
│   └── 70_johndoe\           # 70 = repeat count
│       ├── img001.jpg        # 1024x1024
│       ├── img001.txt        # "johndoe"
│       └── ...
├── action_dataset\
│   └── 40_humanaction\
│       ├── run001.jpg
│       ├── run001.txt        # "humanaction running, side view"
│       └── ...
└── style_dataset\
    └── 30_cyberstyle\
        ├── style001.jpg
        ├── style001.txt      # "cyberstyle, neon jacket, futuristic"
        └── ...
```

### Preprocessing Script

Save as `D:\Flux_Trainer\preprocess_dataset.py`:

```python
import os
from PIL import Image
import shutil

def prepare_dataset(input_dir, output_dir, trigger_word, lora_type="face"):
    """
    Prepare dataset for Flux LoRA training
    Args:
        input_dir: Source images directory
        output_dir: D:\\Flux_Trainer\\dataset\\[name]
        trigger_word: The trigger word for the LoRA
        lora_type: "face", "action", or "style"
    """
    
    # Determine repeat count based on type
    repeats = {
        "face": 70,
        "action": 40,
        "style": 30
    }
    
    repeat_count = repeats.get(lora_type, 50)
    dataset_dir = f"{output_dir}\\{repeat_count}_{trigger_word}"
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Process images
    for i, img_file in enumerate(os.listdir(input_dir)):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Load and resize image
            img_path = os.path.join(input_dir, img_file)
            img = Image.open(img_path)
            
            # Resize to 1024x1024
            img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
            
            # Save processed image
            output_name = f"img{i+1:03d}.jpg"
            img.save(os.path.join(dataset_dir, output_name), quality=95)
            
            # Create caption
            caption_file = os.path.join(dataset_dir, f"img{i+1:03d}.txt")
            
            if lora_type == "face":
                caption = trigger_word
            elif lora_type == "action":
                caption = f"{trigger_word} performing action, full body"
            else:  # style
                caption = f"{trigger_word}, detailed view"
            
            with open(caption_file, 'w') as f:
                f.write(caption)
    
    print(f"Dataset prepared at: {dataset_dir}")
    print(f"Images processed: {i+1}")
    print(f"Repeat count: {repeat_count}")

# Usage
if __name__ == "__main__":
    prepare_dataset(
        input_dir="D:\\raw_images\\person",
        output_dir="D:\\Flux_Trainer\\dataset\\johnsmith",
        trigger_word="johnsmith",
        lora_type="face"
    )
```

---

## 🔬 TRAINING BEST PRACTICES

### Memory Optimization Settings

Add to any config for RTX 5090 (32GB VRAM):

```toml
[memory_optimization]
gradient_checkpointing = true
xformers = true
cache_latents = true
cache_latents_to_disk = false  # Keep in VRAM for speed
persistent_data_loader_workers = true
max_data_loader_n_workers = 8
```

### Multi-GPU Training (If available)

```toml
[distributed_training]
accelerate_launch = true
num_processes = 2  # For 2 GPUs
mixed_precision = "bf16"
gradient_accumulation_steps = 2  # Per GPU
```

### Advanced Sampling Configuration

Create `D:\Flux_Trainer\sample_prompts.txt`:

```text
# Basic test
{trigger}

# Portrait test
{trigger}, portrait photography, studio lighting

# Style transfer test
{trigger}, oil painting style

# Action test (for action LoRAs)
{trigger} running, dynamic pose

# Complex scene
{trigger} in a cyberpunk city, neon lights
```

---

## 🧪 TESTING & VALIDATION

### Automated Testing Script

Save as `D:\Flux_Trainer\test_lora.py`:

```python
import os
import sys
import torch
from pathlib import Path

def test_lora_checkpoint(checkpoint_path, test_prompts, output_dir):
    """
    Test a LoRA checkpoint with various prompts
    """
    
    # Setup paths
    os.chdir("D:\\Flux_Trainer\\sd-scripts-cuda13")
    sys.path.append(".")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Test each prompt
    for i, prompt in enumerate(test_prompts):
        cmd = f"""
        python flux_minimal_inference.py \
            --ckpt D:/Flux_Trainer/models/flux1-dev.safetensors \
            --clip_l D:/Flux_Trainer/models/clip_l.safetensors \
            --t5xxl D:/Flux_Trainer/models/t5xxl_fp16.safetensors \
            --ae D:/Flux_Trainer/models/ae.safetensors \
            --lora {checkpoint_path} \
            --prompt "{prompt}" \
            --output {output_dir}/test_{i:03d}.png \
            --seed 42 \
            --steps 20 \
            --guidance 3.5
        """
        os.system(cmd)
    
    print(f"Tests complete. Results in: {output_dir}")

# Usage
if __name__ == "__main__":
    test_prompts = [
        "johndoe",
        "johndoe, portrait",
        "johndoe wearing a suit",
        "johndoe, anime style"
    ]
    
    test_lora_checkpoint(
        checkpoint_path="D:/Flux_Trainer/output/flux_lora_face-000500.safetensors",
        test_prompts=test_prompts,
        output_dir="D:/Flux_Trainer/output/tests"
    )
```

---

## 🔍 TROUBLESHOOTING GUIDE

### Common Issues and Solutions

| Issue | Symptoms | Solution | File Location |
|-------|----------|----------|---------------|
| **Models Not Found** | FileNotFoundError | Verify models in D:\Flux_Trainer\models | Check config paths |
| **Dataset Not Loading** | No images found | Check D:\Flux_Trainer\dataset structure | Verify folder naming |
| **Out of Memory** | CUDA OOM | Reduce batch_size or network_dim | Edit config.toml |
| **Loss Plateau** | Loss stuck at 0.31-0.35 | Normal for faces, check samples | D:\Flux_Trainer\samples |
| **No Output Saved** | Missing checkpoints | Check output_dir permissions | D:\Flux_Trainer\output |

### Quick Diagnostic Commands

```powershell
# Check GPU status
nvidia-smi

# Verify Python environment
cd D:\Flux_Trainer\sd-scripts-cuda13
.\venv\Scripts\activate
python -c "import torch; print(torch.cuda.is_available())"

# Check dataset structure
Get-ChildItem D:\Flux_Trainer\dataset -Recurse | Select-Object FullName

# Monitor training logs
Get-Content D:\Flux_Trainer\sd-scripts-cuda13\training.log -Tail 50 -Wait

# Check output files
Get-ChildItem D:\Flux_Trainer\output -Filter "*.safetensors" | Sort-Object LastWriteTime
```

---

## 📈 PERFORMANCE OPTIMIZATION

### RTX 5090 Specific Settings

```toml
[rtx5090_optimization]
# Leverage all 32GB VRAM
train_batch_size = 1  # For faces
gradient_accumulation_steps = 4
cache_latents = true
cache_text_encoder_outputs = true

# Use BF16 for Blackwell
mixed_precision = "bf16"
full_bf16 = true

# Custom kernel settings
use_custom_kernels = true
kernel_path = "D:/Flux_Trainer/cuda_kernels"
```

---

## 📋 QUICK REFERENCE CARD

### Command Cheatsheet

```powershell
# Activate environment
cd D:\Flux_Trainer\sd-scripts-cuda13
.\venv\Scripts\activate

# Start training
python flux_train_network.py --config_file config_face.toml --highvram

# Resume training
python flux_train_network.py --config_file config_face.toml --resume D:\Flux_Trainer\output\last.safetensors

# Test checkpoint
python flux_minimal_inference.py --lora D:\Flux_Trainer\output\flux_lora-001000.safetensors --prompt "test"

# Monitor GPU
watch -n 1 nvidia-smi

# Check samples
explorer D:\Flux_Trainer\samples
```

---

**This reference guide provides everything needed for successful Flux LoRA training on RTX 5090 with the training environment at D:\Flux_Trainer.**
