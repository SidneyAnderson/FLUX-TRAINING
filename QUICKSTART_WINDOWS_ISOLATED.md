# RTX 5090 FLUX Training - Windows Quick Start Guide
## 🚀 Isolated Environment - Zero System Impact

### ⚡ 5-Minute Overview

**What This Does:**
- Installs EVERYTHING in `D:\Flux_Trainer` (nothing system-wide)
- Zero impact on your Windows system
- Completely portable (can move/backup/delete easily)
- Professional isolated environment

### Prerequisites
- ✅ RTX 5090 GPU with driver 581.57+
- ✅ Windows 10/11 x64
- ✅ 64GB RAM (recommended)
- ✅ 150GB free space on D: drive
- ✅ Internet connection

---

## 🎯 FASTEST PATH TO TRAINING

### Option 1: Fully Automated (4-6 hours, unattended)

```powershell
# 1. Open PowerShell as Administrator

# 2. Create directory and download setup
New-Item -Path "D:\Flux_Trainer" -ItemType Directory -Force
cd D:\Flux_Trainer

# 3. Download the AI Agent setup prompt
Invoke-WebRequest -Uri "[your-repo]/AI_AGENT_RTX5090_SETUP_ISOLATED.md" -OutFile "setup_prompt.txt"

# 4. Copy contents to your AI assistant (Claude, GPT-4, etc.)
# Let it run for 4-6 hours - it will do EVERYTHING
```

### Option 2: Quick Manual Setup (if you have files)

```powershell
# 1. Extract pre-built environment (if available)
# Download Flux_Trainer_Prebuilt.zip from releases
Expand-Archive -Path "Flux_Trainer_Prebuilt.zip" -DestinationPath "D:\" -Force

# 2. Verify installation
D:\Flux_Trainer\FluxTrainer.bat
# Select option [V] to verify

# 3. Download models (one-time, ~33GB)
cd D:\Flux_Trainer\models
# Download these files from HuggingFace:
# - flux1-dev.safetensors (23GB)
# - ae.safetensors (335MB)
# - clip_l.safetensors (246MB)
# - t5xxl_fp16.safetensors (9.5GB)
```

---

## 📸 PREPARE YOUR DATASET (10 minutes)

### 1. Create Dataset Structure
```powershell
# Your person/style name (no spaces)
$name = "johnsmith"
$trigger = "johnsmith"  # Your trigger word

# Create folders
mkdir D:\Flux_Trainer\dataset\$name\70_$trigger
```

### 2. Add Your Images
- Copy 15-25 high-quality photos to: `D:\Flux_Trainer\dataset\johnsmith\70_johnsmith\`
- Images should be clear, well-lit, consistent hairstyle
- Mix of angles/expressions

### 3. Create Captions (Automatic)
```powershell
cd D:\Flux_Trainer\dataset\johnsmith\70_johnsmith

# Simple captions for faces (BEST for face training)
Get-ChildItem *.jpg | ForEach-Object {
    Set-Content ($_.BaseName + '.txt') "johnsmith"
}
```

---

## 🎮 START TRAINING (2 clicks!)

### Method 1: Using Launcher (Recommended)
```powershell
# Double-click this file:
D:\Flux_Trainer\FluxTrainer.bat

# Then:
# 1. Select [1] for Face Training
# 2. Enter dataset name: johnsmith
# 3. Enter trigger word: johnsmith
# 4. Press Enter to start
```

### Method 2: Direct Command
```powershell
cd D:\Flux_Trainer\sd-scripts-cuda13
.\venv\Scripts\activate

python flux_train_network.py `
    --config_file config_face.toml `
    --train_data_dir "D:\Flux_Trainer\dataset\johnsmith" `
    --output_name "flux_lora_johnsmith" `
    --highvram
```

---

## ⏱️ TIME ESTIMATES

| Phase | Time | Status |
|-------|------|--------|
| **Automated Setup** | 4-6 hours | One-time |
| **Model Download** | 30-60 min | One-time (33GB) |
| **Dataset Prep** | 10-15 min | Per project |
| **Face Training** | 90 min | 1500 steps |
| **Action Training** | 60 min | 1000 steps |
| **Style Training** | 45 min | 800 steps |

---

## 📊 MONITOR PROGRESS

### Watch Training Live
```powershell
# Training shows progress in console:
steps:  100/1500 loss: 0.342 lr: 1.0e-03 time: 0:05:23
steps:  200/1500 loss: 0.318 lr: 1.0e-03 time: 0:10:46
steps:  300/1500 loss: 0.312 lr: 1.0e-03 time: 0:16:09
```

### Sample Images
- Check `D:\Flux_Trainer\samples\` every 100 steps
- Images show progression of learning

### Loss Values (Normal)
- Starting: 0.35-0.40
- Step 500: 0.31-0.33 (plateau - NORMAL!)
- Step 1000: 0.30-0.32
- Step 1500: 0.30-0.31

**Note:** Loss plateau at 0.31 is NORMAL for faces - judge by sample quality, not loss!

---

## ✅ YOUR TRAINED LORA

### Output Location
```
D:\Flux_Trainer\output\
└── flux_lora_johnsmith.safetensors  (4-8MB file)
```

### Using Your LoRA

**In ComfyUI:**
1. Copy `.safetensors` to `ComfyUI\models\loras\`
2. Use trigger word: "johnsmith"
3. LoRA weight: 0.8-1.0

**In Automatic1111:**
1. Copy to `models\Lora\`
2. Prompt: `<lora:flux_lora_johnsmith:1> johnsmith, portrait`

**Test Prompts:**
- `johnsmith`
- `johnsmith, portrait photography`
- `johnsmith wearing a suit`
- `johnsmith, oil painting style`

---

## 🔧 QUICK TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| **"Python not found"** | Run from `FluxTrainer.bat` (sets paths) |
| **Out of memory** | Reduce network_dim to 64 in config |
| **Training slow** | Verify with option [V] in launcher |
| **Poor quality** | Use more images (20-30), train longer |
| **Installation failed** | Ensure D: has 150GB free |
| **Models missing** | Download from HuggingFace to `D:\Flux_Trainer\models` |

---

## 🎯 QUICK CONFIG TWEAKS

### For Better Quality
Edit `D:\Flux_Trainer\sd-scripts-cuda13\config_face.toml`:
```toml
network_dim = 128        # Increase to 192 for more detail
max_train_steps = 2000   # Train longer
sample_every_n_steps = 50  # More frequent samples
```

### For Faster Training
```toml
network_dim = 64         # Reduce for speed
max_train_steps = 1000   # Fewer steps
mixed_precision = "bf16"  # Already optimal
```

---

## 📝 COMPLETE ISOLATED ENVIRONMENT

### What Makes This Special:
- **No Python in Program Files** - Uses portable Python
- **No system PATH changes** - Everything session-local
- **No registry modifications** - Zero Windows registry touches
- **Portable** - Copy folder to any PC
- **Multiple versions** - Run different setups simultaneously
- **Clean uninstall** - Just delete D:\Flux_Trainer

### Directory Overview:
```
D:\Flux_Trainer\
├── FluxTrainer.bat      # ← START HERE (double-click)
├── python\              # Isolated Python 3.11.9
├── cuda_toolkit\        # Local CUDA 13.0
├── models\              # Flux models (download here)
├── dataset\             # Your training images
├── output\              # Your trained LoRAs
└── sd-scripts-cuda13\   # Training scripts
```

---

## 💡 PRO TIPS

### Dataset Quality > Quantity
- 20 great photos > 100 poor photos
- Consistent lighting helps
- Same hairstyle critical for faces
- Clear, unobstructed faces

### Captions Matter
- Faces: Just trigger word
- Actions: `trigger action_verb, details`
- Styles: `trigger, attribute1, attribute2`

### Sample Checking
- Don't judge by loss values
- Check samples every 100-200 steps
- Stop when quality peaks (avoid overtraining)

---

## 🚀 NEXT STEPS

1. **Test Your LoRA** - Try various prompts
2. **Train Another** - Different person/style
3. **Experiment** - Adjust network_dim, steps
4. **Backup** - Copy D:\Flux_Trainer to external drive
5. **Share** - LoRA file is portable

---

## 🆘 GET HELP

### Quick Checks:
```powershell
# 1. Verify installation
D:\Flux_Trainer\FluxTrainer.bat → [V] Verify

# 2. Check GPU
nvidia-smi

# 3. Test performance
D:\Flux_Trainer\FluxTrainer.bat → [9] Environment Info

# 4. View logs
notepad D:\Flux_Trainer\logs\training.log
```

### If All Else Fails:
1. Delete D:\Flux_Trainer
2. Re-run automated setup
3. Everything rebuilds from scratch

---

**Remember:** Everything is in D:\Flux_Trainer - nothing touches your system!

**Happy Training!** 🎨 Your isolated, professional Flux training environment awaits!
