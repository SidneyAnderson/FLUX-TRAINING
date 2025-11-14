# Flux LoRA Training on Windows Subsystem for Linux (WSL2)
## Run native Linux training while keeping assets on Windows 11

---

## Why switch to WSL?

- **Complete Linux toolchain** – no missing build dependencies or half-supported libraries on Windows.
- **Full GPU performance** – NVIDIA’s Windows driver exposes the RTX 5090 directly to WSL2.
- **Shared storage** – keep datasets, captions, and outputs on a Windows NTFS volume that is simultaneously available inside WSL.
- **Clean separation** – Windows stays untouched; all compilers, Python installs, and CUDA user-space libraries live in the Linux VM.

This guide explains how to install, configure, and operate WSL specifically for the Flux LoRA workflow maintained in this repository.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Windows | Windows 11 22H2 or newer, patched via Windows Update |
| Hardware | BIOS/UEFI virtualization (Intel VT-x/AMD-V) enabled, 64 GB RAM, 200 GB free SSD |
| GPU | NVIDIA RTX 5090 with driver **565.90+** (the first branch with CUDA 13/WDDM 3.2 + WSL support) |
| Admin rights | Needed to enable Windows features / install WSL |
| Internet | Downloads Ubuntu image, CUDA toolkits, models, Python packages |

> **Tip**: Update your NVIDIA driver from GeForce/Studio download or NVIDIA Enterprise portal before proceeding. The Linux runtime inside WSL inherits the kernel driver from Windows, so upgrading it first prevents mismatched CUDA components later.

---

## Step 1 – Enable required Windows features

Run an elevated PowerShell session (`Win+X → Windows Terminal (Admin)`) and execute:

```powershell
DISM /Online /Enable-Feature /FeatureName:Microsoft-Windows-Subsystem-Linux /All /NoRestart
DISM /Online /Enable-Feature /FeatureName:VirtualMachinePlatform /All /NoRestart
```

Reboot when prompted. These features supply the virtualization plumbing that WSL2 needs.

---

## Step 2 – Install WSL and Ubuntu 22.04

1. Open an elevated PowerShell or Windows Terminal.
2. Install WSL with Ubuntu and set version 2 by default:

   ```powershell
   wsl --install -d Ubuntu-22.04
   wsl --set-default-version 2
   ```

3. When prompted, create your Linux username/password. This account owns the home directory under `/home/<you>`.
4. Verify status:

   ```powershell
   wsl --status
   wsl -l -v            # should show Ubuntu-22.04, Version 2, Running
   ```

---

## Step 3 – Prepare WSL for GPU compute

1. Launch Ubuntu from the Start menu or via `wsl -d Ubuntu-22.04`.
2. Update and install core dependencies:

   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y build-essential git pkg-config ninja-build cmake \
        curl wget unzip zip autoconf libtool clang llvm
   ```

3. Install Python 3.11 and tooling (Ubuntu 22.04 ships 3.10 by default):

   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa -y
   sudo apt update
   sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
   ```

4. Install CUDA user-space libraries inside WSL. The Windows host driver already contains the kernel component. Use NVIDIA’s CUDA 13.0 repo (replace URL if a newer minor release is posted):

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
   sudo apt update
   sudo apt install -y cuda-toolkit-13-0
   echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

5. Validate GPU visibility from inside WSL:

   ```bash
   nvidia-smi          # should show the RTX 5090 and CUDA 13.0
   nvcc --version      # confirm release 13.0
   ```

If `nvidia-smi` fails, confirm the Windows driver is at least 565.90, reboot, and rerun `wsl --shutdown` before starting Ubuntu again.

---

## Step 4 – Plan shared storage between Windows and WSL

You have two reliable options:

### Option A: Clone the repo on the Windows side and access it via `/mnt/<drive>`

1. Create a Windows folder that will store everything, e.g., `D:\Flux-WSL`.
2. From Ubuntu, use the mounted drive path:

   ```bash
   cd /mnt/d
   git clone https://github.com/<your-org>/FLUX-TRAINING.git Flux-WSL
   cd Flux-WSL
   ```

3. Inside WSL the path is `/mnt/d/Flux-WSL`, while in Windows you can open the same files through File Explorer (`D:\Flux-WSL`). This is the easiest way to drop images/captions from Windows tools while training in Linux.

### Option B: Keep the repo in Linux and expose it to Windows through `\\wsl$`

1. Clone under your Linux home, e.g., `~/projects/flux-training`.
2. Access it from Windows via Explorer → `\\wsl$\Ubuntu-22.04\home\<you>\projects\flux-training`.

This option gives Linux-native filesystem performance (recommended for heavy git operations) while still letting Windows apps open/edit files.

> **Dataset workflow**: Whichever option you choose, store datasets under a path that both environments can see (e.g., `/mnt/d/FluxData`). In Flux configs use the Windows-style path when you edit from Windows, and the corresponding Linux path when running training ( `/mnt/d/FluxData` ).

---

## Step 5 – Set up the Flux training stack inside WSL

1. In Ubuntu, navigate to the repo folder (per Step 4).
2. Install Python dependencies in an isolated virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip setuptools wheel
   ```

3. Follow the Linux quickstart that already exists in the repo (`QUICKSTART.md`, `scripts/*.sh`). For convenience, common commands:

   ```bash
   ./scripts/00_verify_prerequisites.sh
   ./scripts/02_setup_python.sh
   ./scripts/03_build_pytorch.sh
   ./scripts/04_build_xformers.sh
   ./scripts/05_build_blackwell_kernels.sh
   ./scripts/06_setup_sd_scripts.sh
   ```

   - Ensure `TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"` and `CUDA_HOME=/usr/local/cuda-13.0` before building PyTorch/xFormers.
   - Use the same config files described in `FLUX_LORA_TRAINING_REFERENCE.md`; the only difference is the root path (Linux vs Windows) for datasets and outputs.

4. Store outputs in a Windows-visible location so you can inspect `.safetensors` from the host:

   ```bash
   export FLUX_DATA_ROOT=/mnt/d/FluxData
   mkdir -p $FLUX_DATA_ROOT/{dataset,output,samples,logs}
   ```

   Update config TOMLs to reference `/mnt/d/...` paths. Windows apps will simultaneously see `D:\FluxData`.

5. Launch training from WSL:

   ```bash
   cd sd-scripts-cuda13
   source ../.venv/bin/activate
   python flux_train_network.py --config_file config_face.toml --highvram
   ```

---

## Step 6 – Make Windows editing tools part of the workflow

- **Add datasets/captions**: Drop files into `D:\FluxData\dataset\...`; WSL immediately sees them at `/mnt/d/FluxData/dataset`.
- **Use Windows editors** (Photoshop, Notepad++, VS Code) on the shared folders. Avoid saving with CRLF line endings for caption `.txt` files; configure your editor to use LF.
- **Monitor training logs** from Windows with:

  ```powershell
  Get-Content D:\FluxData\logs\training.log -Wait
  ```

  while WSL writes to `/mnt/d/FluxData/logs`.

---

## Step 7 – Backups and maintenance

- To snapshot the entire WSL instance (including packages and the repo if stored inside Linux):

  ```powershell
  wsl --shutdown
  wsl --export Ubuntu-22.04 D:\Backups\ubuntu2204_flux.tar
  ```

- To back up only the shared data, use standard Windows tools (`robocopy D:\FluxData E:\FluxBackup /MIR`).
- Keep WSL up to date:

  ```powershell
  wsl --update          # updates the WSL kernel
  wsl --shutdown        # restart instance so kernel update applies
  ```

- Inside Ubuntu, run `sudo apt update && sudo apt upgrade` weekly.

---

## Troubleshooting quick hits

| Symptom | Fix |
|---------|-----|
| `nvidia-smi` missing inside WSL | Upgrade Windows NVIDIA driver, reboot, `wsl --shutdown`, relaunch Ubuntu |
| `nvcc` reports CUDA 12.x | Reinstall CUDA 13 repo, ensure `/usr/local/cuda-13.0/bin` is ahead of other CUDA paths |
| Slow file access under `/mnt/c` | Keep git repo inside Linux (`~/projects`) and only place large datasets on `/mnt/d`; avoid `/mnt/c/Users/...` for training I/O heavy workloads |
| Windows editor can’t find Linux files | Use `\\wsl$\Ubuntu-22.04\home\<you>\...` UNC path or map it to a drive letter |
| Permission errors on shared folders | Run `sudo chown -R $USER:$USER /mnt/d/FluxData` once to ensure the Linux user owns the directories |

---

## What’s next?

- Use existing documentation (`FLUX_LORA_TRAINING_REFERENCE.md`, `AI_AGENT_RTX5090_SETUP_PROMPT.md`, etc.) for dataset prep, optimizer settings, and training recipes—only the host environment changed.
- Consider scripting a `wsl_flux_launcher.sh` analogue to `FluxTrainer_Launcher.bat` so launching from Windows is a single PowerShell command (`wsl -d Ubuntu-22.04 ~/flux-training/launch.sh`).
- Document every successful run (datasets used, configs, checkpoint paths) in `CONTEXT.md` so future reboots retain continuity.

With WSL2 in place you gain Linux completeness plus Windows convenience: heavy compilation occurs in Ubuntu, while datasets, captions, and outputs remain easy to curate from Windows 11.
