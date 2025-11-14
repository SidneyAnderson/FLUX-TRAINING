# Flux LoRA WSL Migration – Worklog

## Latest status (before reboot)

- Goal: move RTX 5090 Flux LoRA workflow into WSL2 so Linux libraries are complete while datasets stay on NTFS and editable from Windows.
- We wrote `WSL_FLUX_TRAINING_GUIDE.md`, covering everything from enabling features to shared storage paths and CUDA 13.0 setup inside Ubuntu.
- Attempted to enable `Microsoft-Windows-Subsystem-Linux` and `VirtualMachinePlatform` via DISM but hit `Error 740` because the terminal was not elevated.

## Next actions (after reboot)

1. **Open an elevated PowerShell / Windows Terminal (Admin).**
2. Rerun:
   ```powershell
   DISM /Online /Enable-Feature /FeatureName:Microsoft-Windows-Subsystem-Linux /All /NoRestart
   DISM /Online /Enable-Feature /FeatureName:VirtualMachinePlatform /All /NoRestart
   ```
3. Reboot if prompted.
4. Continue with the WSL-on-D setup:
   - Install WSL core (`wsl --install --no-distribution`, `wsl --update`, `wsl --shutdown`).
   - Download Ubuntu rootfs to `D:\WSL\ubuntu2204.tar.gz`.
   - Import with `wsl --import Ubuntu-22.04 D:\WSL\Ubuntu-22.04 ... --version 2`.
   - Follow `WSL_FLUX_TRAINING_GUIDE.md` for CUDA/Python setup and shared-folder workflow.

## Handy references

- `WSL_FLUX_TRAINING_GUIDE.md` – complete WSL playbook.
- `SETUP_CHOICE_GUIDE.md` – context on why we moved away from Windows-isolated setup.

Ping this file on next session so Codex regains context immediately.
