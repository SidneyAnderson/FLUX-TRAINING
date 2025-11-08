#Requires -RunAsAdministrator
###############################################################################
# Python 3.11.9 Clean Installation (Windows 11)
# Installs EXACT version for maximum compatibility
###############################################################################

$ErrorActionPreference = "Stop"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "PYTHON 3.11.9 CLEAN INSTALLATION (WINDOWS 11)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$PYTHON_VERSION = "3.11.9"
$PYTHON_URL = "https://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION-amd64.exe"
$INSTALL_DIR = "C:\Python311"
$TEMP_INSTALLER = "$env:TEMP\python-$PYTHON_VERSION-amd64.exe"

# Expected SHA256 hash for Python 3.11.9
$EXPECTED_HASH = "A90CE56F31AE8C2C5B07751BE76D972D9D5DC299F510F93DEDD3D34852D89111"

# Step 1: Remove all existing Python installations
Write-Host "[1/5] Removing existing Python installations..." -ForegroundColor Yellow
Write-Host ""

# Find and uninstall Python packages
$pythonApps = Get-WmiObject -Class Win32_Product | Where-Object { $_.Name -like "*Python*" }
if ($pythonApps) {
    foreach ($app in $pythonApps) {
        Write-Host "  Removing: $($app.Name)" -ForegroundColor Red
        try {
            $app.Uninstall() | Out-Null
        } catch {
            Write-Host "  Warning: Could not uninstall $($app.Name)" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  No Python installations found via WMI" -ForegroundColor Gray
}

# Clean registry
Write-Host "  Cleaning registry..." -ForegroundColor Gray
$registryPaths = @(
    "HKLM:\SOFTWARE\Python",
    "HKCU:\SOFTWARE\Python",
    "HKLM:\SOFTWARE\Wow6432Node\Python"
)

foreach ($path in $registryPaths) {
    if (Test-Path $path) {
        Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  Cleaned: $path" -ForegroundColor Gray
    }
}

# Clean PATH environment variables
Write-Host "  Cleaning PATH environment variables..." -ForegroundColor Gray

# Machine PATH
$machinePath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
$newMachinePath = ($machinePath -split ';' | Where-Object { $_ -notmatch 'Python' }) -join ';'
[Environment]::SetEnvironmentVariable("PATH", $newMachinePath, "Machine")

# User PATH
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
$newUserPath = ($userPath -split ';' | Where-Object { $_ -notmatch 'Python' }) -join ';'
[Environment]::SetEnvironmentVariable("PATH", $newUserPath, "User")

# Remove Python directories
Write-Host "  Removing Python directories..." -ForegroundColor Gray
$pythonDirs = @(
    "$env:LOCALAPPDATA\Programs\Python",
    "$env:APPDATA\Python",
    "C:\Python*"
)

foreach ($dir in $pythonDirs) {
    Get-Item $dir -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# Remove from C:\ specifically
Get-ChildItem C:\ -Directory -Filter "Python*" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "✓ Cleanup complete" -ForegroundColor Green
Write-Host ""

# Step 2: Download Python 3.11.9
Write-Host "[2/5] Downloading Python $PYTHON_VERSION..." -ForegroundColor Yellow

if (Test-Path $TEMP_INSTALLER) {
    Write-Host "  Installer already exists, verifying..." -ForegroundColor Gray
} else {
    Write-Host "  Downloading from: $PYTHON_URL" -ForegroundColor Gray
    Write-Host "  This may take a few minutes..." -ForegroundColor Gray

    # Download with progress
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $PYTHON_URL -OutFile $TEMP_INSTALLER -UseBasicParsing
        $ProgressPreference = 'Continue'
    } catch {
        $ProgressPreference = 'Continue'
        Write-Host "✗ Download failed: $_" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✓ Download complete" -ForegroundColor Green
Write-Host ""

# Step 3: Verify installer integrity
Write-Host "[3/5] Verifying installer integrity..." -ForegroundColor Yellow

$actualHash = (Get-FileHash -Path $TEMP_INSTALLER -Algorithm SHA256).Hash
Write-Host "  Expected: $EXPECTED_HASH" -ForegroundColor Gray
Write-Host "  Actual:   $actualHash" -ForegroundColor Gray

if ($actualHash -eq $EXPECTED_HASH) {
    Write-Host "✓ Hash verified" -ForegroundColor Green
} else {
    Write-Host "✗ Hash mismatch! Installer may be corrupted or tampered." -ForegroundColor Red
    Write-Host "  Deleting installer..." -ForegroundColor Yellow
    Remove-Item $TEMP_INSTALLER -Force
    exit 1
}
Write-Host ""

# Step 4: Install Python 3.11.9
Write-Host "[4/5] Installing Python $PYTHON_VERSION..." -ForegroundColor Yellow
Write-Host "  Install directory: $INSTALL_DIR" -ForegroundColor Gray
Write-Host "  This will take 2-3 minutes..." -ForegroundColor Gray
Write-Host ""

$installArgs = @(
    "/quiet"
    "InstallAllUsers=1"
    "TargetDir=$INSTALL_DIR"
    "PrependPath=1"
    "Include_test=1"
    "Include_pip=1"
    "Include_doc=0"
    "Include_dev=1"
    "Include_debug=1"
    "Include_symbols=1"
    "Include_tcltk=1"
    "InstallLauncherAllUsers=1"
    "CompileAll=1"
)

$process = Start-Process -FilePath $TEMP_INSTALLER -ArgumentList $installArgs -Wait -PassThru -NoNewWindow

if ($process.ExitCode -eq 0) {
    Write-Host "✓ Installation complete" -ForegroundColor Green
} else {
    Write-Host "✗ Installation failed with exit code $($process.ExitCode)" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 5: Configure environment and verify
Write-Host "[5/5] Configuring environment..." -ForegroundColor Yellow

# Set system environment variables
Write-Host "  Setting environment variables..." -ForegroundColor Gray
[Environment]::SetEnvironmentVariable("PYTHONHOME", $INSTALL_DIR, "Machine")
[Environment]::SetEnvironmentVariable("PYTHONPATH", "$INSTALL_DIR\Lib;$INSTALL_DIR\DLLs", "Machine")

# Update PATH (ensure Python is first)
$machinePath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
$pythonPaths = "$INSTALL_DIR;$INSTALL_DIR\Scripts"

# Remove any existing Python paths and add new ones at the beginning
$cleanPath = ($machinePath -split ';' | Where-Object { $_ -notmatch 'Python' }) -join ';'
$newPath = "$pythonPaths;$cleanPath"
[Environment]::SetEnvironmentVariable("PATH", $newPath, "Machine")

# Refresh environment for current session
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
$env:PYTHONHOME = $INSTALL_DIR
$env:PYTHONPATH = "$INSTALL_DIR\Lib;$INSTALL_DIR\DLLs"

Write-Host "✓ Environment configured" -ForegroundColor Green
Write-Host ""

# Verify installation
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "VERIFICATION" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

# Test Python
Write-Host ""
Write-Host "Testing Python installation..." -ForegroundColor Yellow

$pythonExe = "$INSTALL_DIR\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "✗ Python executable not found at: $pythonExe" -ForegroundColor Red
    exit 1
}

# Check version
$versionOutput = & $pythonExe --version 2>&1
Write-Host "  $versionOutput" -ForegroundColor Gray

if ($versionOutput -match "Python $PYTHON_VERSION") {
    Write-Host "✓ Python $PYTHON_VERSION verified" -ForegroundColor Green
} else {
    Write-Host "✗ Wrong Python version: $versionOutput" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $pythonExe -m pip install --upgrade pip --quiet
Write-Host "✓ pip upgraded" -ForegroundColor Green

# Install essential build tools
Write-Host ""
Write-Host "Installing essential build tools..." -ForegroundColor Yellow
& $pythonExe -m pip install setuptools==69.0.3 wheel==0.42.0 --quiet
Write-Host "✓ Build tools installed" -ForegroundColor Green

# Create verification script
Write-Host ""
Write-Host "Running comprehensive verification..." -ForegroundColor Yellow

$verifyScript = @'
import sys
import platform
import struct

print("=" * 80)
print("PYTHON VERIFICATION REPORT")
print("=" * 80)
print(f"Version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
print(f"Bits: {struct.calcsize('P') * 8}-bit")
print(f"Executable: {sys.executable}")
print(f"Prefix: {sys.prefix}")
print("=" * 80)

# Verify exact version
assert sys.version_info[:3] == (3, 11, 9), f"Wrong version: {sys.version_info[:3]}"
assert struct.calcsize('P') * 8 == 64, "Not 64-bit Python!"

print("")
print("✅ All checks passed - Python 3.11.9 x64 ready for RTX 5090!")
'@

& $pythonExe -c $verifyScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host "✅ PYTHON 3.11.9 INSTALLATION COMPLETE!" -ForegroundColor Green
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host ""
    Write-Host "Installation Directory: $INSTALL_DIR" -ForegroundColor Cyan
    Write-Host "Python Command: python" -ForegroundColor Cyan
    Write-Host "Pip Command: pip" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "⚠  IMPORTANT: Close and reopen PowerShell/Command Prompt" -ForegroundColor Yellow
    Write-Host "   for PATH changes to take full effect" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Next step: .\scripts\02_verify_cuda.ps1" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "✗ Verification failed" -ForegroundColor Red
    exit 1
}

# Cleanup
Remove-Item $TEMP_INSTALLER -Force -ErrorAction SilentlyContinue
