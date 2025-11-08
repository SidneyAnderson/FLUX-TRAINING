#Requires -RunAsAdministrator
###############################################################################
# Visual Studio 2022 Build Tools Installation (Windows 11)
# Installs MSVC v143 and required components for CUDA compilation
###############################################################################

$ErrorActionPreference = "Stop"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "VISUAL STUDIO 2022 BUILD TOOLS INSTALLATION (WINDOWS 11)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$VS_BUILDTOOLS_URL = "https://aka.ms/vs/17/release/vs_buildtools.exe"
$INSTALLER_PATH = "$env:TEMP\vs_buildtools.exe"
$VS_INSTALL_PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"

# Check if already installed
Write-Host "Checking for existing Visual Studio 2022 Build Tools..." -ForegroundColor Yellow
Write-Host ""

if (Test-Path $VS_INSTALL_PATH) {
    Write-Host "✓ Visual Studio 2022 Build Tools found at:" -ForegroundColor Green
    Write-Host "  $VS_INSTALL_PATH" -ForegroundColor Gray
    Write-Host ""

    # Check for MSVC v143
    $msvcPath = Get-ChildItem "$VS_INSTALL_PATH\VC\Tools\MSVC" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($msvcPath) {
        Write-Host "✓ MSVC found: $($msvcPath.Name)" -ForegroundColor Green

        # Set environment variables
        $env:VS170COMNTOOLS = "$VS_INSTALL_PATH\Common7\Tools\"
        $env:DISTUTILS_USE_SDK = "1"

        [Environment]::SetEnvironmentVariable("VS170COMNTOOLS", "$VS_INSTALL_PATH\Common7\Tools\", "Machine")
        [Environment]::SetEnvironmentVariable("DISTUTILS_USE_SDK", "1", "Machine")

        Write-Host ""
        Write-Host "=" * 80 -ForegroundColor Green
        Write-Host "✅ VISUAL STUDIO 2022 BUILD TOOLS VERIFIED" -ForegroundColor Green
        Write-Host "=" * 80 -ForegroundColor Green
        Write-Host ""
        Write-Host "Installation Path: $VS_INSTALL_PATH" -ForegroundColor Cyan
        Write-Host "MSVC Version: $($msvcPath.Name)" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "⚠  IMPORTANT: All compilation must be done in" -ForegroundColor Yellow
        Write-Host "   'x64 Native Tools Command Prompt for VS 2022'" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Next step: .\scripts\04_setup_python_env.ps1" -ForegroundColor Cyan
        Write-Host ""
        exit 0
    }
}

# Need to install
Write-Host "Visual Studio 2022 Build Tools not found. Installation required." -ForegroundColor Yellow
Write-Host ""
Write-Host "This will install:" -ForegroundColor Cyan
Write-Host "  • C++ Build Tools" -ForegroundColor White
Write-Host "  • MSVC v143 (x64/x86)" -ForegroundColor White
Write-Host "  • Windows 11 SDK" -ForegroundColor White
Write-Host "  • CMake tools" -ForegroundColor White
Write-Host ""
Write-Host "⚠  This installation requires ~7GB disk space" -ForegroundColor Yellow
Write-Host "⚠  Installation will take 15-30 minutes" -ForegroundColor Yellow
Write-Host ""

$answer = Read-Host "Proceed with installation? (Y/N)"
if ($answer -ne "Y" -and $answer -ne "y") {
    Write-Host "Installation cancelled." -ForegroundColor Yellow
    exit 1
}

# Download Visual Studio Build Tools
Write-Host ""
Write-Host "Downloading Visual Studio 2022 Build Tools..." -ForegroundColor Yellow
Write-Host "  Source: $VS_BUILDTOOLS_URL" -ForegroundColor Gray
Write-Host "  This may take a few minutes..." -ForegroundColor Gray
Write-Host ""

try {
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $VS_BUILDTOOLS_URL -OutFile $INSTALLER_PATH -UseBasicParsing
    $ProgressPreference = 'Continue'
    Write-Host "✓ Download complete" -ForegroundColor Green
} catch {
    $ProgressPreference = 'Continue'
    Write-Host "✗ Download failed: $_" -ForegroundColor Red
    exit 1
}

# Install Visual Studio Build Tools
Write-Host ""
Write-Host "Installing Visual Studio 2022 Build Tools..." -ForegroundColor Yellow
Write-Host "  This will take 15-30 minutes..." -ForegroundColor Gray
Write-Host "  A Visual Studio Installer window will appear - do not close it" -ForegroundColor Gray
Write-Host ""

$installArgs = @(
    "--quiet"
    "--wait"
    "--norestart"
    "--nocache"
    "--add", "Microsoft.VisualStudio.Workload.VCTools"
    "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
    "--add", "Microsoft.VisualStudio.Component.VC.CMake.Project"
    "--add", "Microsoft.VisualStudio.Component.Windows11SDK.22621"
    "--add", "Microsoft.VisualStudio.Component.VC.ATLMFC"
    "--includeRecommended"
)

Write-Host "Starting installation..." -ForegroundColor Gray
$process = Start-Process -FilePath $INSTALLER_PATH -ArgumentList $installArgs -Wait -PassThru -NoNewWindow

if ($process.ExitCode -eq 0 -or $process.ExitCode -eq 3010) {
    Write-Host "✓ Installation complete" -ForegroundColor Green

    if ($process.ExitCode -eq 3010) {
        Write-Host "  ⚠ Reboot required (exit code 3010)" -ForegroundColor Yellow
    }
} else {
    Write-Host "✗ Installation failed with exit code: $($process.ExitCode)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common exit codes:" -ForegroundColor Yellow
    Write-Host "  • -1073720687: Installation cancelled by user" -ForegroundColor Gray
    Write-Host "  • 1602: Installation cancelled by user" -ForegroundColor Gray
    Write-Host "  • Other: See Visual Studio Installer log" -ForegroundColor Gray
    exit 1
}

# Verify installation
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow

if (Test-Path $VS_INSTALL_PATH) {
    Write-Host "✓ Visual Studio 2022 Build Tools installed" -ForegroundColor Green

    # Check MSVC
    $msvcPath = Get-ChildItem "$VS_INSTALL_PATH\VC\Tools\MSVC" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($msvcPath) {
        Write-Host "✓ MSVC installed: $($msvcPath.Name)" -ForegroundColor Green
    } else {
        Write-Host "⚠ MSVC not found" -ForegroundColor Yellow
    }

    # Check Windows SDK
    $sdkPath = "$VS_INSTALL_PATH\VC\Auxiliary\Build"
    if (Test-Path $sdkPath) {
        Write-Host "✓ Build tools found" -ForegroundColor Green
    }
} else {
    Write-Host "✗ Installation verification failed" -ForegroundColor Red
    exit 1
}

# Set environment variables
Write-Host ""
Write-Host "Configuring environment variables..." -ForegroundColor Yellow
[Environment]::SetEnvironmentVariable("VS170COMNTOOLS", "$VS_INSTALL_PATH\Common7\Tools\", "Machine")
[Environment]::SetEnvironmentVariable("DISTUTILS_USE_SDK", "1", "Machine")

$env:VS170COMNTOOLS = "$VS_INSTALL_PATH\Common7\Tools\"
$env:DISTUTILS_USE_SDK = "1"

Write-Host "✓ Environment configured" -ForegroundColor Green

# Cleanup
Remove-Item $INSTALLER_PATH -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Green
Write-Host "✅ VISUAL STUDIO 2022 BUILD TOOLS INSTALLATION COMPLETE" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green
Write-Host ""
Write-Host "Installation Path: $VS_INSTALL_PATH" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠  IMPORTANT FOR COMPILATION:" -ForegroundColor Yellow
Write-Host ""
Write-Host "All CUDA/PyTorch compilation MUST be done in:" -ForegroundColor White
Write-Host "  'x64 Native Tools Command Prompt for VS 2022'" -ForegroundColor Cyan
Write-Host ""
Write-Host "To open it:" -ForegroundColor White
Write-Host "  1. Press Windows key" -ForegroundColor Gray
Write-Host "  2. Type: 'x64 native tools'" -ForegroundColor Gray
Write-Host "  3. Run as Administrator" -ForegroundColor Gray
Write-Host ""
Write-Host "Or from PowerShell:" -ForegroundColor White
Write-Host "  `$vsPath = '$VS_INSTALL_PATH'" -ForegroundColor Gray
Write-Host "  & `"`$vsPath\VC\Auxiliary\Build\vcvars64.bat`"" -ForegroundColor Gray
Write-Host ""

if ($process.ExitCode -eq 3010) {
    Write-Host "⚠  REBOOT REQUIRED" -ForegroundColor Yellow
    Write-Host "   Please reboot your system before continuing." -ForegroundColor White
    Write-Host ""
    $reboot = Read-Host "Reboot now? (Y/N)"
    if ($reboot -eq "Y" -or $reboot -eq "y") {
        Write-Host "Rebooting..." -ForegroundColor Yellow
        Restart-Computer -Force
    }
} else {
    Write-Host "Next step: .\scripts\04_setup_python_env.ps1" -ForegroundColor Cyan
    Write-Host ""
}
