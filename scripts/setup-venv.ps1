# Setup script using standard Python venv for Windows (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "RM Abstract - Setup with venv (Windows)" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] $pythonVersion" -ForegroundColor Green

    # Extract version number
    if ($pythonVersion -match "Python (\d+\.\d+)") {
        $version = [version]$matches[1]
        $required = [version]"3.9"

        if ($version -lt $required) {
            Write-Host "[ERROR] Python 3.9 or higher is required" -ForegroundColor Red
            Write-Host "Current version: $version" -ForegroundColor Red
            exit 1
        }
    }
}
catch {
    Write-Host "[ERROR] Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.9 or higher from https://python.org" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv

Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host ""
Write-Host "Installing package in development mode..." -ForegroundColor Cyan
Write-Host "Choose installation type:" -ForegroundColor Yellow
Write-Host "  1) Basic (CPU only)"
Write-Host "  2) GPU support"
Write-Host "  3) Development (with all tools)"
Write-Host "  4) All (GPU + Development)"
$choice = Read-Host "Enter choice [1-4]"

switch ($choice) {
    "1" {
        Write-Host "Installing basic package..." -ForegroundColor Cyan
        pip install -e .
    }
    "2" {
        Write-Host "Installing with GPU support..." -ForegroundColor Cyan
        pip install -e ".[gpu]"
    }
    "3" {
        Write-Host "Installing with development tools..." -ForegroundColor Cyan
        pip install -e ".[dev]"
    }
    "4" {
        Write-Host "Installing everything..." -ForegroundColor Cyan
        pip install -e ".[all]"
    }
    default {
        Write-Host "Invalid choice. Installing basic package..." -ForegroundColor Yellow
        pip install -e .
    }
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "[OK] Setup completed successfully!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Virtual environment: venv" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White
Write-Host ""
Write-Host "To verify installation:" -ForegroundColor Yellow
Write-Host "  python -m rm_abstract.verify" -ForegroundColor White
Write-Host ""
Write-Host "To list plugins:" -ForegroundColor Yellow
Write-Host "  rm-abstract list-plugins" -ForegroundColor White
Write-Host ""
