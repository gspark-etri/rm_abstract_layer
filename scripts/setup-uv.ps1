# Setup script using uv for Windows (PowerShell)
# https://github.com/astral-sh/uv

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "RM Abstract - Setup with uv (Windows)" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if uv is installed
try {
    $uvVersion = uv --version
    Write-Host "[OK] uv is already installed ($uvVersion)" -ForegroundColor Green
}
catch {
    Write-Host "[!] uv is not installed. Installing..." -ForegroundColor Yellow

    # Install uv
    try {
        irm https://astral.sh/uv/install.ps1 | iex
        Write-Host "[OK] uv installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] Failed to install uv" -ForegroundColor Red
        Write-Host "Please install uv manually: https://github.com/astral-sh/uv" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Creating virtual environment with uv..." -ForegroundColor Cyan
uv venv

Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

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
        uv pip install -e .
    }
    "2" {
        Write-Host "Installing with GPU support..." -ForegroundColor Cyan
        uv pip install -e ".[gpu]"
    }
    "3" {
        Write-Host "Installing with development tools..." -ForegroundColor Cyan
        uv pip install -e ".[dev]"
    }
    "4" {
        Write-Host "Installing everything..." -ForegroundColor Cyan
        uv pip install -e ".[all]"
    }
    default {
        Write-Host "Invalid choice. Installing basic package..." -ForegroundColor Yellow
        uv pip install -e .
    }
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "[OK] Setup completed successfully!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Virtual environment: .venv" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To verify installation:" -ForegroundColor Yellow
Write-Host "  python -m rm_abstract.verify" -ForegroundColor White
Write-Host ""
Write-Host "To list plugins:" -ForegroundColor Yellow
Write-Host "  rm-abstract list-plugins" -ForegroundColor White
Write-Host ""
