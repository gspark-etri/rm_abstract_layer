#!/bin/bash
# Setup script using uv (ultra-fast Python package installer)
# https://github.com/astral-sh/uv

set -e

echo "========================================="
echo "RM Abstract - Setup with uv"
echo "========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "[!] uv is not installed. Installing..."

    # Install uv
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    else
        echo "[ERROR] Unsupported OS: $OSTYPE"
        echo "Please install uv manually: https://github.com/astral-sh/uv"
        exit 1
    fi

    echo "[OK] uv installed successfully"
else
    echo "[OK] uv is already installed ($(uv --version))"
fi

echo ""
echo "Creating virtual environment with uv..."
uv venv

echo ""
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

echo ""
echo "Installing package in development mode..."
echo "Choose installation type:"
echo "  1) Basic (CPU only)"
echo "  2) GPU support"
echo "  3) Development (with all tools)"
echo "  4) All (GPU + Development)"
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "Installing basic package..."
        uv pip install -e .
        ;;
    2)
        echo "Installing with GPU support..."
        uv pip install -e ".[gpu]"
        ;;
    3)
        echo "Installing with development tools..."
        uv pip install -e ".[dev]"
        ;;
    4)
        echo "Installing everything..."
        uv pip install -e ".[all]"
        ;;
    *)
        echo "Invalid choice. Installing basic package..."
        uv pip install -e .
        ;;
esac

echo ""
echo "========================================="
echo "[OK] Setup completed successfully!"
echo "========================================="
echo ""
echo "Virtual environment: .venv"
echo ""
echo "To activate the environment:"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "  source .venv/Scripts/activate"
else
    echo "  source .venv/bin/activate"
fi
echo ""
echo "To verify installation:"
echo "  python -m rm_abstract.verify"
echo ""
echo "To list plugins:"
echo "  rm-abstract list-plugins"
echo ""
