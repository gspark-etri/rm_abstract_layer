#!/bin/bash
# Setup script using standard Python venv

set -e

echo "========================================="
echo "RM Abstract - Setup with venv"
echo "========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "[ERROR] Python $REQUIRED_VERSION or higher is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo "[OK] Python version: $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

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
        pip install -e .
        ;;
    2)
        echo "Installing with GPU support..."
        pip install -e ".[gpu]"
        ;;
    3)
        echo "Installing with development tools..."
        pip install -e ".[dev]"
        ;;
    4)
        echo "Installing everything..."
        pip install -e ".[all]"
        ;;
    *)
        echo "Invalid choice. Installing basic package..."
        pip install -e .
        ;;
esac

echo ""
echo "========================================="
echo "[OK] Setup completed successfully!"
echo "========================================="
echo ""
echo "Virtual environment: venv"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "To verify installation:"
echo "  python -m rm_abstract.verify"
echo ""
echo "To list plugins:"
echo "  rm-abstract list-plugins"
echo ""
