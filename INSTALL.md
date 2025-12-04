# Installation Guide

Complete guide for installing LLM Heterogeneous Resource Orchestrator.

---

## Table of Contents

1. [Quick Install](#quick-install)
2. [Installation Methods](#installation-methods)
3. [Backend-Specific Installation](#backend-specific-installation)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

---

## Quick Install

### For CPU-only (No GPU/NPU)

```bash
pip install rm-abstract
```

### For GPU (NVIDIA CUDA)

```bash
# Install PyTorch with CUDA first
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install rm-abstract with GPU support
pip install rm-abstract[gpu]
```

### For Development

```bash
git clone https://github.com/yourusername/rm_abstract_layer.git
cd rm_abstract_layer
pip install -e ".[dev]"
```

---

## Installation Methods

### Method 1: PyPI (Recommended for Users)

```bash
# Basic installation
pip install rm-abstract

# With specific backend support
pip install rm-abstract[gpu]           # GPU support
pip install rm-abstract[npu-rbln]      # Rebellions NPU
pip install rm-abstract[npu-furiosa]   # FuriosaAI NPU

# Install everything
pip install rm-abstract[all]
```

### Method 2: From Source (Recommended for Developers)

```bash
# Clone repository
git clone https://github.com/yourusername/rm_abstract_layer.git
cd rm_abstract_layer

# Install in editable mode
pip install -e .

# Or with extras
pip install -e ".[dev,gpu]"
```

### Method 3: Using setup.py directly

```bash
# Clone and navigate to repository
git clone https://github.com/yourusername/rm_abstract_layer.git
cd rm_abstract_layer

# Install
python setup.py install

# Or develop mode
python setup.py develop
```

---

## Backend-Specific Installation

### GPU Backend (vLLM)

#### Prerequisites
- NVIDIA GPU with CUDA 11.8+ or 12.1+
- NVIDIA Driver 525+ (for CUDA 12.1)

#### Installation

```bash
# 1. Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Install vLLM
pip install vllm>=0.2.0

# 3. Install rm-abstract
pip install rm-abstract[gpu]
```

#### Verification

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

### NPU Backend (Rebellions ATOM)

#### Prerequisites
- Rebellions ATOM NPU hardware
- RBLN SDK installed

#### Installation

```bash
# 1. Install RBLN SDK (follow vendor instructions)
# Usually something like:
sudo apt-get install rbln-sdk

# 2. Install rm-abstract with RBLN support
pip install rm-abstract[npu-rbln]
```

#### Verification

```bash
# Check RBLN SDK
rbln-smi  # Should show NPU devices

# Test rm-abstract
python -c "import rm_abstract; print(rm_abstract.list_plugins())"
```

### NPU Backend (FuriosaAI RNGD)

#### Prerequisites
- FuriosaAI RNGD NPU hardware
- Furiosa SDK installed

#### Installation

```bash
# 1. Install Furiosa SDK (follow vendor instructions)
sudo apt-get install furiosa-sdk

# 2. Install rm-abstract with Furiosa support
pip install rm-abstract[npu-furiosa]
```

### CPU Backend (Fallback)

CPU backend is included by default. No additional installation needed.

```bash
pip install rm-abstract
```

---

## Verification

### Quick Verification

Run the built-in verification script:

```bash
python -m rm_abstract.verify
```

Expected output:
```
========================================
RM Abstract - Installation Verification
========================================

[✓] PyTorch: 2.1.0
[✓] Core modules: OK
[✓] Available backends:
    - cpu: Available
    - gpu: Not available (vLLM not installed)
    - rbln: Not available (SDK not found)
    - furiosa: Not available (SDK not found)

[✓] Plugin system: OK
[✓] All tests passed!
```

### Manual Verification

```python
import rm_abstract

# List available plugins
plugins = rm_abstract.list_plugins(available_only=True)
print("Available plugins:", list(plugins.keys()))

# Initialize with auto-selection
rm_abstract.init(device="auto", use_plugin_system=True)
print("Initialization successful!")
```

### Run Example Tests

```bash
# Run simple plugin test
python examples/simple_plugin_test.py

# Run GPU to NPU migration demo
python examples/gpu_to_npu_migration.py
```

---

## Troubleshooting

### Common Issues

#### 1. "No module named 'rm_abstract'"

**Solution:**
```bash
# Make sure you're in the right directory
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/rm_abstract_layer/src"
```

#### 2. "CUDA not available"

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 3. "vLLM import error"

**Solution:**
```bash
# vLLM requires specific CUDA version
pip install vllm --no-cache-dir

# If still fails, try building from source
pip install git+https://github.com/vllm-project/vllm.git
```

#### 4. "Plugin not found"

**Solution:**
```python
# Check if backend is properly installed
import rm_abstract
print(rm_abstract.list_plugins(available_only=False))

# Install missing backend
pip install rm-abstract[gpu]  # or [npu-rbln], [npu-furiosa]
```

#### 5. Windows-specific encoding issues

**Solution:**
```bash
# Set UTF-8 encoding
set PYTHONIOENCODING=utf-8

# Or in PowerShell
$env:PYTHONIOENCODING="utf-8"
```

### Getting Help

If you encounter issues:

1. **Check documentation**: See [PLUGIN_ARCHITECTURE.md](PLUGIN_ARCHITECTURE.md)
2. **Run verification**: `python -m rm_abstract.verify`
3. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
4. **Report issues**: https://github.com/yourusername/rm_abstract_layer/issues

---

## Development Installation

For contributing to the project:

```bash
# Clone repository
git clone https://github.com/yourusername/rm_abstract_layer.git
cd rm_abstract_layer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,gpu]"

# Run tests
pytest tests/

# Run linters
black src/ tests/ examples/
ruff check src/ tests/ examples/
mypy src/
```

---

## Uninstallation

```bash
# Uninstall rm-abstract
pip uninstall rm-abstract

# Clean up cache (optional)
rm -rf ~/.rm_abstract/
```

---

## Next Steps

After installation:

1. **Read the README**: [README.md](README.md)
2. **Try examples**: Start with [examples/simple_plugin_test.py](examples/simple_plugin_test.py)
3. **Explore architecture**: See [PLUGIN_ARCHITECTURE.md](PLUGIN_ARCHITECTURE.md)
4. **Check migration guide**: [examples/gpu_to_npu_migration.py](examples/gpu_to_npu_migration.py)

Happy orchestrating! 🚀
