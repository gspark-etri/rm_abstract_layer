# Environment Setup Guide

Complete guide for setting up development environment with uv, venv, or traditional pip.

---

## Table of Contents

1. [Quick Setup](#quick-setup)
2. [Using uv (Recommended)](#using-uv-recommended)
3. [Using venv](#using-venv)
4. [Using pip (Traditional)](#using-pip-traditional)
5. [Lock Files & Reproducibility](#lock-files--reproducibility)
6. [Version Management](#version-management)
7. [Troubleshooting](#troubleshooting)

---

## Quick Setup

### Method 1: uv (Fastest & Recommended)

```bash
# Linux/Mac
bash scripts/setup-uv.sh

# Windows PowerShell
.\scripts\setup-uv.ps1
```

### Method 2: venv (Standard Python)

```bash
# Linux/Mac
bash scripts/setup-venv.sh

# Windows PowerShell
.\scripts\setup-venv.ps1
```

---

## Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver, written in Rust.

### Why uv?

- ⚡ **10-100x faster** than pip
- 🔒 **Built-in dependency resolution** with lock files
- 📦 **Compatibility** with pip and PyPI
- 🎯 **Reproducible** builds

### Installation

#### Linux/Mac

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows (PowerShell)

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup Environment

#### Automated Setup

```bash
# Linux/Mac
bash scripts/setup-uv.sh

# Windows
.\scripts\setup-uv.ps1
```

#### Manual Setup

```bash
# 1. Create virtual environment
uv venv

# 2. Activate environment
# Linux/Mac:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\activate

# 3. Install package
# Basic
uv pip install -e .

# With GPU support
uv pip install -e ".[gpu]"

# Development mode
uv pip install -e ".[dev]"

# Everything
uv pip install -e ".[all]"
```

### Lock File Management

uv supports lock files for reproducible builds:

```bash
# Generate lock file (this ensures exact versions)
uv pip freeze > requirements.lock

# Install from lock file
uv pip install -r requirements.lock

# Update dependencies
uv pip install -e ".[all]" --upgrade
uv pip freeze > requirements.lock
```

### Sync Dependencies

```bash
# Install exactly what's in lock file
uv pip sync requirements.lock
```

---

## Using venv

Standard Python virtual environment.

### Setup

#### Automated Setup

```bash
# Linux/Mac
bash scripts/setup-venv.sh

# Windows
.\scripts\setup-venv.ps1
```

#### Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
# Linux/Mac:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install package
# Basic
pip install -e .

# With GPU support
pip install -e ".[gpu]"

# Development mode
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

### Lock File

```bash
# Generate requirements file
pip freeze > requirements.lock

# Install from requirements file
pip install -r requirements.lock
```

---

## Using pip (Traditional)

Direct installation without virtual environment (not recommended for development).

```bash
# Basic installation
pip install -e .

# With extras
pip install -e ".[gpu,dev]"
```

---

## Lock Files & Reproducibility

### Why Lock Files?

Lock files ensure:
- ✅ **Exact versions** of all dependencies
- ✅ **Reproducible builds** across machines
- ✅ **No version conflicts**
- ✅ **Deterministic installations**

### Creating Lock Files

#### Using uv

```bash
# Install with all extras
uv pip install -e ".[all]"

# Generate lock file
uv pip freeze > requirements.lock

# Commit lock file
git add requirements.lock
git commit -m "Update dependency lock file"
```

#### Using pip

```bash
# Install with all extras
pip install -e ".[all]"

# Generate lock file
pip freeze > requirements.lock
```

### Using Lock Files

```bash
# With uv (faster)
uv pip sync requirements.lock

# With pip
pip install -r requirements.lock
```

### Lock File Best Practices

1. **Separate lock files** for different environments:
   ```
   requirements.lock         # Production
   requirements-dev.lock     # Development
   requirements-gpu.lock     # GPU environment
   ```

2. **Update regularly**:
   ```bash
   # Update all dependencies
   uv pip install -e ".[all]" --upgrade
   uv pip freeze > requirements.lock
   ```

3. **CI/CD Integration**:
   ```yaml
   # .github/workflows/ci.yml
   - name: Install dependencies
     run: uv pip sync requirements.lock
   ```

---

## Version Management

### Pinning Versions

#### In pyproject.toml

```toml
[project]
dependencies = [
    "torch>=2.0.0,<3.0.0",  # Allow minor updates
    "numpy>=1.20.0,<2.0.0", # Strict upper bound
]
```

#### Version Constraints

| Constraint | Meaning | Example |
|------------|---------|---------|
| `==` | Exact version | `torch==2.0.0` |
| `>=` | Minimum version | `torch>=2.0.0` |
| `<` | Maximum version (exclusive) | `torch<3.0.0` |
| `~=` | Compatible release | `torch~=2.0.0` (allows 2.0.x) |
| No constraint | Latest | `torch` |

### Checking Versions

```bash
# With uv
uv pip list

# With pip
pip list

# Check specific package
uv pip show torch

# Check outdated packages
uv pip list --outdated
```

### Updating Dependencies

```bash
# Update specific package
uv pip install --upgrade torch

# Update all packages
uv pip install -e ".[all]" --upgrade

# Update lock file
uv pip freeze > requirements.lock
```

---

## Environment Variables

### Setting Up

```bash
# Linux/Mac
export RM_DEVICE="auto"
export RM_USE_PLUGINS="true"
export RM_CACHE_DIR="$HOME/.rm_abstract/cache"

# Windows PowerShell
$env:RM_DEVICE="auto"
$env:RM_USE_PLUGINS="true"
$env:RM_CACHE_DIR="$HOME\.rm_abstract\cache"
```

### .env File

Create `.env` file in project root:

```bash
# .env
RM_DEVICE=auto
RM_USE_PLUGINS=true
RM_CACHE_DIR=~/.rm_abstract/cache
```

Load with python-dotenv:

```python
from dotenv import load_dotenv
load_dotenv()

import rm_abstract
rm_abstract.init()  # Uses env variables
```

---

## Multiple Environments

### Development Setup

```bash
# Create dev environment
uv venv --name dev-env
source dev-env/bin/activate  # or .\dev-env\Scripts\activate
uv pip install -e ".[dev]"
```

### GPU Environment

```bash
# Create GPU environment
uv venv --name gpu-env
source gpu-env/bin/activate
uv pip install -e ".[gpu]"
```

### Testing Environment

```bash
# Create test environment
uv venv --name test-env
source test-env/bin/activate
uv pip install -e ".[dev]"
pytest tests/
```

---

## Comparison Table

| Feature | uv | venv + pip | pip only |
|---------|----|-----------| ---------|
| Speed | ⚡⚡⚡ | ⚡ | ⚡ |
| Lock files | ✅ Built-in | ⚠️ Manual | ⚠️ Manual |
| Isolation | ✅ | ✅ | ❌ |
| Reproducibility | ✅ | ⚠️ | ❌ |
| Setup time | ~10s | ~30s | N/A |
| Recommended for | All | Standard | Quick tests |

---

## Troubleshooting

### uv Installation Issues

```bash
# Check if uv is installed
uv --version

# Reinstall uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if needed)
export PATH="$HOME/.cargo/bin:$PATH"
```

### Virtual Environment Issues

```bash
# Remove existing environment
rm -rf venv .venv

# Create fresh environment
uv venv  # or python -m venv venv

# Verify activation
which python  # Should point to venv
```

### Dependency Conflicts

```bash
# With uv (better conflict resolution)
uv pip install -e ".[all]" --resolution=highest

# Check for conflicts
uv pip check
```

### Lock File Out of Sync

```bash
# Regenerate lock file
uv pip install -e ".[all]" --upgrade
uv pip freeze > requirements.lock

# Force sync
uv pip sync requirements.lock --force-reinstall
```

### Permission Issues

```bash
# Linux/Mac - Fix permissions
chmod +x scripts/*.sh

# Windows - Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Best Practices

### For Development

1. **Always use virtual environments**
   ```bash
   uv venv  # Create isolated environment
   ```

2. **Use lock files for reproducibility**
   ```bash
   uv pip freeze > requirements.lock
   ```

3. **Install in editable mode**
   ```bash
   uv pip install -e ".[dev]"
   ```

4. **Update dependencies regularly**
   ```bash
   uv pip install -e ".[all]" --upgrade
   ```

### For Production

1. **Use exact versions from lock file**
   ```bash
   uv pip sync requirements.lock
   ```

2. **Pin critical dependencies**
   ```toml
   torch = "==2.0.0"
   ```

3. **Test before deploying**
   ```bash
   pytest tests/
   ```

### For CI/CD

1. **Use lock files**
   ```yaml
   - run: uv pip sync requirements.lock
   ```

2. **Cache dependencies**
   ```yaml
   - uses: actions/cache@v3
     with:
       path: ~/.cache/uv
   ```

3. **Test multiple Python versions**
   ```yaml
   strategy:
     matrix:
       python-version: ["3.9", "3.10", "3.11", "3.12"]
   ```

---

## Next Steps

After setup:

1. **Verify installation**
   ```bash
   python -m rm_abstract.verify
   ```

2. **Check plugins**
   ```bash
   rm-abstract list-plugins
   ```

3. **Run examples**
   ```bash
   python examples/simple_plugin_test.py
   ```

4. **Read documentation**
   - [QUICKSTART.md](QUICKSTART.md)
   - [INSTALL.md](INSTALL.md)
   - [README.md](README.md)
