# Quick Start Guide

Get started with LLM Heterogeneous Resource Orchestrator in 5 minutes!

---

## 1. Installation (30 seconds)

### Basic Installation

```bash
pip install rm-abstract
```

### With GPU Support

```bash
# Install PyTorch with CUDA first
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install rm-abstract with GPU support
pip install rm-abstract[gpu]
```

### From Source (for development)

```bash
git clone https://github.com/yourusername/rm_abstract_layer.git
cd rm_abstract_layer
pip install -e .
```

---

## 2. Verify Installation (30 seconds)

### Using CLI

```bash
# Verify installation
python -m rm_abstract.verify

# Show system info
python -m rm_abstract.cli info

# List available plugins
python -m rm_abstract.cli list-plugins
```

### Using Python

```python
import rm_abstract

# List available plugins
plugins = rm_abstract.list_plugins()
print(f"Available plugins: {list(plugins.keys())}")
```

---

## 3. First Program (1 minute)

Create `hello_orchestrator.py`:

```python
import rm_abstract

# Initialize with auto-selection
rm_abstract.init(device="auto", use_plugin_system=True)

print("RM Abstract initialized successfully!")
print("Using plugin:", rm_abstract.get_resource_manager().device_name)
```

Run it:

```bash
python hello_orchestrator.py
```

Expected output:
```
[RM Abstract] Initialized with plugin: cpu
RM Abstract initialized successfully!
Using plugin: cpu
```

---

## 4. Using Different Backends (2 minutes)

### CPU Backend (Always Available)

```python
import rm_abstract

# Use CPU backend
rm_abstract.init(device="cpu", use_plugin_system=True)

manager = rm_abstract.get_resource_manager()
print(f"Using: {manager.device_name}")
```

### GPU Backend (if available)

```python
import rm_abstract

# Use GPU backend
rm_abstract.init(device="gpu:0", use_plugin_system=True)

# Check capabilities
manager = rm_abstract.get_resource_manager()
caps = manager.get_capabilities()
print(f"Max batch size: {caps.get('max_batch_size')}")
```

### Auto Selection

```python
import rm_abstract

# Let orchestrator choose best backend
rm_abstract.init(device="auto", use_plugin_system=True)
```

---

## 5. Migration Example (2 minutes)

### Original GPU Code

```python
# Original GPU-only code
device = "cuda:0"
model = load_model().to(device)

def inference(prompt):
    return model.generate(prompt)
```

### Refactored for Orchestrator

```python
from rm_abstract.examples.gpu_to_npu_migration import (
    DeviceRuntime, GpuTorchRuntime, NpuRuntime, LLMApplication
)

# Choose runtime
runtime = GpuTorchRuntime(device="cuda:0")  # Or NpuRuntime() for NPU

# Application code (unchanged!)
app = LLMApplication(runtime=runtime)
app.setup("/path/to/model")
result = app.process_request("Hello, world!")
```

---

## 6. Run Examples (1 minute)

### Using CLI

```bash
# Run simple test
python -m rm_abstract.cli example simple

# Run migration demo
python -m rm_abstract.cli example migration

# Run full plugin demo
python -m rm_abstract.cli example plugin
```

### Direct Execution

```bash
# Simple plugin test
python examples/simple_plugin_test.py

# GPU to NPU migration
python examples/gpu_to_npu_migration.py

# Full plugin system demo
python examples/plugin_system_demo.py
```

---

## 7. Create Custom Plugin (5 minutes)

```python
from rm_abstract.core.plugin import Plugin, PluginMetadata, PluginPriority

class MyAcceleratorPlugin(Plugin):
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="my_accel",
            display_name="My Custom Accelerator",
            version="1.0.0",
            vendor="MyCompany",
            priority=PluginPriority.HIGH,
            device_types=["custom"],
            description="My custom accelerator backend"
        )

    def is_available(self) -> bool:
        # Check if hardware is available
        return True

    def initialize(self) -> None:
        print("Initializing my accelerator...")
        self._initialized = True

    def prepare_resource(self, resource, config=None):
        print(f"Preparing resource: {resource}")
        return resource

    def execute(self, resource, inputs, **kwargs):
        print(f"Executing on my accelerator: {inputs}")
        return f"Result from my accelerator: {inputs}"

    def cleanup(self) -> None:
        print("Cleaning up...")
        self._initialized = False

# Register and use
from rm_abstract.core.plugin import get_registry

get_registry().register(MyAcceleratorPlugin)

import rm_abstract
rm_abstract.init(device="my_accel", use_plugin_system=True)
```

---

## 8. Environment Variables

Set environment variables for configuration:

```bash
# Linux/Mac
export RM_DEVICE="auto"
export RM_USE_PLUGINS="true"
export RM_CACHE_DIR="/path/to/cache"

# Windows PowerShell
$env:RM_DEVICE="auto"
$env:RM_USE_PLUGINS="true"
$env:RM_CACHE_DIR="C:\cache"

# Then use without arguments
python -c "import rm_abstract; rm_abstract.init()"
```

---

## Common Commands Cheat Sheet

```bash
# Verification
python -m rm_abstract.verify              # Full verification
python -m rm_abstract.cli info            # System info
python -m rm_abstract.cli list-plugins    # List plugins

# Examples
python -m rm_abstract.cli example simple      # Simple test
python -m rm_abstract.cli example migration   # Migration demo

# Interactive
python -m rm_abstract.cli init            # Interactive setup

# Testing
python -m rm_abstract.cli test            # Run basic tests
```

---

## Next Steps

Now that you're up and running:

1. **Read Full Documentation**
   - [README.md](README.md) - Complete project overview
   - [PLUGIN_ARCHITECTURE.md](PLUGIN_ARCHITECTURE.md) - Plugin system details
   - [INSTALL.md](INSTALL.md) - Detailed installation guide

2. **Explore Examples**
   - [simple_plugin_test.py](examples/simple_plugin_test.py)
   - [gpu_to_npu_migration.py](examples/gpu_to_npu_migration.py)
   - [plugin_system_demo.py](examples/plugin_system_demo.py)

3. **Install Backend Support**
   ```bash
   pip install rm-abstract[gpu]           # GPU (vLLM)
   pip install rm-abstract[npu-rbln]      # Rebellions NPU
   pip install rm-abstract[npu-furiosa]   # FuriosaAI NPU
   pip install rm-abstract[all]           # Everything
   ```

4. **Join Community**
   - Report issues: https://github.com/yourusername/rm_abstract_layer/issues
   - Contribute: See [CONTRIBUTING.md](CONTRIBUTING.md)
   - Discuss: Join our Discord/Slack

---

## Troubleshooting

### "No module named 'rm_abstract'"

```bash
# Make sure you're in the right directory
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### "No plugins available"

```bash
# Check what's installed
python -m rm_abstract.cli list-plugins --all

# Install backend support
pip install rm-abstract[gpu]  # or [npu-rbln], [npu-furiosa]
```

### Need more help?

- Check [INSTALL.md](INSTALL.md#troubleshooting) for detailed troubleshooting
- Run: `python -m rm_abstract.verify` for diagnostics
- Enable debug logging:
  ```python
  import logging
  logging.basicConfig(level=logging.DEBUG)
  ```

---

Happy orchestrating! 🚀
