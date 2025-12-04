# RM Abstract Layer

Heterogeneous AI Accelerator Unified Compatibility Library - GPU/NPU Abstraction Layer

## Overview

An abstraction layer library that enables existing GPU inference scripts to run on NPU/GPU **without code modification**.

```
┌─────────────────────────────────────────────────────────────┐
│              Existing User Inference Code (unchanged)        │
├─────────────────────────────────────────────────────────────┤
│               RM Abstract Layer (add 1 line)                 │
│    import rm_abstract; rm_abstract.init(device="npu:0")     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┬─────────────┬─────────────┐                │
│  │  vLLM/CUDA  │  RBLN SDK   │ Furiosa SDK │                │
│  │    (GPU)    │ (Rebellions)│ (FuriosaAI) │                │
│  └─────────────┴─────────────┴─────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Zero Code Change**: Works by adding just 1 line to existing inference code
- **Transparent Compilation**: Automatically handles NPU compilation when needed (with caching)
- **Auto Device Selection**: Automatically selects optimal device with `device="auto"`
- **Plugin Architecture**: Extensible plugin system for any accelerator type
  - Easy to add new backends without modifying core code
  - Auto-discovery of available plugins
  - Priority-based auto-selection
  - Runtime plugin switching
- **Dual System Support**: Legacy Backend system + new Plugin system

## Supported Environments

| Device | Inference Engine | NPU Model |
|--------|------------------|-----------|
| GPU | vLLM | NVIDIA CUDA |
| NPU (Rebellions) | RBLN Runtime | ATOM |
| NPU (FuriosaAI) | Furiosa Runtime | RNGD |
| CPU | PyTorch | - |

## Installation

```bash
pip install rm-abstract
```

## Quick Start

### Basic Usage

```python
# Just add 1 line!
import rm_abstract
rm_abstract.init(device="npu:0")  # or "gpu:0", "auto"

# Use existing code as-is
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

### Environment Variable Configuration

```bash
export RM_DEVICE="npu:0"
export RM_CACHE_DIR="/data/npu_cache"
python your_inference_script.py
```

### Device Options

```python
import rm_abstract

# GPU (vLLM)
rm_abstract.init(device="gpu:0")

# Rebellions ATOM NPU
rm_abstract.init(device="rbln:0")

# FuriosaAI RNGD NPU
rm_abstract.init(device="furiosa:0")

# Auto selection
rm_abstract.init(device="auto")
```

### Compilation Options (NPU)

```python
import rm_abstract

rm_abstract.init(
    device="rbln:0",
    compile_options={
        "optimization_level": 3,
        "precision": "fp16",
    },
    cache_dir="~/.rm_abstract/cache",
    verbose=True  # Show compilation progress
)
```

## How It Works

```
rm_abstract.init(device="npu:0")
        │
        ▼
┌───────────────────────┐
│ DeviceFlowController  │
│ - Select backend      │
│ - Activate hooks      │
└───────────────────────┘
        │
        ▼
model = AutoModel.from_pretrained("llama")
        │
        ▼
┌───────────────────────┐
│ NPUBackend.prepare    │
│ - Check cache         │
│ - Convert to ONNX     │
│ - Compile for NPU     │
│ - Save to cache       │
└───────────────────────┘
        │
        ▼
model.generate(inputs)
        │
        ▼
┌───────────────────────┐
│ NPUBackend.execute    │
│ - Run on NPU runtime  │
└───────────────────────┘
```

## Plugin System (New!)

The library now features an extensible **plugin architecture** for managing any type of accelerator:

### List Available Plugins

```python
import rm_abstract

# List all available plugins
plugins = rm_abstract.list_plugins(available_only=True)
for name, info in plugins.items():
    print(f"{info['display_name']} - Priority: {info['priority']}")
```

### Using Plugin System

```python
import rm_abstract

# Enable plugin system (experimental)
rm_abstract.init(device="auto", use_plugin_system=True)

# Or via environment variable
# export RM_USE_PLUGINS=true
```

### Create Custom Plugin

```python
from rm_abstract.core.plugin import Plugin, PluginMetadata, PluginPriority

class MyAcceleratorPlugin(Plugin):
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="my_accel",
            display_name="My Custom Accelerator",
            version="1.0.0",
            priority=PluginPriority.HIGH,
            device_types=["custom"],
            description="Custom accelerator plugin"
        )

    def is_available(self) -> bool:
        # Check if hardware is available
        return True

    # Implement other methods...

# Register and use
from rm_abstract.core.plugin import get_registry
get_registry().register(MyAcceleratorPlugin)
```

For detailed information, see [PLUGIN_ARCHITECTURE.md](./PLUGIN_ARCHITECTURE.md).

## Project Structure

```text
rm_abstract_layer/
├── src/rm_abstract/
│   ├── __init__.py           # Main entry point
│   ├── core/
│   │   ├── backend.py        # Backend ABC (legacy)
│   │   ├── controller.py     # DeviceFlowController (legacy)
│   │   ├── config.py         # Configuration management
│   │   ├── plugin.py         # Plugin system (NEW!)
│   │   └── resource_manager.py  # ResourceManager (NEW!)
│   ├── backends/
│   │   ├── plugin_adapter.py # Backend→Plugin adapter (NEW!)
│   │   ├── auto_register.py  # Auto-registration (NEW!)
│   │   ├── gpu/              # vLLM backend
│   │   ├── npu/              # NPU backends
│   │   │   └── plugins/
│   │   │       ├── rebellions.py
│   │   │       └── furiosa.py
│   │   └── cpu/              # CPU backend
│   └── hooks/                # Auto-hooking system
├── tests/
├── examples/
│   └── plugin_system_demo.py # Plugin system demo (NEW!)
├── docs/
├── PLUGIN_ARCHITECTURE.md    # Plugin architecture guide (NEW!)
└── README.md
```

## Development Status

- [x] Phase 1: Project design and planning
- [x] Phase 2: Core infrastructure
- [ ] Phase 3: GPU backend (vLLM)
- [ ] Phase 4: NPU backends (Rebellions, FuriosaAI)
- [ ] Phase 5: Testing and validation
- [ ] Phase 6: Documentation and deployment

## License

MIT License

## Contributing

Contributions are welcome! Please refer to [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md) for the development roadmap.
