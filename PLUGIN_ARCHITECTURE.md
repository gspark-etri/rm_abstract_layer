# Plugin Architecture

RM Abstract Layer now features a powerful plugin-based architecture for extensible resource management. This document explains the design, usage, and benefits of the new plugin system.

## Overview

The plugin system provides a **flexible, extensible framework** for integrating any type of backend (GPU, NPU, custom accelerators) without modifying core code. It follows the **Open/Closed Principle**: open for extension, closed for modification.

```
┌─────────────────────────────────────────────────────┐
│         User Application (Unchanged)                │
├─────────────────────────────────────────────────────┤
│         RM Abstract Layer API                       │
├─────────────────────────────────────────────────────┤
│         ResourceManager                             │
│  ┌───────────────────────────────────────────┐     │
│  │       PluginRegistry                      │     │
│  │  - Auto-discovery                         │     │
│  │  - Dynamic registration                   │     │
│  │  - Priority-based selection               │     │
│  └───────────────────────────────────────────┘     │
├─────────────────────────────────────────────────────┤
│       Plugin Ecosystem                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  GPU     │  │  NPU     │  │ Custom   │  ...     │
│  │ Plugin   │  │ Plugin   │  │ Plugin   │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────┘
```

## Key Components

### 1. Plugin Base Class

All plugins inherit from the `Plugin` abstract base class:

```python
from rm_abstract.core.plugin import Plugin, PluginMetadata, PluginPriority

class MyPlugin(Plugin):
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="my_plugin",
            display_name="My Custom Plugin",
            version="1.0.0",
            vendor="MyCompany",
            priority=PluginPriority.HIGH,
            device_types=["custom"],
            requires=["torch"],
            description="My custom accelerator plugin"
        )

    def is_available(self) -> bool:
        # Check if hardware/drivers are available
        return True

    def initialize(self) -> None:
        # Initialize device, runtime, etc.
        self._initialized = True

    def prepare_resource(self, resource, config=None):
        # Prepare/compile resource for device
        return resource

    def execute(self, resource, inputs, **kwargs):
        # Execute inference
        return result

    def cleanup(self) -> None:
        # Cleanup resources
        self._initialized = False
```

### 2. PluginRegistry

The `PluginRegistry` manages plugin discovery and instantiation:

- **Auto-discovery**: Automatically finds plugins in packages
- **Manual registration**: Register plugins explicitly
- **Priority-based selection**: Auto-select best plugin by priority
- **Lazy loading**: Plugins only loaded when needed

### 3. ResourceManager

The `ResourceManager` provides a unified interface for resource management:

```python
from rm_abstract.core.resource_manager import ResourceManager

manager = ResourceManager(config)
manager.initialize(auto_discover=True)

# Prepare resource (model, data, etc.)
prepared = manager.prepare_resource(model, config={"precision": "fp16"})

# Execute
result = manager.execute(prepared, inputs)

# Switch plugins at runtime
manager.switch_plugin("gpu", device_id=0)
```

## Plugin Metadata

Plugins declare their capabilities through metadata:

```python
@dataclass
class PluginMetadata:
    name: str                    # Unique identifier
    display_name: str            # Human-readable name
    version: str                 # Plugin version
    vendor: Optional[str]        # Vendor/author
    priority: PluginPriority     # Auto-selection priority
    device_types: List[str]      # ["gpu", "npu", "custom", ...]
    requires: List[str]          # Required Python packages
    description: str             # Plugin description
    extra: Dict[str, Any]        # Additional metadata
```

### Priority Levels

Plugins have priority levels for auto-selection:

- **HIGHEST (100)**: NPU accelerators (Rebellions, FuriosaAI)
- **HIGH (75)**: GPU accelerators (vLLM)
- **NORMAL (50)**: Standard backends
- **LOW (25)**: Experimental backends
- **LOWEST (0)**: Fallback backends (CPU)

## Usage Examples

### Basic Usage (Legacy System)

```python
import rm_abstract

# Use existing Backend system (default)
rm_abstract.init(device="gpu:0")

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
# Model runs on GPU via vLLM
```

### Using Plugin System

```python
import rm_abstract

# Enable new plugin system
rm_abstract.init(device="auto", use_plugin_system=True)

# Or via environment variable
# export RM_USE_PLUGINS=true
```

### List Available Plugins

```python
import rm_abstract

plugins = rm_abstract.list_plugins(available_only=True)
for name, info in plugins.items():
    print(f"{name}: {info['display_name']} (priority={info['priority']})")
```

### Create Custom Plugin

```python
from rm_abstract.core.plugin import Plugin, PluginMetadata, PluginPriority, get_registry

class TPUPlugin(Plugin):
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="tpu",
            display_name="Google TPU",
            version="1.0.0",
            vendor="Google",
            priority=PluginPriority.HIGHEST,
            device_types=["tpu"],
            requires=["jax"],
            description="Google TPU backend"
        )

    def is_available(self) -> bool:
        try:
            import jax
            return jax.devices("tpu") is not None
        except:
            return False

    # ... implement other methods ...

# Register plugin
get_registry().register(TPUPlugin)

# Use it
rm_abstract.init(device="tpu:0", use_plugin_system=True)
```

### Switch Plugins at Runtime

```python
import rm_abstract

# Start with CPU
rm_abstract.init(device="cpu", use_plugin_system=True)

manager = rm_abstract.get_resource_manager()

# Switch to GPU when available
manager.switch_plugin("gpu", device_id=0)

# Switch to NPU
manager.switch_plugin("rbln", device_id=0)
```

## Plugin Lifecycle

```
┌──────────────┐
│  Register    │  ← Plugin added to registry
└──────┬───────┘
       │
┌──────▼───────┐
│ is_available │  ← Check if plugin can run on system
└──────┬───────┘
       │
┌──────▼───────┐
│ initialize   │  ← Setup device, runtime, etc.
└──────┬───────┘
       │
    ┌──▼──────────────────────────────┐
    │                                  │
┌───▼─────────────┐      ┌────────────▼──┐
│ prepare_resource│ ←──→ │    execute    │  (repeat)
└───┬─────────────┘      └────────────┬──┘
    │                                  │
    └──────────────┬───────────────────┘
                   │
            ┌──────▼───────┐
            │   cleanup    │  ← Release resources
            └──────────────┘
```

## Migration from Backend to Plugin

Existing `Backend` implementations can be automatically adapted to plugins:

```python
from rm_abstract.backends.plugin_adapter import create_backend_plugin
from rm_abstract.core.plugin import PluginPriority

# Wrap existing Backend as Plugin
gpu_plugin = create_backend_plugin(
    backend_class=VLLMBackend,
    name="gpu",
    display_name="vLLM GPU Backend",
    version="1.0.0",
    priority=PluginPriority.HIGH,
    requires=["torch", "vllm"]
)

# Register it
get_registry().register(gpu_plugin)
```

This allows **gradual migration** without breaking existing code.

## Benefits

### 1. **Extensibility**
- Add new backends without modifying core code
- Plugin discovery finds backends automatically
- Support any type of accelerator

### 2. **Flexibility**
- Switch backends at runtime
- Mix and match plugins
- Custom plugins for specific use cases

### 3. **Maintainability**
- Clean separation of concerns
- Plugin-specific code isolated
- Easy to test individual plugins

### 4. **User Experience**
- Auto-selection picks best available backend
- Zero code changes for users
- Gradual migration path from old system

### 5. **Ecosystem**
- Third parties can create plugins
- Community-driven backend support
- Vendor-provided official plugins

## Architecture Patterns

### Plugin Adapter Pattern

Wraps existing `Backend` implementations as `Plugin`:

```python
class BackendPluginAdapter(Plugin):
    """Adapter to wrap Backend as Plugin"""

    def __init__(self, backend_class, metadata, **kwargs):
        self._backend_class = backend_class
        self._metadata = metadata
        self._backend = None

    def prepare_resource(self, resource, config=None):
        return self._backend.prepare_model(resource, config)
```

### Registry Pattern

Central registry for plugin management:

```python
class PluginRegistry:
    def register(self, plugin_class):
        """Register plugin"""

    def get_plugin(self, name, **kwargs):
        """Get or create plugin instance"""

    def auto_select(self, device_type=None):
        """Auto-select best plugin"""

    def discover_plugins(self, package_path):
        """Auto-discover plugins in package"""
```

### Resource Manager Pattern

Unified interface for resource management:

```python
class ResourceManager:
    def __init__(self, config):
        self.registry = get_registry()
        self._active_plugin = None

    def prepare_resource(self, resource, config):
        """Prepare using active plugin"""
        return self._active_plugin.prepare_resource(resource, config)
```

## Future Enhancements

1. **Plugin Marketplace**: Central repository for community plugins
2. **Plugin Versioning**: Semantic versioning and compatibility checks
3. **Hot Reload**: Load/unload plugins without restart
4. **Plugin Chains**: Compose multiple plugins (preprocessing + inference + postprocessing)
5. **Resource Sharing**: Share resources across plugins
6. **Performance Profiling**: Built-in profiling per plugin
7. **Plugin Dependencies**: Declare dependencies between plugins

## Best Practices

### For Plugin Developers

1. **Implement all abstract methods** in Plugin base class
2. **Provide accurate metadata** for auto-discovery
3. **Check requirements** in `is_available()`
4. **Handle errors gracefully** with informative messages
5. **Cleanup resources** in `cleanup()` method
6. **Document capabilities** via `get_capabilities()`
7. **Test on target hardware** before releasing

### For Users

1. **Use `auto` device** for automatic plugin selection
2. **Check available plugins** with `list_plugins()`
3. **Handle plugin unavailability** gracefully
4. **Use environment variables** for configuration
5. **Enable verbose mode** for debugging
6. **Test plugin switching** before production use

## Comparison: Backend vs Plugin System

| Feature | Legacy Backend | Plugin System |
|---------|---------------|---------------|
| Extensibility | Hard-coded registration | Auto-discovery |
| Runtime switching | Limited | Full support |
| Third-party support | Manual integration | Automatic |
| Priority selection | Manual order | Metadata-driven |
| Resource management | Per-backend | Unified |
| Migration path | Breaking changes | Gradual |
| Testing | Per-backend | Isolated plugins |

## Conclusion

The plugin architecture transforms RM Abstract Layer into a **truly extensible platform** for heterogeneous accelerator support. It provides:

- ✅ **Easy integration** of new backends
- ✅ **Zero-code changes** for end users
- ✅ **Flexible resource management**
- ✅ **Community-driven ecosystem**

The system maintains **full backward compatibility** with the legacy Backend system while providing a clear migration path to the new plugin architecture.
