# Heterogeneous AI Accelerator Unified Compatibility Library (RM Abstract Layer)

## Project Overview

**Core Goal**: An abstraction layer that enables existing GPU inference scripts to run on NPU/GPU **without code modification**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Existing User Inference Code (unchanged)              │
│   model = AutoModelForCausalLM.from_pretrained("llama")                 │
│   output = model.generate(input_ids)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                    RM Abstract Layer (add 1 line)                        │
│   import rm_abstract; rm_abstract.init(device="npu:0")                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Device Flow Controller                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │  │
│  │  │  GPU Flow   │  │  NPU Flow   │  │  CPU Flow   │               │  │
│  │  │ (direct)    │  │(compile→run)│  │ (direct)    │               │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐              │
│  │  vLLM/CUDA  │ RBLN SDK    │ Furiosa SDK │  CPU Runtime│              │
│  │  (GPU)      │ (Rebellions)│ (FuriosaAI) │  (PyTorch)  │              │
│  └─────────────┴─────────────┴─────────────┴─────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Supported Environments

### Inference Engines
| Device | Inference Engine | Notes |
|--------|------------------|-------|
| **GPU** | vLLM | Continuous Batching, Tensor Parallel support |
| **NPU (Rebellions)** | RBLN Runtime / rbln-vllm | ATOM NPU inference |
| **NPU (FuriosaAI)** | Furiosa Runtime | RNGD NPU inference |
| **CPU** | PyTorch / ONNX Runtime | Fallback |

### NPU Vendors
| Vendor | NPU Model | SDK | Features |
|--------|-----------|-----|----------|
| **Rebellions** | ATOM | RBLN SDK | LLM optimized, vLLM compatible |
| **FuriosaAI** | RNGD | Furiosa SDK | LLM optimized, high-performance inference |

---

## Core Design Principles

### 1. Zero Code Change
```python
# Existing inference code (never modify this code)
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

```python
# With RM Abstract Layer (add only 1 line)
import rm_abstract
rm_abstract.init(device="npu:0")  # Add only this line!

# Rest of the code remains the same
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
# ... same as before
```

### 2. Device-Aware Flow
```
Device Configuration
    │
    ├─→ GPU detected → GPU Flow (direct execution, use PyTorch as-is)
    │
    ├─→ NPU detected → NPU Flow
    │       │
    │       ├─→ Compiled model cache exists? → Load cache → Execute
    │       │
    │       └─→ No cache → Convert to ONNX → NPU compile → Save cache → Execute
    │
    └─→ CPU detected → CPU Flow (PyTorch CPU execution)
```

### 3. Transparent Compilation
- NPU requires compilation, but users don't need to be aware of it
- Automatic compilation on first run, cache used afterwards
- Compilation progress shown via logging

---

## Phase 1: Core Infrastructure

### 1.1 Project Initial Setup
- [ ] Python package structure design
- [ ] pyproject.toml configuration
- [ ] Dependency management (requirements.txt)

### 1.2 Core Abstraction Interfaces
- [ ] `Backend` abstract base class
  ```python
  class Backend(ABC):
      @abstractmethod
      def is_available(self) -> bool: ...

      @abstractmethod
      def prepare_model(self, model) -> Any:
          """GPU: return as-is, NPU: return after compilation"""
          ...

      @abstractmethod
      def execute(self, prepared_model, inputs) -> Any: ...

      @abstractmethod
      def get_device_info(self) -> DeviceInfo: ...
  ```

- [ ] `DeviceFlowController` class
  ```python
  class DeviceFlowController:
      def __init__(self, device: str):
          self.backend = self._select_backend(device)

      def _select_backend(self, device: str) -> Backend:
          """Parse device string and return appropriate backend"""
          ...

      def intercept_model_call(self, model, method_name, *args, **kwargs):
          """Intercept model method calls and route to appropriate backend"""
          ...
  ```

- [ ] `ModelInterceptor` class (key component!)
  ```python
  class ModelInterceptor:
      """Transparently intercept PyTorch model's forward/generate etc."""

      def wrap(self, model):
          """Wrap model's key methods to be backend-aware"""
          original_forward = model.forward
          model.forward = self._create_intercepted_forward(original_forward)
          return model
  ```

### 1.3 Auto Hooking System
- [ ] `transformers` library monkey-patching
- [ ] `torch.nn.Module` level hooking
- [ ] Auto-detect and wrap models at load time

---

## Phase 2: GPU Backend (vLLM Integration)

### 2.1 GPU Backend Implementation (vLLM-based)
```python
class GPUBackend(Backend):
    """GPU Backend - utilizing vLLM inference engine"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.llm_engine = None

    def prepare_model(self, model_name_or_path: str, **kwargs):
        from vllm import LLM
        # Initialize vLLM engine
        self.llm_engine = LLM(
            model=model_name_or_path,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            dtype=kwargs.get("dtype", "auto"),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
        )
        return self.llm_engine

    def execute(self, prompts, sampling_params=None):
        from vllm import SamplingParams
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
        return self.llm_engine.generate(prompts, sampling_params)
```

### 2.2 vLLM Integration Options
```python
# vLLM-based GPU inference example
import rm_abstract

rm_abstract.init(
    device="gpu:0",
    inference_engine="vllm",  # Use vLLM
    engine_options={
        "tensor_parallel_size": 2,  # Multi-GPU
        "dtype": "float16",
        "gpu_memory_utilization": 0.85,
    }
)
```

- [ ] vLLM engine integration
- [ ] CUDA availability check
- [ ] Multi-GPU support (Tensor Parallel)
- [ ] Memory management utilities
- [ ] Continuous Batching utilization

---

## Phase 3: NPU Backend (Compilation Flow Included)

### 3.1 NPU Common Interface
```python
class NPUBackend(Backend, ABC):
    """NPU Backend common base - includes compilation flow"""

    def __init__(self, device_id: int, cache_dir: str):
        self.cache_dir = cache_dir
        self.compiled_models = {}

    def prepare_model(self, model):
        cache_key = self._get_cache_key(model)

        # 1. Check cache
        if self._cache_exists(cache_key):
            return self._load_from_cache(cache_key)

        # 2. Convert to ONNX
        onnx_model = self._convert_to_onnx(model)

        # 3. NPU compilation (vendor-specific implementation)
        compiled = self._compile_for_npu(onnx_model)

        # 4. Save to cache
        self._save_to_cache(cache_key, compiled)

        return compiled

    @abstractmethod
    def _compile_for_npu(self, onnx_model) -> Any:
        """Call vendor-specific NPU compiler"""
        ...

    @abstractmethod
    def _execute_on_npu(self, compiled_model, inputs) -> Any:
        """Execute on vendor-specific NPU runtime"""
        ...
```

### 3.2 NPU Compilation Pipeline
```
PyTorch Model
     │
     ▼
┌─────────────────┐
│  ONNX Export    │  torch.onnx.export()
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  ONNX Optimize  │  onnxoptimizer, onnx-simplifier
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  NPU Compiler   │  Vendor SDK (compile_model())
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Compiled Model │  .npu, .enf, .blob etc. vendor-specific format
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Model Cache    │  ~/.rm_abstract/cache/
└─────────────────┘
```

### 3.3 NPU Vendor-specific Backend Implementation

#### Supported NPU Vendors
| Vendor | SDK | Compiler | Runtime | Notes |
|--------|-----|----------|---------|-------|
| **Rebellions** | RBLN SDK | rbln-compiler | RBLN Runtime | ATOM NPU |
| **FuriosaAI** | Furiosa SDK | furiosa-compiler | furiosa-runtime | RNGD NPU |

#### Rebellions (RBLN) Backend
```python
class RBLNBackend(NPUBackend):
    """Rebellions ATOM NPU Backend"""

    def _compile_for_npu(self, onnx_model):
        import rebel
        # Compile model with RBLN compiler
        compiled = rebel.compile_from_onnx(
            onnx_model,
            target="atom",
            optimization_level=3
        )
        return compiled

    def _execute_on_npu(self, compiled_model, inputs):
        import rebel
        runtime = rebel.Runtime()
        return runtime.run(compiled_model, inputs)
```

#### FuriosaAI (RNGD) Backend
```python
class FuriosaBackend(NPUBackend):
    """FuriosaAI RNGD NPU Backend"""

    def _compile_for_npu(self, onnx_model):
        from furiosa import compiler
        # Compile model with Furiosa compiler
        compiled = compiler.compile(
            onnx_model,
            target="rngd",  # RNGD NPU target
            batch_size=1
        )
        return compiled

    def _execute_on_npu(self, compiled_model, inputs):
        from furiosa import runtime
        sess = runtime.create_runner(compiled_model)
        return sess.run(inputs)
```

- [ ] Rebellions RBLN SDK integration
- [ ] FuriosaAI SDK integration
- [ ] NPU vendor plugin interface definition
- [ ] Plugin auto-discovery and loading
- [ ] Graceful fallback when vendor SDK is unavailable

### 3.4 Compilation Cache System
- [ ] Model hash-based cache key generation
- [ ] Cache directory management (`~/.rm_abstract/cache/`)
- [ ] Cache invalidation policy (model change, SDK version change, etc.)
- [ ] Cache statistics and management CLI

---

## Phase 4: Transparent Integration API

### 4.1 Main Entry Point
```python
# rm_abstract/__init__.py

_global_controller = None

def init(device: str = "auto",
         cache_dir: str = None,
         compile_options: dict = None,
         verbose: bool = True):
    """
    Initialize RM Abstract Layer

    Args:
        device: "auto", "gpu:0", "npu:0", "cpu" etc.
        cache_dir: NPU compilation cache directory
        compile_options: NPU compilation options
        verbose: Show compilation progress
    """
    global _global_controller
    _global_controller = DeviceFlowController(
        device=device,
        cache_dir=cache_dir,
        compile_options=compile_options,
        verbose=verbose
    )

    # Auto-hook Transformers
    _hook_transformers()

    # Auto-hook PyTorch modules
    _hook_pytorch_modules()
```

### 4.2 Auto Hooking System
```python
def _hook_transformers():
    """Auto-patch Hugging Face Transformers library"""
    try:
        import transformers

        original_from_pretrained = transformers.PreTrainedModel.from_pretrained

        @wraps(original_from_pretrained)
        def patched_from_pretrained(cls, *args, **kwargs):
            model = original_from_pretrained.__func__(cls, *args, **kwargs)
            return _global_controller.prepare_model(model)

        transformers.PreTrainedModel.from_pretrained = classmethod(patched_from_pretrained)
    except ImportError:
        pass  # Ignore if transformers not installed

def _hook_pytorch_modules():
    """Hook PyTorch nn.Module"""
    import torch.nn as nn

    original_call = nn.Module.__call__

    @wraps(original_call)
    def patched_call(self, *args, **kwargs):
        if _global_controller and _global_controller.should_intercept(self):
            return _global_controller.execute(self, *args, **kwargs)
        return original_call(self, *args, **kwargs)

    nn.Module.__call__ = patched_call
```

### 4.3 Usage Examples

#### Example 1: Basic Usage (simplest)
```python
import rm_abstract
rm_abstract.init(device="npu:0")

# Below code is 100% identical to original
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
result = generator("Hello, I'm a language model")
```

#### Example 2: Environment Variable Configuration
```bash
export RM_DEVICE="npu:0"
export RM_CACHE_DIR="/data/npu_cache"
python existing_inference_script.py  # No code modification!
```

```python
# existing_inference_script.py (no modification, just add rm_abstract import)
import rm_abstract  # Auto-recognize environment variables
rm_abstract.init()  # Load settings from environment variables

from transformers import AutoModelForCausalLM
# ... existing code as-is
```

#### Example 3: Explicit Compilation Control
```python
import rm_abstract

# Detailed NPU compilation options
rm_abstract.init(
    device="npu:0",
    compile_options={
        "optimization_level": 3,
        "precision": "fp16",
        "batch_size": [1, 4, 8],  # Dynamic batch
    },
    verbose=True  # Show compilation progress
)

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("llama-7b")
# First run: "Compiling model for NPU... [====>    ] 45%"
# Subsequent runs: "Loading compiled model from cache..."
```

#### Example 4: Device Switching
```python
import rm_abstract

# Start with GPU
rm_abstract.init(device="gpu:0")
# ... run inference ...

# Switch to NPU
rm_abstract.switch_device("npu:0")
# ... run inference on NPU with same code ...
```

---

## Phase 5: Testing and Validation

### 5.1 Unit Tests
- [ ] Each backend basic functionality
- [ ] Hooking system proper operation
- [ ] Cache system

### 5.2 Integration Tests
- [ ] Real model tests (GPT-2, BERT, LLaMA)
- [ ] GPU ↔ NPU result consistency verification
- [ ] Various inference scenarios

### 5.3 Performance Benchmarks
- [ ] Abstraction overhead measurement
- [ ] Compilation time measurement
- [ ] Cache hit rate analysis

---

## Phase 6: Documentation and Deployment

### 6.1 Documentation
- [ ] Quick Start guide
- [ ] NPU vendor plugin development guide
- [ ] API reference
- [ ] FAQ / Troubleshooting

### 6.2 Deployment
- [ ] PyPI package
- [ ] Docker image (with NPU SDK)
- [ ] CI/CD pipeline

---

## Project Structure

```
rm_abstract_layer/
├── src/
│   └── rm_abstract/
│       ├── __init__.py              # Main entry point (init, switch_device)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── backend.py           # Backend ABC
│       │   ├── controller.py        # DeviceFlowController
│       │   ├── interceptor.py       # ModelInterceptor
│       │   └── config.py            # Configuration management
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── registry.py          # Backend plugin registry
│       │   ├── gpu/
│       │   │   ├── __init__.py
│       │   │   └── cuda_backend.py  # GPU Passthrough
│       │   ├── npu/
│       │   │   ├── __init__.py
│       │   │   ├── base.py          # NPU common (compilation flow)
│       │   │   ├── compiler.py      # Compilation pipeline
│       │   │   ├── cache.py         # Compilation cache management
│       │   │   └── plugins/         # Vendor-specific plugins
│       │   │       ├── __init__.py
│       │   │       ├── rebellions.py    # Rebellions ATOM
│       │   │       └── furiosa.py       # FuriosaAI RNGD
│       │   └── cpu/
│       │       ├── __init__.py
│       │       └── cpu_backend.py
│       ├── hooks/
│       │   ├── __init__.py
│       │   ├── transformers_hook.py # HF Transformers hooking
│       │   └── pytorch_hook.py      # PyTorch module hooking
│       ├── conversion/
│       │   ├── __init__.py
│       │   └── onnx_utils.py        # ONNX conversion utilities
│       └── utils/
│           ├── __init__.py
│           ├── logger.py
│           └── device_utils.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── examples/
│   ├── basic_usage.py
│   ├── npu_compilation.py
│   └── device_switching.py
├── docs/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Core Flow Diagram

```
User Code Execution
       │
       ▼
rm_abstract.init(device="npu:0")
       │
       ▼
┌─────────────────────────────┐
│   DeviceFlowController      │
│   - Select backend (NPU)    │
│   - Activate hooking system │
└─────────────────────────────┘
       │
       ▼
model = AutoModel.from_pretrained("llama")
       │
       ▼
┌─────────────────────────────┐
│   Transformers Hook Fired   │
│   - Intercept from_pretrained│
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   NPUBackend.prepare_model  │
│   ┌───────────────────────┐ │
│   │ Check cache           │ │
│   │   ├─ Hit → Load       │ │
│   │   └─ Miss → Compile   │ │
│   │       ├─ ONNX convert │ │
│   │       ├─ NPU compile  │ │
│   │       └─ Save cache   │ │
│   └───────────────────────┘ │
└─────────────────────────────┘
       │
       ▼
model.generate(inputs)
       │
       ▼
┌─────────────────────────────┐
│   PyTorch Hook Fired        │
│   - Intercept __call__      │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   NPUBackend.execute        │
│   - Run on NPU runtime      │
│   - Return result           │
└─────────────────────────────┘
       │
       ▼
output (returned to user)
```

---

## Next Steps

1. **Start Phase 1**: Create project structure and implement core interfaces
2. Implement Backend ABC, DeviceFlowController, ModelInterceptor
3. Implement GPU backend (Passthrough) to verify basic operation
4. Implement NPU compilation flow and cache system

---

*Document created: 2025-12-02*
*Version: 0.2.0-draft*
