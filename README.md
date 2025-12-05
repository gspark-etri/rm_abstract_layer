# LLM Heterogeneous Resource Orchestrator

> **GPU / NPU / PIM and other heterogeneous accelerators managed through a unified meta-serving layer**

---

## 1. Project Overview

The landscape of Large Language Model (LLM) inference and training is no longer defined solely by GPUs.

* General-purpose **GPU**
* Vendor-specific **NPU** with proprietary compilation and runtime
* **PIM (Processing-In-Memory)** for in-memory computation
* Remote LLM services and dedicated servers (vLLM, DeepSpeed-Inference, TGI, TensorRT-LLM, etc.)

Each resource has:

* Different supported operation sets
* Additional requirements like **pre-compilation / graph transformation / kernel tuning**
* Its own software stack and runtime ecosystem

Especially for **NPU**, the following characteristics are common:

* Engine / Runtime / Compiler are often provided as **binary (closed-source)** only
* Integration is only possible through CLI tools, config files, and proprietary runtime libraries
* Internal implementation is opaque, exposing only limited APIs and options

**LLM Heterogeneous Resource Orchestrator** (hereafter "Orchestrator") addresses this environment by:

* **Minimizing changes** to existing GPU-based inference code
* **Naturally integrating NPU / PIM / new accelerators** as plugins
* **Unifying management** of pre-compilation, build, and runtime configuration requirements for each resource
* **Wrapping closed-source binary-only NPU engines/compilers** as black boxes through CLI/C API
* Providing applications with a **single LLM API** abstraction

This is a **meta-serving / resource abstraction layer** project.

---

## 2. Philosophy and Design Goals

### 2.1 Philosophy

1. **Resource Type Neutral (GPU / NPU / PIM Neutral)**

   * Focus on "what computational capabilities and constraints are provided" rather than "what chip is it"
   * Treat GPU / NPU / PIM / CPU / Remote all as the same concept: `Resource`

2. **Software Stack Agnostic**

   * CUDA / ROCm / NPU SDK / PIM libraries / ONNX Runtime / TensorRT / vLLM / DeepSpeed...
     → All wrapped with the same `BackendPlugin` interface

3. **Pre-compilation/Build Requirements as First-Class Citizens**

   * Model "pre-compilation", "graph transformation", "kernel auto-tuning" required by NPU/PIM
     are modeled as **essential characteristics of resources**, not incidental runtime tasks

4. **Binary/Black-box Stack Friendly**

   * Even when compiler/engine is provided only as binary without exposing code/IR,
     integration is possible using only CLI, config files, and limited C APIs
     → **Black-box assumption built into the design**

5. **Minimal Invasiveness**

   * Existing GPU-only code can be **partially refactored** to work on NPU
   * No need to completely rewrite existing code to add NPU/PIM support

6. **Incremental Heterogenization**

   * Stage 1: GPU only
   * Stage 2: GPU + NPU
   * Stage 3: GPU + NPU + PIM + Remote LLM
     → Progressive expansion within the same architecture

### 2.2 Design Goals

* **Single Logical Model Endpoint**

  * A single logical name like `model = "chat-main"` works regardless of whether it runs on GPU/NPU/PIM internally

* **Backend Plugin-based Extensibility**

  * Adding a new accelerator and its dependent runtime/compilation stack only requires adding one `BackendPlugin`

* **Unified Resource + Build Characteristics Management**

  * Manage not only "memory/compute capability" but also
    "pre-compilation requirements, compilation pipeline, generated artifacts" together

* **Binary-only NPU Stack as Plugin**

  * Even when compiler/engine is provided only as binary CLI or C API,
    it naturally fits into Orchestrator's build/plugin interface

* **Migration Path from GPU Code to NPU/PIM**

  * Centralize device-specific parts,
    and simply swap GPU·NPU·PIM implementations

---

## 3. Architecture: Position of Our Layer

### 3.1 LLM Stack Layering

Simplified LLM stack layers:

```text
+-----------------------------------------------------+
|                  Application Layer                  |
|  - Web / gRPC API                                   |
|  - RAG, Agents (LangChain, LlamaIndex, etc.)        |
+--------------------------▲--------------------------+
                           │  (Single LLM API)
                           │
+--------------------------┴--------------------------+
|  Heterogeneous Resource Orchestrator  (THIS PROJECT)|
|  - Request routing & policy engine                  |
|  - Resource abstraction (GPU / NPU / PIM / REMOTE)  |
|  - Backend plugin registry                          |
|  - Compile / build artifact management              |
+-----------▲----------------------------▲------------+
            │                            │
            │ Backend Plugin Interface   │ Backend Plugin Interface
            │                            │
+-----------┴------------+   +-----------┴------------+
|   GPU Backend Plugins  |   |   NPU Backend Plugins  |
|  - vLLM on CUDA        |   |  - Vendor NPU SDK      |
|  - DeepSpeed           |   |  - NPU + ONNXRuntime   |
|  - TensorRT-LLM        |   |  - Binary-only engine  |
+------------------------+   +------------------------+
+------------------------+   +------------------------+
|   PIM Backend Plugins  |   | Remote LLM Plugins     |
|  - PIM GEMM runtime    |   |  - OpenAI-like HTTP    |
|  - In-memory kernels   |   |  - vLLM/TGI server     |
+------------------------+   +------------------------+

(Actual frameworks / SDKs / services / devices sit below each plugin)
```

Our project sits **between applications and hardware-specific runtimes**:

* Unified abstraction of resource/backend information
* Decision on which combination (GPU/NPU/PIM) to use per request
* Management of compilation/build pipeline and binary stack interfaces required by each resource

---

## 4. Resource Model: GPU / NPU / PIM Focus

### 4.1 Resource

```python
class ResourceType(Enum):
    GPU = "gpu"
    NPU = "npu"
    PIM = "pim"
    CPU = "cpu"
    REMOTE = "remote"

class Resource:
    id: str
    type: ResourceType
    attributes: Dict[str, Any]   # memory_gb, bandwidth, vendor, device_id, etc.
    tags: List[str]              # "low-latency", "low-power", "experimental"
```

### 4.2 Capability (Computational Perspective)

```python
class Capability:
    max_batch_size: int
    max_seq_len: int
    support_kv_cache: bool
    support_streaming: bool
    dtype: List[str]             # "fp16", "bf16", "int8", "fp8", ...
    optimized_for: List[str]     # "throughput", "latency", "energy", ...
```

* GPU: High versatility + fp16/bf16 throughput
* NPU: Very high efficiency for specific patterns (e.g., matmul-heavy transformers)
* PIM: Strong in **memory-intensive operations** (Attention, some GEMV/GEMM)

Orchestrator can make policy decisions:

* Long context + large batch → GPU / NPU
* Memory bandwidth critical → PIM
* Single short request + ultra-low latency → Pre-compiled NPU engine

---

## 5. Build / Pre-compilation Model (GPU / NPU / PIM + Binary Stack)

### 5.1 BuildProfile & Artifact

NPU / PIM typically require **additional steps**:

* Convert model to **ONNX or intermediate representation (IR)**
* **Pre-compile** with NPU/PIM compiler
* Generate **binary/graph/engine files** for runtime loading
* Tightly coupled with specific **SDK version, driver version**

Orchestrator includes build concepts:

```python
class BuildProfile:
    target_resource_type: ResourceType       # GPU / NPU / PIM
    compiler: str                            # "tensorrt", "vendor_npu_compiler", ...
    compiler_version: str
    flags: Dict[str, Any]                    # optimization level, precision, etc.
    interface: str                           # "cli", "c-api", "config-file", etc.

class BuildArtifact:
    id: str
    model_id: str
    build_profile: BuildProfile
    path: str                                # local path or remote URI
    metadata: Dict[str, Any]                 # checksum, created_at, etc.
```

### 5.2 Build Adapters for Binary-only NPU Stacks

Closed-source NPU stacks typically operate as:

* **Model conversion CLI**: `npu_compile --input model.onnx --output engine.bin --config config.yaml`
* **Runtime library**: Load/execute `engine.bin` through C API of `libnpu_runtime.so`
* Internal IR/code is invisible, only config files and limited APIs are modifiable

Orchestrator accommodates this through **external tool wrapping**, not source-level integration:

* `BinaryCompilerAdapter`

  * When `BuildProfile.interface == "cli"`:
    * Define CLI command template, environment variables, config file format per plugin
    * Orchestrator's build pipeline calls this to generate `BuildArtifact`

* `BinaryRuntimeAdapter`

  * When `BuildProfile.interface == "c-api"`:
    * Write minimal C FFI wrapper matching provided headers/so files
    * Orchestrator's `BackendSession` only calls this wrapper

These adapters **wrap black-box CLI & C API** to fit upper abstraction layers.

> The key point: Orchestrator doesn't need to know NPU internal implementation details,
> only *build inputs (ONNX, etc.), build commands/options, build outputs (engine files), runtime API signatures*.

---

## 6. Backend Plugin Design: GPU / NPU / PIM + Binary Stack

### 6.1 Common Interface (Summary)

```python
class BackendPlugin(Protocol):
    name: str  # "gpu_vllm", "npu_vendorA", "pim_xxx", ...

    def probe(self, resources: List[Resource]) -> List[Resource]:
        """Select usable resources from available ones"""
        ...

    def required_build_profiles(self, config: BackendConfig) -> List[BuildProfile]:
        """Build profiles required by this backend"""
        ...

    def create_session(
        self,
        resources: List[Resource],
        config: BackendConfig,
        artifacts: List[BuildArtifact],
    ) -> BackendSession:
        """Create session assuming required artifacts are ready"""
        ...

    def get_capability(
        self,
        resources: List[Resource],
        config: BackendConfig,
        artifacts: List[BuildArtifact],
    ) -> Capability:
        ...
```

### 6.2 Binary-only NPU Backend Plugin Example

```python
class VendorNpuBackendPlugin(BackendPlugin):
    name = "npu_vendorA"

    def probe(self, resources):
        return [r for r in resources
                if r.type == ResourceType.NPU
                and r.attributes.get("vendor") == "VendorA"]

    def required_build_profiles(self, config):
        return [
            BuildProfile(
                target_resource_type=ResourceType.NPU,
                compiler="vendorA_npu_compiler",
                compiler_version=config.get("compiler_version", "1.0"),
                flags={
                    "precision": config.get("precision", "fp16"),
                    "max_seq_len": config.get("max_seq_len", 4096),
                },
                interface="cli",
            )
        ]

    def create_session(self, resources, config, artifacts):
        artifact = select_artifact_for_profile(artifacts)
        runtime = BinaryRuntimeAdapter(
            engine_path=artifact.path,
            npu_device_id=resources[0].attributes["device_id"],
            runtime_so=config.get("runtime_so", "libvendorA_npu_runtime.so"),
        )
        return NpuBackendSession(runtime)
```

This way, even when compiler/runtime are closed-source with only CLI/C API,
integration is possible through `BackendPlugin` + `BinaryCompilerAdapter` + `BinaryRuntimeAdapter`.

---

## 7. Primary Goal: "Minor GPU Code Changes to Run on NPU"

### 7.1 Current Situation Assumption

Existing GPU inference code looks like:

```python
# Existing: GPU-only code
device = "cuda:0"
model = load_model(...).to(device)
tokenizer = load_tokenizer(...)

def run_inference(prompt: str, max_tokens: int) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_new_tokens=max_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

### 7.2 Minimal Refactoring: DeviceRuntime Abstraction

To centralize GPU/NPU/PIM differences,
introduce `DeviceRuntime` concept:

```python
class DeviceRuntime(Protocol):
    def load_model(self, model_path: str): ...
    def generate(self, prompt: str, max_tokens: int) -> str: ...

class GpuTorchRuntime(DeviceRuntime):
    def __init__(self, device: str):
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self, model_path: str):
        self.model = load_model(model_path).to(self.device)
        self.tokenizer = load_tokenizer(model_path)

    def generate(self, prompt: str, max_tokens: int) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids, max_new_tokens=max_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
```

Now existing code becomes:

```python
runtime = GpuTorchRuntime(device="cuda:0")
runtime.load_model("/path/to/model")

def run_inference(prompt, max_tokens):
    return runtime.generate(prompt, max_tokens)
```

(Core logic remains almost the same)

### 7.3 Binary-only NPU Implementation

For NPU, typically need:

* ONNX export or proprietary IR conversion
* NPU compiler pre-compilation (CLI)
* NPU runtime API calls (C API)

Implement as `NpuRuntime(DeviceRuntime)`:

```python
class NpuRuntime(DeviceRuntime):
    def __init__(self, npu_device_id: int, artifact: BuildArtifact, runtime_so: str):
        self.npu_device_id = npu_device_id
        self.artifact = artifact   # NPU-compiled model binary path
        self.runtime = BinaryRuntimeAdapter(
            engine_path=artifact.path,
            npu_device_id=npu_device_id,
            runtime_so=runtime_so,
        )

    def load_model(self, model_path: str):
        # Actual execution uses artifact (engine file)
        self.runtime.load()

    def generate(self, prompt: str, max_tokens: int) -> str:
        input_ids = common_tokenizer(prompt)
        output_ids = self.runtime.run(input_ids, max_tokens=max_tokens)
        return common_detokenizer(output_ids)
```

Orchestrator's `VendorNpuBackendPlugin`:

1. Finds NPU Resource + BuildArtifact
2. Creates `NpuRuntime`
3. Wraps it as `BackendSession` for upper layers

This allows **extending without major GPU code restructuring**:

* GPU: `GpuTorchRuntime`
* NPU: `NpuRuntime`

Just swap the two runtimes.

---

## 8. Adding New Resources & Software Stacks

When new accelerator and dependent stack (compiler, runtime) are added:

1. **Define Resource type/attributes**

   * New NPU/PIM generation or "Hybrid GPU+PIM" → Express with attributes/tags

2. **Implement Backend Plugin**

   * Add one `NewAccelBackendPlugin` class:
     * `probe()`: Select matching devices
     * `required_build_profiles()`: Define compilation pipeline
     * `create_session()`: Initialize SDK/runtime
     * `get_capability()`: Describe performance/constraints

3. **(Optional) Define BinaryCompilerAdapter / BinaryRuntimeAdapter**

   * If only CLI/C API patterns differ, reuse or inherit existing adapters

4. **Policy-level Integration**

   * In Orchestrator's policy engine:
     * When to prioritize new accelerator
     * Add scoring logic for advantages vs existing GPU/NPU/PIM

**New hardware + proprietary software stack** integration:

* Existing application code unchanged
* Changes only in Orchestrator and plugin/adapter layers
* Managed consistently from resource/build/policy perspectives

---

## 9. Current Implementation Status

### Implemented

✅ Plugin-based architecture
✅ Resource model (Resource, Capability, BuildProfile, BuildArtifact)
✅ Binary adapters (BinaryCompilerAdapter, BinaryRuntimeAdapter)
✅ Auto-discovery and registration system
✅ Priority-based auto-selection
✅ **GPU Backend** - vLLM high-performance inference
✅ **CPU Backend** - PyTorch fallback
✅ **NPU Backend** - Rebellions ATOM (vLLM-RBLN + Optimum-RBLN dual mode)
✅ **Multi-Serving Engine Support** - vLLM, Triton, TorchServe
✅ **System Information** - Hardware/software discovery
✅ **Runtime Device Switching** - GPU ↔ CPU ↔ NPU

### In Progress

🔄 FuriosaAI NPU backend
🔄 Policy engine for resource selection
🔄 PIM backend support

### Planned

📋 Remote LLM backend support
📋 Multi-resource orchestration (GPU + NPU + PIM)
📋 Advanced build pipeline with optimization
📋 Ray Serve integration

---

## 10. Installation & Quick Start

### Installation

```bash
# Basic installation
pip install rm-abstract

# With GPU support
pip install rm-abstract[gpu]

# Development installation
git clone https://github.com/gspark-etri/rm_abstract_layer.git
cd rm_abstract_layer
pip install -e ".[dev]"
```

### System Information

Check available resources, backends, and serving engines:

```bash
# CLI
python -m rm_abstract.system_info
```

```python
# Python API
import rm_abstract

# Print formatted system info
rm_abstract.print_system_info()

# Get detailed info
info = rm_abstract.get_system_info()
print(f"GPUs: {len(info.gpus)}")
print(f"NPUs: {len(info.npus)}")
print(f"Available backends: {[b.name for b in info.backends if b.available]}")
```

Output example:
```
======================================================================
  RM Abstract Layer - System Information
======================================================================

🎮 GPUs (8 detected)
   ✓ [0] NVIDIA GeForce RTX 3090 - 24 GB

🔮 NPUs (0 detected)
   ✗ No NPUs detected

⚙️  Backends
   ✓ vLLM GPU Backend (0.12.0)
   ✓ PyTorch CPU Backend
   ✓ Rebellions ATOM NPU

🚀 Serving Engines
   ✓ vLLM
   ✓ Triton Inference Server
   ✓ TorchServe

📊 Summary
   Hardware: 8 GPU(s), 0 NPU(s)
   Backends: 3/4 available
   Serving Engines: 3/3 available
======================================================================
```

### Basic Usage

```python
import rm_abstract

# Initialize with auto-selection
rm_abstract.init(device="auto", verbose=True)

# Your existing HuggingFace code works as-is!
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Generate text - automatically uses best available backend
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Device Switching

```python
import rm_abstract

# Start with GPU
rm_abstract.init(device="gpu:0")

# ... use GPU ...

# Switch to CPU at runtime
rm_abstract.switch_device("cpu")

# Check current device
info = rm_abstract.get_device_info()
print(f"Current: {info['device_type']}:{info['device_id']}")
```

### GPU to NPU Migration Example

See [examples/gpu_to_npu_migration.py](examples/gpu_to_npu_migration.py) for detailed examples.

---

## 11. Multi-Engine Model Serving

Support for multiple serving frameworks with unified API:

### Serving Engines Comparison

| Feature         | vLLM           | Triton          | TorchServe   |
|-----------------|----------------|-----------------|--------------|
| Performance     | ⭐⭐⭐ (Best)   | ⭐⭐⭐           | ⭐⭐          |
| Ease of Use     | ⭐⭐⭐          | ⭐⭐             | ⭐⭐⭐        |
| Multi-model     | ⭐             | ⭐⭐⭐ (Best)    | ⭐⭐⭐        |
| GPU Support     | ✓              | ✓               | ✓            |
| RBLN NPU        | ✓              | ✓               | ✓            |
| OpenAI API      | ✓ (Built-in)   | Custom          | Custom       |

### Usage Examples

```python
from rm_abstract.serving import (
    create_serving_engine, 
    ServingConfig, 
    ServingEngineType, 
    DeviceTarget
)

# vLLM Engine (recommended for LLM)
config = ServingConfig(
    engine=ServingEngineType.VLLM,
    device=DeviceTarget.GPU,
    model_name="gpt2",
)
engine = create_serving_engine(config)
engine.load_model("gpt2")
output = engine.infer("Hello, I am", max_tokens=30)

# Triton Engine (for multi-model serving)
config = ServingConfig(
    engine=ServingEngineType.TRITON,
    device=DeviceTarget.GPU,
)
engine = create_serving_engine(config)
engine.setup_model_repository("/path/to/models")
engine.load_model("gpt2")

# TorchServe Engine (PyTorch native)
config = ServingConfig(
    engine=ServingEngineType.TORCHSERVE,
    device=DeviceTarget.GPU,
)
engine = create_serving_engine(config)
engine.create_model_archive("gpt2")
```

See [examples/serving_engines_demo.py](examples/serving_engines_demo.py) for complete examples.

---

## 12. Rebellions ATOM NPU Support

Dual mode support for Rebellions ATOM NPU:

### vLLM-RBLN Mode (High Performance)

```python
# Requires: pip install vllm-rbln
import rm_abstract
rm_abstract.init(device="rbln:0")

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

### Optimum-RBLN Mode (HuggingFace Integration)

```python
# Requires: pip install optimum-rbln
from optimum.rbln import RBLNModelForCausalLM

model = RBLNModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    export=True,
    rbln_device_id=0,
)
```

### Mode Selection

```python
from rm_abstract.backends.npu.plugins.rebellions import RBLNBackend, RBLNMode

# Auto mode (default) - automatically selects based on available SDK
backend = RBLNBackend(device_id=0, mode="auto")

# Force specific mode
backend = RBLNBackend(device_id=0, mode="vllm")    # vLLM-RBLN
backend = RBLNBackend(device_id=0, mode="optimum") # Optimum-RBLN
```

Reference: [RBLN SDK Documentation](https://docs.rbln.ai/latest/)

---

## 13. Summary

This project:

* **Targets GPU / NPU / PIM** as primary heterogeneous resources
* Abstracts each resource's **pre-compilation/build/runtime stack** requirements
* **Integrates closed-source binary-only NPU/PIM stacks** as black boxes through CLI/C API
* Provides **migration path from existing GPU code** to NPU/PIM with minimal refactoring
* **Supports multiple serving engines** (vLLM, Triton, TorchServe) with unified API
* **Rebellions ATOM NPU** dual mode support (vLLM-RBLN + Optimum-RBLN)
* **System discovery** for hardware/software availability checking

Ultimate goal:

> "Regardless of what resources (GPU/NPU/PIM) are available,
> their Software Stack and build requirements are naturally absorbed by the Orchestrator,
> so applications can focus solely on the LLM API."

---

## License

MIT License

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Documentation

- [Architecture Overview](PLUGIN_ARCHITECTURE.md)
- [Binary Adapter Guide](docs/binary_adapter_guide.md)
- [Adding New Backends](docs/adding_backends.md)

## Examples

- [GPU/vLLM Usage](examples/gpu_vllm_usage.py) - GPU inference with device switching
- [Serving Engines Demo](examples/serving_engines_demo.py) - vLLM, Triton, TorchServe comparison
- [GPU to NPU Migration](examples/gpu_to_npu_migration.py) - Migration guide
