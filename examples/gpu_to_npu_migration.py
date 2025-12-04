"""
GPU to NPU Migration Example

Demonstrates how to migrate existing GPU-based inference code
to run on NPU with minimal changes using the DeviceRuntime abstraction.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Part 1: Original GPU-only Code (Before refactoring)
# ============================================================================


class OriginalGpuCode:
    """
    Original GPU-only inference code

    This represents typical GPU-based LLM inference code.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self, model_path: str):
        """Load model to GPU"""
        logger.info(f"[Original] Loading model to {self.device}")
        # In real code: self.model = load_model(model_path).to(self.device)
        self.model = f"model_on_{self.device}"
        self.tokenizer = "tokenizer"

    def run_inference(self, prompt: str, max_tokens: int = 100) -> str:
        """Run inference on GPU"""
        logger.info(f"[Original] Running inference on {self.device}")
        # In real code: input_ids = self.tokenizer(prompt, ...).to(self.device)
        # output = self.model.generate(input_ids, max_new_tokens=max_tokens)
        result = f"[GPU] Generated text for '{prompt}' (max_tokens={max_tokens})"
        return result


# ============================================================================
# Part 2: Refactored with DeviceRuntime Abstraction
# ============================================================================


class DeviceRuntime(ABC):
    """
    Device Runtime Abstraction

    Centralizes device-specific operations (GPU/NPU/PIM)
    to enable easy swapping between different backends.
    """

    @abstractmethod
    def load_model(self, model_path: str):
        """Load model for this device"""
        ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate text"""
        ...

    @abstractmethod
    def cleanup(self):
        """Cleanup resources"""
        ...


class GpuTorchRuntime(DeviceRuntime):
    """
    GPU Runtime using PyTorch

    GPU implementation of DeviceRuntime.
    Minimal changes from original code.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self.tokenizer = None
        logger.info(f"[GpuRuntime] Initialized for {device}")

    def load_model(self, model_path: str):
        """Load model to GPU"""
        logger.info(f"[GpuRuntime] Loading model: {model_path}")
        # In real code:
        # self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = f"model_on_{self.device}"
        self.tokenizer = "tokenizer"

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate on GPU"""
        logger.info(f"[GpuRuntime] Generating: '{prompt}'")
        # In real code:
        # input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # output = self.model.generate(input_ids, max_new_tokens=max_tokens)
        # return self.tokenizer.decode(output[0], skip_special_tokens=True)
        return f"[GPU] Generated for '{prompt}' (max_tokens={max_tokens})"

    def cleanup(self):
        """Cleanup GPU resources"""
        logger.info("[GpuRuntime] Cleanup")
        # In real code: torch.cuda.empty_cache()
        self.model = None


class NpuRuntime(DeviceRuntime):
    """
    NPU Runtime using Binary Adapter

    NPU implementation that wraps closed-source compiler/runtime.
    """

    def __init__(
        self,
        npu_device_id: int = 0,
        artifact_path: Optional[str] = None,
        runtime_so: str = "libnpu_runtime.so",
    ):
        self.npu_device_id = npu_device_id
        self.artifact_path = artifact_path
        self.runtime_so = runtime_so
        self.runtime_adapter = None
        logger.info(f"[NpuRuntime] Initialized for NPU device {npu_device_id}")

    def load_model(self, model_path: str):
        """Load compiled NPU model"""
        logger.info(f"[NpuRuntime] Loading compiled artifact: {self.artifact_path}")

        # In real implementation, use BinaryRuntimeAdapter:
        # from rm_abstract.core.binary_adapter import BinaryRuntimeAdapter
        # self.runtime_adapter = BinaryRuntimeAdapter(
        #     engine_path=self.artifact_path,
        #     device_id=self.npu_device_id,
        #     runtime_so=self.runtime_so
        # )
        # self.runtime_adapter.load()

        # Simulated for demo
        self.runtime_adapter = f"npu_engine_{self.npu_device_id}"

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate on NPU"""
        logger.info(f"[NpuRuntime] Generating on NPU: '{prompt}'")

        # In real code:
        # input_ids = common_tokenizer(prompt)
        # output_ids = self.runtime_adapter.run(input_ids, max_tokens=max_tokens)
        # return common_detokenizer(output_ids)

        return f"[NPU] Generated for '{prompt}' (max_tokens={max_tokens})"

    def cleanup(self):
        """Cleanup NPU resources"""
        logger.info("[NpuRuntime] Cleanup")
        # In real code: self.runtime_adapter.unload()
        self.runtime_adapter = None


class PimRuntime(DeviceRuntime):
    """
    PIM Runtime

    Processing-In-Memory runtime for memory-intensive operations.
    """

    def __init__(self, pim_device_id: int = 0):
        self.pim_device_id = pim_device_id
        logger.info(f"[PimRuntime] Initialized for PIM device {pim_device_id}")

    def load_model(self, model_path: str):
        """Load model for PIM"""
        logger.info(f"[PimRuntime] Loading model: {model_path}")
        # PIM-specific model loading
        self.model = f"pim_model_{self.pim_device_id}"

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate using PIM"""
        logger.info(f"[PimRuntime] Generating on PIM: '{prompt}'")
        return f"[PIM] Generated for '{prompt}' (max_tokens={max_tokens})"

    def cleanup(self):
        """Cleanup PIM resources"""
        logger.info("[PimRuntime] Cleanup")
        self.model = None


# ============================================================================
# Part 3: Application Code (Minimal Changes)
# ============================================================================


class LLMApplication:
    """
    LLM Application

    Application code that remains almost unchanged
    regardless of backend (GPU/NPU/PIM).
    """

    def __init__(self, runtime: DeviceRuntime):
        """
        Initialize with runtime

        Args:
            runtime: DeviceRuntime implementation (GPU/NPU/PIM)
        """
        self.runtime = runtime

    def setup(self, model_path: str):
        """Setup model"""
        self.runtime.load_model(model_path)

    def process_request(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Process inference request

        This is the core application logic that doesn't change
        regardless of which backend is used.
        """
        return self.runtime.generate(prompt, max_tokens)

    def shutdown(self):
        """Shutdown and cleanup"""
        self.runtime.cleanup()


# ============================================================================
# Part 4: Demonstration
# ============================================================================


def demo_original_gpu_code():
    """Demo: Original GPU-only code"""
    print("\n" + "=" * 60)
    print("Demo 1: Original GPU-only Code")
    print("=" * 60)

    app = OriginalGpuCode(device="cuda:0")
    app.load_model("/path/to/model")
    result = app.run_inference("Hello, world!", max_tokens=50)
    print(f"Result: {result}\n")


def demo_refactored_gpu():
    """Demo: Refactored GPU code with DeviceRuntime"""
    print("\n" + "=" * 60)
    print("Demo 2: Refactored GPU Code (with DeviceRuntime)")
    print("=" * 60)

    # Create GPU runtime
    gpu_runtime = GpuTorchRuntime(device="cuda:0")

    # Application code (unchanged)
    app = LLMApplication(runtime=gpu_runtime)
    app.setup("/path/to/model")
    result = app.process_request("Hello, world!", max_tokens=50)
    print(f"Result: {result}\n")
    app.shutdown()


def demo_npu_migration():
    """Demo: Same app code, NPU runtime"""
    print("\n" + "=" * 60)
    print("Demo 3: NPU Runtime (Same Application Code!)")
    print("=" * 60)

    # Create NPU runtime (different implementation, same interface)
    npu_runtime = NpuRuntime(
        npu_device_id=0,
        artifact_path="/path/to/compiled_model.bin",
        runtime_so="libvendor_npu.so",
    )

    # Application code (EXACTLY THE SAME as GPU version)
    app = LLMApplication(runtime=npu_runtime)
    app.setup("/path/to/model")  # Artifact already compiled
    result = app.process_request("Hello, world!", max_tokens=50)
    print(f"Result: {result}\n")
    app.shutdown()


def demo_pim_runtime():
    """Demo: PIM runtime"""
    print("\n" + "=" * 60)
    print("Demo 4: PIM Runtime")
    print("=" * 60)

    # Create PIM runtime
    pim_runtime = PimRuntime(pim_device_id=0)

    # Application code (still the same)
    app = LLMApplication(runtime=pim_runtime)
    app.setup("/path/to/model")
    result = app.process_request("Hello, world!", max_tokens=50)
    print(f"Result: {result}\n")
    app.shutdown()


def demo_runtime_switching():
    """Demo: Runtime switching at... runtime"""
    print("\n" + "=" * 60)
    print("Demo 5: Runtime Switching")
    print("=" * 60)

    runtimes = {
        "gpu": GpuTorchRuntime(device="cuda:0"),
        "npu": NpuRuntime(npu_device_id=0, artifact_path="/compiled.bin"),
        "pim": PimRuntime(pim_device_id=0),
    }

    prompts = [
        ("Short query", "npu"),  # NPU for low latency
        ("Long context analysis...", "gpu"),  # GPU for throughput
        ("Memory-intensive task", "pim"),  # PIM for memory bandwidth
    ]

    for prompt, backend in prompts:
        print(f"\nProcessing: '{prompt}' on {backend.upper()}")
        app = LLMApplication(runtime=runtimes[backend])
        app.setup("/path/to/model")
        result = app.process_request(prompt, max_tokens=50)
        print(f"Result: {result}")
        app.shutdown()


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("GPU to NPU Migration Demo")
    print("Minimal Code Changes for Heterogeneous Resources")
    print("=" * 60)

    demo_original_gpu_code()
    demo_refactored_gpu()
    demo_npu_migration()
    demo_pim_runtime()
    demo_runtime_switching()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Original GPU code needs minimal refactoring")
    print("2. Centralize device-specific code in DeviceRuntime")
    print("3. Application code remains unchanged across backends")
    print("4. Easy to add new backends (NPU, PIM, etc.)")
    print("5. Runtime switching enables dynamic resource allocation")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
