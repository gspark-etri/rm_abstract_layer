"""
RM Abstract Layer - Heterogeneous AI Accelerator Unified Compatibility Library

An abstraction layer that enables existing GPU inference scripts to run on NPU/GPU
without any code modification.
"""

from typing import Optional, Dict, Any
import os

from .core.controller import DeviceFlowController
from .core.config import Config

__version__ = "0.1.0"
__all__ = [
    "init",
    "switch_device",
    "get_device_info",
    "get_controller",
    "get_available_backends",
]

# Global controller instance
_global_controller: Optional[DeviceFlowController] = None


def init(
    device: str = "auto",
    cache_dir: Optional[str] = None,
    compile_options: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> DeviceFlowController:
    """
    Initialize RM Abstract Layer

    Args:
        device: Device specification
            - "auto": Auto-select (NPU > GPU > CPU priority)
            - "gpu:0", "gpu:1": Specific GPU
            - "rbln:0": Rebellions ATOM NPU
            - "furiosa:0": FuriosaAI RNGD NPU
            - "cpu": CPU
        cache_dir: NPU compilation cache directory (default: ~/.rm_abstract/cache)
        compile_options: NPU compilation options
        verbose: Whether to print compilation progress

    Returns:
        DeviceFlowController instance

    Example:
        >>> import rm_abstract
        >>> rm_abstract.init(device="rbln:0")
        >>> # Use existing code as-is
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    """
    global _global_controller

    # Register all backends
    _register_backends()

    # Load settings from environment variables
    device = os.environ.get("RM_DEVICE", device)
    cache_dir = os.environ.get("RM_CACHE_DIR", cache_dir)

    # Create Config
    config = Config(
        device=device,
        cache_dir=cache_dir,
        compile_options=compile_options or {},
        verbose=verbose,
    )

    # Create and initialize Controller
    _global_controller = DeviceFlowController(config)
    _global_controller.activate_hooks()

    if verbose:
        print(f"[RM Abstract] Initialized with device: {_global_controller.device_name}")

    return _global_controller


def switch_device(device: str) -> None:
    """
    Switch device at runtime

    Args:
        device: New device specification
    """
    global _global_controller

    if _global_controller is None:
        raise RuntimeError("RM Abstract Layer not initialized. Call init() first.")

    _global_controller.switch_device(device)


def get_device_info() -> Dict[str, Any]:
    """
    Return current device information

    Returns:
        Device information dictionary
    """
    global _global_controller

    if _global_controller is None:
        raise RuntimeError("RM Abstract Layer not initialized. Call init() first.")

    return _global_controller.get_device_info()


def get_controller() -> Optional[DeviceFlowController]:
    """
    Return current global controller

    Returns:
        DeviceFlowController instance or None
    """
    return _global_controller


def get_available_backends() -> Dict[str, bool]:
    """
    Return available backends

    Returns:
        Dictionary of {backend_name: is_available}
    """
    # Register backends first
    _register_backends()
    return DeviceFlowController.get_available_backends()


def _register_backends() -> None:
    """Register all available backends"""
    # CPU backend (always available if torch is installed)
    try:
        from .backends.cpu.cpu_backend import CPUBackend

        DeviceFlowController.register_backend("cpu", CPUBackend)
    except ImportError:
        pass

    # GPU backend (vLLM)
    try:
        from .backends.gpu.vllm_backend import VLLMBackend

        DeviceFlowController.register_backend("gpu", VLLMBackend)
    except ImportError:
        pass

    # Rebellions NPU backend
    try:
        from .backends.npu.plugins.rebellions import RBLNBackend

        DeviceFlowController.register_backend("rbln", RBLNBackend)
    except ImportError:
        pass

    # FuriosaAI NPU backend
    try:
        from .backends.npu.plugins.furiosa import FuriosaBackend

        DeviceFlowController.register_backend("furiosa", FuriosaBackend)
    except ImportError:
        pass
