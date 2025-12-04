"""
Auto-register existing backends as plugins

Provides automatic migration from old Backend system to new Plugin system.
"""

import logging

from ..core.plugin import get_registry, PluginPriority
from .plugin_adapter import create_backend_plugin

logger = logging.getLogger(__name__)


def auto_register_backends() -> None:
    """
    Auto-discover and register all existing backends as plugins

    This function wraps existing Backend implementations as Plugins,
    allowing seamless migration without code changes.
    """
    registry = get_registry()

    # Register GPU (vLLM) backend
    try:
        from .gpu.vllm_backend import VLLMBackend

        gpu_plugin = create_backend_plugin(
            backend_class=VLLMBackend,
            name="gpu",
            display_name="vLLM GPU Backend",
            version="1.0.0",
            vendor="vLLM",
            priority=PluginPriority.HIGH,
            requires=["torch", "vllm"],
            description="High-performance LLM inference using vLLM on NVIDIA GPUs",
            device_types=["gpu"],
        )
        registry.register(gpu_plugin)
        logger.debug("Registered GPU (vLLM) plugin")
    except ImportError as e:
        logger.debug(f"GPU plugin not available: {e}")

    # Register CPU backend
    try:
        from .cpu.cpu_backend import CPUBackend

        cpu_plugin = create_backend_plugin(
            backend_class=CPUBackend,
            name="cpu",
            display_name="PyTorch CPU Backend",
            version="1.0.0",
            vendor="PyTorch",
            priority=PluginPriority.LOWEST,
            requires=["torch"],
            description="Fallback CPU inference using PyTorch",
            device_types=["cpu"],
        )
        registry.register(cpu_plugin)
        logger.debug("Registered CPU plugin")
    except ImportError as e:
        logger.debug(f"CPU plugin not available: {e}")

    # Register Rebellions NPU backend
    try:
        from .npu.plugins.rebellions import RBLNBackend

        rbln_plugin = create_backend_plugin(
            backend_class=RBLNBackend,
            name="rbln",
            display_name="Rebellions ATOM NPU",
            version="1.0.0",
            vendor="Rebellions",
            priority=PluginPriority.HIGHEST,
            requires=["rbln"],
            description="Rebellions ATOM NPU backend with on-device compilation",
            device_types=["npu"],
        )
        registry.register(rbln_plugin)
        logger.debug("Registered Rebellions (RBLN) plugin")
    except ImportError as e:
        logger.debug(f"Rebellions plugin not available: {e}")

    # Register FuriosaAI NPU backend
    try:
        from .npu.plugins.furiosa import FuriosaBackend

        furiosa_plugin = create_backend_plugin(
            backend_class=FuriosaBackend,
            name="furiosa",
            display_name="FuriosaAI RNGD NPU",
            version="1.0.0",
            vendor="FuriosaAI",
            priority=PluginPriority.HIGHEST,
            requires=["furiosa"],
            description="FuriosaAI RNGD NPU backend with optimized compilation",
            device_types=["npu"],
        )
        registry.register(furiosa_plugin)
        logger.debug("Registered FuriosaAI plugin")
    except ImportError as e:
        logger.debug(f"FuriosaAI plugin not available: {e}")


def list_registered_plugins() -> dict:
    """
    List all registered plugins with their status

    Returns:
        Dictionary of plugin info
    """
    registry = get_registry()
    plugins = registry.list_plugins(available_only=False)

    result = {}
    for metadata in plugins:
        # Check availability
        plugin = registry.get_plugin(metadata.name)
        available = plugin is not None and plugin.is_available()

        result[metadata.name] = {
            "display_name": metadata.display_name,
            "version": metadata.version,
            "vendor": metadata.vendor,
            "priority": metadata.priority.value,
            "device_types": metadata.device_types,
            "available": available,
            "description": metadata.description,
        }

    return result
