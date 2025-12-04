"""
Plugin Adapter - Bridge between old Backend and new Plugin system

Allows existing Backend implementations to work with the new Plugin system
without requiring immediate refactoring.
"""

from typing import Any, Dict, Optional, Type
import logging

from ..core.plugin import Plugin, PluginMetadata, PluginPriority
from ..core.backend import Backend, DeviceType

logger = logging.getLogger(__name__)


class BackendPluginAdapter(Plugin):
    """
    Adapter to wrap existing Backend as Plugin

    This allows gradual migration from Backend to Plugin system.
    """

    def __init__(self, backend_class: Type[Backend], metadata: PluginMetadata, **kwargs):
        """
        Initialize adapter

        Args:
            backend_class: Backend class to wrap
            metadata: Plugin metadata
            **kwargs: Backend configuration
        """
        super().__init__(**kwargs)
        self._backend_class = backend_class
        self._metadata = metadata
        self._backend: Optional[Backend] = None

        # Extract device_id from kwargs
        self._device_id = kwargs.get("device_id", 0)

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Override this in subclass or pass metadata to __init__"""
        raise NotImplementedError("Subclass must override metadata()")

    def is_available(self) -> bool:
        """Check if backend is available"""
        try:
            # Create temporary instance to check availability
            backend = self._backend_class(device_id=self._device_id, **self.config)
            return backend.is_available()
        except Exception as e:
            logger.debug(f"Backend availability check failed: {e}")
            return False

    def initialize(self) -> None:
        """Initialize backend"""
        if self._initialized:
            return

        try:
            self._backend = self._backend_class(device_id=self._device_id, **self.config)
            self._backend.initialize()
            self._initialized = True
            logger.info(f"Initialized backend adapter for {self._metadata.name}")
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            raise

    def prepare_resource(
        self, resource: Any, config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Prepare resource using backend"""
        if self._backend is None:
            raise RuntimeError("Backend not initialized")

        return self._backend.prepare_model(resource, config)

    def execute(self, resource: Any, inputs: Any, **kwargs) -> Any:
        """Execute using backend"""
        if self._backend is None:
            raise RuntimeError("Backend not initialized")

        return self._backend.execute(resource, inputs, **kwargs)

    def cleanup(self) -> None:
        """Cleanup backend"""
        if self._backend:
            self._backend.cleanup()
            self._backend = None
        self._initialized = False

    def get_capabilities(self) -> Dict[str, Any]:
        """Return backend capabilities"""
        capabilities = {
            "can_compile": False,
            "can_cache": False,
            "supports_batch": True,
            "supports_streaming": False,
        }

        # NPU backends support compilation and caching
        if self._backend and hasattr(self._backend, "compile_model"):
            capabilities["can_compile"] = True
            capabilities["can_cache"] = True

        return capabilities


def create_backend_plugin(
    backend_class: Type[Backend],
    name: str,
    display_name: str,
    version: str = "1.0.0",
    vendor: Optional[str] = None,
    priority: PluginPriority = PluginPriority.NORMAL,
    requires: Optional[list] = None,
    description: str = "",
    device_types: Optional[list] = None,
) -> Type[Plugin]:
    """
    Factory function to create Plugin from Backend class

    Args:
        backend_class: Backend class to wrap
        name: Plugin name
        display_name: Human-readable name
        version: Plugin version
        vendor: Vendor name
        priority: Auto-selection priority
        requires: Required packages
        description: Plugin description
        device_types: Supported device types

    Returns:
        Plugin class wrapping the backend
    """
    # Determine device types from backend if not specified
    if device_types is None:
        try:
            temp_backend = backend_class(device_id=0)
            device_type = temp_backend.device_type
            if device_type == DeviceType.GPU:
                device_types = ["gpu"]
            elif device_type == DeviceType.NPU:
                device_types = ["npu"]
            elif device_type == DeviceType.CPU:
                device_types = ["cpu"]
            else:
                device_types = []
        except Exception:
            device_types = []

    metadata = PluginMetadata(
        name=name,
        display_name=display_name,
        version=version,
        vendor=vendor,
        priority=priority,
        device_types=device_types,
        requires=requires or [],
        description=description,
    )

    # Create dynamic plugin class
    class DynamicBackendPlugin(BackendPluginAdapter):
        def __init__(self, **kwargs):
            super().__init__(backend_class, metadata, **kwargs)

        @classmethod
        def metadata(cls) -> PluginMetadata:
            return metadata

    # Set class name for better debugging
    DynamicBackendPlugin.__name__ = f"{backend_class.__name__}Plugin"
    DynamicBackendPlugin.__qualname__ = DynamicBackendPlugin.__name__

    return DynamicBackendPlugin
