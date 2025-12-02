"""
DeviceFlowController - Device Flow Controller

Core controller responsible for device selection, backend management, and model hooking.
"""

from typing import Any, Dict, Optional, Type
import logging

from .backend import Backend, DeviceType, DeviceInfo
from .config import Config

logger = logging.getLogger(__name__)


class DeviceFlowController:
    """
    Device Flow Controller

    - Device string parsing and backend selection
    - Model auto-hooking management
    - Inter-backend switching support
    """

    # Registered backend classes
    _backend_registry: Dict[str, Type[Backend]] = {}

    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self._backend: Optional[Backend] = None
        self._hooks_activated = False
        self._prepared_models: Dict[int, Any] = {}

        # Initialize backend
        self._initialize_backend()

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[Backend]) -> None:
        """
        Register a backend

        Args:
            name: Backend name (e.g., 'gpu', 'rbln', 'furiosa')
            backend_class: Backend class
        """
        cls._backend_registry[name.lower()] = backend_class

    @classmethod
    def get_available_backends(cls) -> Dict[str, bool]:
        """
        Return list of available backends

        Returns:
            {backend_name: availability} dictionary
        """
        result = {}
        for name, backend_class in cls._backend_registry.items():
            try:
                backend = backend_class(device_id=0)
                result[name] = backend.is_available()
            except Exception:
                result[name] = False
        return result

    def _initialize_backend(self) -> None:
        """Initialize backend for the device"""
        device_type = self.config.device_type
        device_id = self.config.device_id

        if device_type == "auto":
            self._backend = self._auto_select_backend()
        else:
            self._backend = self._create_backend(device_type, device_id)

        if self._backend is not None:
            self._backend.initialize()

    def _auto_select_backend(self) -> Optional[Backend]:
        """Auto-select optimal backend (NPU > GPU > CPU)"""
        # Priority: NPU > GPU > CPU
        priority_order = ["rbln", "furiosa", "gpu", "cpu"]

        for backend_name in priority_order:
            if backend_name in self._backend_registry:
                try:
                    backend = self._create_backend(backend_name, 0)
                    if backend and backend.is_available():
                        logger.info(f"Auto-selected backend: {backend_name}")
                        return backend
                except Exception as e:
                    logger.debug(f"Backend {backend_name} not available: {e}")

        logger.warning("No backend available")
        return None

    def _create_backend(self, device_type: str, device_id: int) -> Optional[Backend]:
        """Create backend instance"""
        backend_class = self._backend_registry.get(device_type.lower())

        if backend_class is None:
            raise ValueError(f"Unknown device type: {device_type}")

        # Pass additional options for NPU backends
        if device_type in ["rbln", "furiosa"]:
            return backend_class(
                device_id=device_id,
                cache_dir=self.config.cache_dir,
                compile_options=self.config.compile_options,
            )
        else:
            return backend_class(device_id=device_id)

    @property
    def backend(self) -> Optional[Backend]:
        """Return current backend"""
        return self._backend

    @property
    def device_name(self) -> str:
        """Return current device name"""
        if self._backend is None:
            return "none"
        return f"{self._backend.name}:{self._backend.device_id}"

    def switch_device(self, device: str) -> None:
        """
        Switch device at runtime

        Args:
            device: New device specification
        """
        # Cleanup existing backend
        if self._backend:
            self._backend.cleanup()

        # Update with new settings
        self.config.device = device
        self._prepared_models.clear()

        # Initialize new backend
        self._initialize_backend()

        logger.info(f"Switched to device: {self.device_name}")

    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Prepare model (convert/compile for the backend)

        Args:
            model: Original model
            model_config: Model configuration

        Returns:
            Prepared model
        """
        if self._backend is None:
            logger.warning("No backend available, returning original model")
            return model

        model_id = id(model)

        # Check if model is already prepared
        if model_id in self._prepared_models:
            return self._prepared_models[model_id]

        # Prepare model in backend
        prepared = self._backend.prepare_model(model, model_config)
        self._prepared_models[model_id] = prepared

        return prepared

    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute inference

        Args:
            model: Prepared model
            inputs: Input data
            **kwargs: Additional options

        Returns:
            Inference result
        """
        if self._backend is None:
            raise RuntimeError("No backend available")

        return self._backend.execute(model, inputs, **kwargs)

    def should_intercept(self, model: Any) -> bool:
        """
        Determine whether to intercept this model's calls

        Args:
            model: Model object

        Returns:
            True if should intercept
        """
        # Intercept if model is already prepared
        return id(model) in self._prepared_models

    def get_device_info(self) -> Dict[str, Any]:
        """Return current device information"""
        if self._backend is None:
            return {"status": "no_backend"}

        info = self._backend.get_device_info()
        return {
            "device_type": info.device_type.value,
            "device_id": info.device_id,
            "name": info.name,
            "vendor": info.vendor,
            "memory_total": info.memory_total,
            "memory_available": info.memory_available,
            "extra": info.extra,
        }

    def activate_hooks(self) -> None:
        """Activate hooking system"""
        if self._hooks_activated:
            return

        from ..hooks import activate_all_hooks

        activate_all_hooks(self)
        self._hooks_activated = True

    def deactivate_hooks(self) -> None:
        """Deactivate hooking system"""
        if not self._hooks_activated:
            return

        from ..hooks import deactivate_all_hooks

        deactivate_all_hooks()
        self._hooks_activated = False

    def cleanup(self) -> None:
        """Cleanup resources"""
        self.deactivate_hooks()
        if self._backend:
            self._backend.cleanup()
        self._prepared_models.clear()
