"""
Rebellions ATOM NPU Backend

Inference support for Rebellions ATOM NPU using RBLN SDK
"""

from typing import Any, Dict, Optional
import logging

from ..base import NPUBackendBase
from ....core.backend import DeviceType, DeviceInfo

logger = logging.getLogger(__name__)


class RBLNBackend(NPUBackendBase):
    """
    Rebellions ATOM NPU Backend

    Performs LLM inference on ATOM NPU via RBLN SDK
    """

    def __init__(
        self,
        device_id: int = 0,
        cache_dir: Optional[str] = None,
        compile_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(device_id, cache_dir, compile_options, **kwargs)
        self._runtime = None

    @property
    def name(self) -> str:
        return "rbln"

    @property
    def compiled_model_extension(self) -> str:
        return "rbln"

    def is_available(self) -> bool:
        """Check Rebellions SDK and NPU availability"""
        try:
            import rebel

            # Check NPU devices
            devices = rebel.get_devices()
            return len(devices) > self.device_id
        except ImportError:
            logger.debug("Rebellions SDK (rebel) not installed")
            return False
        except Exception as e:
            logger.debug(f"Rebellions NPU not available: {e}")
            return False

    def initialize(self) -> None:
        """Initialize Rebellions backend"""
        if self._initialized:
            return

        try:
            import rebel

            # Check device
            devices = rebel.get_devices()
            if self.device_id >= len(devices):
                raise RuntimeError(
                    f"RBLN device {self.device_id} not found. Available: {len(devices)}"
                )

            self._initialized = True
            logger.info(f"Rebellions backend initialized on device {self.device_id}")

        except Exception as e:
            logger.error(f"Failed to initialize Rebellions backend: {e}")
            raise

    def compile_model(self, model: Any, **kwargs) -> Any:
        """
        Compile model for Rebellions NPU

        Args:
            model: ONNX model path or model object
            **kwargs: Compilation options

        Returns:
            Compiled RBLN model
        """
        import rebel

        # Set compilation options
        optimization_level = kwargs.get("optimization_level", 3)
        precision = kwargs.get("precision", "fp16")

        logger.info(
            f"Compiling model for ATOM NPU (opt_level={optimization_level}, precision={precision})"
        )

        # If ONNX path
        if isinstance(model, str):
            compiled = rebel.compile_from_onnx(
                model,
                target="atom",
                optimization_level=optimization_level,
            )
        else:
            # Compile PyTorch model directly
            compiled = rebel.compile(
                model,
                target="atom",
                optimization_level=optimization_level,
            )

        return compiled

    def load_compiled_model(self, path: str) -> Any:
        """Load compiled RBLN model"""
        import rebel

        logger.info(f"Loading compiled RBLN model from: {path}")
        return rebel.load(path)

    def save_compiled_model(self, model: Any, path: str) -> None:
        """Save compiled RBLN model"""
        import rebel

        rebel.save(model, path)
        logger.info(f"Compiled RBLN model saved to: {path}")

    def _execute_on_npu(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute inference on Rebellions NPU

        Args:
            model: Compiled RBLN model
            inputs: Input data
            **kwargs: Additional options

        Returns:
            Inference result
        """
        import rebel
        import numpy as np

        # Convert input data
        if hasattr(inputs, "numpy"):
            inputs = inputs.numpy()
        elif isinstance(inputs, dict):
            inputs = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in inputs.items()}

        # Create runtime and execute
        if self._runtime is None:
            self._runtime = rebel.Runtime(device_id=self.device_id)

        outputs = self._runtime.run(model, inputs)

        return outputs

    def get_device_info(self) -> DeviceInfo:
        """Return Rebellions NPU device information"""
        try:
            import rebel

            devices = rebel.get_devices()
            if self.device_id < len(devices):
                device = devices[self.device_id]
                return DeviceInfo(
                    device_type=DeviceType.NPU,
                    device_id=self.device_id,
                    name=f"Rebellions ATOM",
                    vendor="Rebellions",
                    extra={
                        "sdk_version": getattr(rebel, "__version__", "unknown"),
                    },
                )
        except Exception:
            pass

        return DeviceInfo(
            device_type=DeviceType.NPU,
            device_id=self.device_id,
            name="Rebellions ATOM",
            vendor="Rebellions",
        )

    def cleanup(self) -> None:
        """Cleanup Rebellions resources"""
        self._runtime = None
        super().cleanup()
