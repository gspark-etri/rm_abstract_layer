"""
FuriosaAI RNGD NPU Backend

Inference support for RNGD NPU using Furiosa SDK
"""

from typing import Any, Dict, Optional
import logging

from ..base import NPUBackendBase
from ....core.backend import DeviceType, DeviceInfo

logger = logging.getLogger(__name__)


class FuriosaBackend(NPUBackendBase):
    """
    FuriosaAI RNGD NPU Backend

    Performs LLM inference on RNGD NPU via Furiosa SDK
    """

    def __init__(
        self,
        device_id: int = 0,
        cache_dir: Optional[str] = None,
        compile_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(device_id, cache_dir, compile_options, **kwargs)
        self._runner = None

    @property
    def name(self) -> str:
        return "furiosa"

    @property
    def compiled_model_extension(self) -> str:
        return "enf"

    def is_available(self) -> bool:
        """Check Furiosa SDK and NPU availability"""
        try:
            from furiosa import runtime

            # Check NPU devices
            devices = runtime.list_devices()
            return len(devices) > self.device_id
        except ImportError:
            logger.debug("Furiosa SDK not installed")
            return False
        except Exception as e:
            logger.debug(f"Furiosa NPU not available: {e}")
            return False

    def initialize(self) -> None:
        """Initialize Furiosa backend"""
        if self._initialized:
            return

        try:
            from furiosa import runtime

            # Check device
            devices = runtime.list_devices()
            if self.device_id >= len(devices):
                raise RuntimeError(
                    f"Furiosa device {self.device_id} not found. Available: {len(devices)}"
                )

            self._initialized = True
            logger.info(f"Furiosa backend initialized on device {self.device_id}")

        except Exception as e:
            logger.error(f"Failed to initialize Furiosa backend: {e}")
            raise

    def compile_model(self, model: Any, **kwargs) -> Any:
        """
        Compile model for FuriosaAI RNGD NPU

        Args:
            model: ONNX model path or model object
            **kwargs: Compilation options

        Returns:
            Compiled Furiosa model (ENF format)
        """
        from furiosa import compiler

        # Set compilation options
        batch_size = kwargs.get("batch_size", 1)
        target = kwargs.get("target", "rngd")

        logger.info(f"Compiling model for RNGD NPU (target={target}, batch_size={batch_size})")

        # If ONNX path
        if isinstance(model, str):
            compiled = compiler.compile(
                model,
                target=target,
                batch_size=batch_size,
            )
        else:
            raise ValueError("Furiosa compiler requires ONNX model path")

        return compiled

    def load_compiled_model(self, path: str) -> Any:
        """Load compiled Furiosa model"""
        logger.info(f"Loading compiled Furiosa model from: {path}")

        with open(path, "rb") as f:
            return f.read()

    def save_compiled_model(self, model: Any, path: str) -> None:
        """Save compiled Furiosa model"""
        with open(path, "wb") as f:
            f.write(model)
        logger.info(f"Compiled Furiosa model saved to: {path}")

    def _execute_on_npu(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute inference on FuriosaAI RNGD NPU

        Args:
            model: Compiled Furiosa model (ENF binary)
            inputs: Input data
            **kwargs: Additional options

        Returns:
            Inference result
        """
        from furiosa import runtime
        import numpy as np

        # Convert input data
        if hasattr(inputs, "numpy"):
            inputs = inputs.numpy()
        elif isinstance(inputs, dict):
            inputs = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in inputs.items()}

        # Create runner and execute
        if self._runner is None:
            self._runner = runtime.create_runner(model, device=f"npu{self.device_id}")

        # If input is dictionary
        if isinstance(inputs, dict):
            outputs = self._runner.run(**inputs)
        else:
            outputs = self._runner.run(inputs)

        return outputs

    def get_device_info(self) -> DeviceInfo:
        """Return FuriosaAI NPU device information"""
        try:
            from furiosa import runtime

            devices = runtime.list_devices()
            if self.device_id < len(devices):
                device = devices[self.device_id]
                return DeviceInfo(
                    device_type=DeviceType.NPU,
                    device_id=self.device_id,
                    name="FuriosaAI RNGD",
                    vendor="FuriosaAI",
                    extra={
                        "device_info": str(device),
                    },
                )
        except Exception:
            pass

        return DeviceInfo(
            device_type=DeviceType.NPU,
            device_id=self.device_id,
            name="FuriosaAI RNGD",
            vendor="FuriosaAI",
        )

    def cleanup(self) -> None:
        """Cleanup Furiosa resources"""
        if self._runner is not None:
            try:
                self._runner.close()
            except Exception:
                pass
            self._runner = None
        super().cleanup()
