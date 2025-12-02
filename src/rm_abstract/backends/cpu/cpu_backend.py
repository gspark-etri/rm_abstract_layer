"""
CPU Backend

Inference using PyTorch CPU (Fallback)
"""

from typing import Any, Dict, Optional
import logging

from ...core.backend import Backend, DeviceType, DeviceInfo

logger = logging.getLogger(__name__)


class CPUBackend(Backend):
    """
    CPU Backend

    Used as fallback when GPU/NPU is unavailable
    """

    def __init__(self, device_id: int = 0, **kwargs):
        super().__init__(device_id, **kwargs)

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CPU

    @property
    def name(self) -> str:
        return "cpu"

    def is_available(self) -> bool:
        """CPU is always available"""
        try:
            import torch

            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        """Initialize CPU backend"""
        if self._initialized:
            return

        self._initialized = True
        logger.info("CPU backend initialized")

    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Prepare model (move to CPU)

        Args:
            model: PyTorch model
            model_config: Model configuration

        Returns:
            Model moved to CPU
        """
        import torch

        if hasattr(model, "to"):
            model = model.to("cpu")

        if hasattr(model, "eval"):
            model.eval()

        logger.info("Model prepared for CPU inference")
        return model

    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute inference on CPU

        Args:
            model: PyTorch model
            inputs: Input data
            **kwargs: Additional options

        Returns:
            Inference result
        """
        import torch

        with torch.no_grad():
            # Move tensor inputs to CPU
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to("cpu")
            elif isinstance(inputs, dict):
                inputs = {
                    k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
                }

            # Use generate method if available (for LLMs)
            if hasattr(model, "generate"):
                return model.generate(**inputs if isinstance(inputs, dict) else inputs, **kwargs)
            else:
                return model(inputs)

    def get_device_info(self) -> DeviceInfo:
        """Return CPU device information"""
        import platform
        import os

        return DeviceInfo(
            device_type=DeviceType.CPU,
            device_id=0,
            name=platform.processor() or "CPU",
            vendor=platform.system(),
            extra={
                "cpu_count": os.cpu_count(),
                "platform": platform.platform(),
            },
        )

    def cleanup(self) -> None:
        """Cleanup CPU resources"""
        super().cleanup()
