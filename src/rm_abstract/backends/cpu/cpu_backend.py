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
            inputs: Input data (can be tensor, dict, or None)
            **kwargs: Additional options (may include proxy metadata)

        Returns:
            Inference result
        """
        import torch

        # Extract proxy metadata (set by ModelProxy)
        proxy_method = kwargs.pop("_proxy_method", None)
        original_model = kwargs.pop("original_model", None)

        with torch.no_grad():
            # Prepare inputs - handle various formats
            if inputs is None:
                # All inputs are in kwargs (e.g., input_ids=..., attention_mask=...)
                exec_kwargs = {
                    k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
                }
                exec_inputs = None
            elif isinstance(inputs, dict):
                # Dict inputs - merge with kwargs, move tensors to CPU
                exec_kwargs = {**kwargs}
                for k, v in inputs.items():
                    exec_kwargs[k] = v.to("cpu") if isinstance(v, torch.Tensor) else v
                exec_inputs = None
            elif isinstance(inputs, torch.Tensor):
                exec_inputs = inputs.to("cpu")
                exec_kwargs = kwargs
            else:
                exec_inputs = inputs
                exec_kwargs = kwargs

            # Route based on proxy method
            if proxy_method == "generate" and hasattr(model, "generate"):
                if exec_inputs is None:
                    return model.generate(**exec_kwargs)
                else:
                    return model.generate(exec_inputs, **exec_kwargs)
            elif proxy_method in ("forward", "__call__"):
                if exec_inputs is None:
                    return model(**exec_kwargs)
                else:
                    return model(exec_inputs, **exec_kwargs)
            else:
                # Default: use generate if available, otherwise forward
                if hasattr(model, "generate"):
                    if exec_inputs is None:
                        return model.generate(**exec_kwargs)
                    else:
                        return model.generate(exec_inputs, **exec_kwargs)
                else:
                    if exec_inputs is None:
                        return model(**exec_kwargs)
                    else:
                        return model(exec_inputs, **exec_kwargs)

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
