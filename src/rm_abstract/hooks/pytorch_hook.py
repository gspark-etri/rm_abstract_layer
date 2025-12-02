"""
PyTorch Module Hooking

Note: With the ModelProxy pattern, this hook is now optional.
ModelProxy handles method interception (generate, forward, __call__) directly.

This hook is kept for:
1. Models loaded without from_pretrained (e.g., custom PyTorch models)
2. Legacy support for code that creates models directly
"""

from functools import wraps
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

# Store original functions
_original_module_call = None
_controller = None


def activate_pytorch_hook(controller) -> None:
    """
    Activate PyTorch nn.Module hooking

    Note: This is a fallback hook for models not loaded via from_pretrained.
    For models loaded via from_pretrained, ModelProxy handles interception.
    """
    global _original_module_call, _controller

    try:
        import torch.nn as nn

        _controller = controller
        _original_module_call = nn.Module.__call__

        @wraps(_original_module_call)
        def patched_call(self, *args, **kwargs):
            """Intercept model calls and route to backend"""
            # Skip if this is already a ModelProxy (avoid double interception)
            from ..core.model_proxy import ModelProxy

            if isinstance(self, ModelProxy):
                return _original_module_call(self, *args, **kwargs)

            # If controller exists and model should be intercepted
            if _controller is not None and _controller.should_intercept(self):
                logger.debug(f"Intercepted model call: {type(self).__name__}")
                return _controller.execute(self, args[0] if args else kwargs, **kwargs)

            # Call original method
            return _original_module_call(self, *args, **kwargs)

        nn.Module.__call__ = patched_call
        logger.debug("PyTorch hook activated")

    except ImportError:
        logger.debug("PyTorch not installed, skipping hook")


def deactivate_pytorch_hook() -> None:
    """Deactivate PyTorch nn.Module hooking"""
    global _original_module_call, _controller

    if _original_module_call is None:
        return

    try:
        import torch.nn as nn

        nn.Module.__call__ = _original_module_call
        _original_module_call = None
        _controller = None
        logger.debug("PyTorch hook deactivated")
    except ImportError:
        pass
