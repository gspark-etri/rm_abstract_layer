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

    Note: With ModelProxy pattern, this hook is DISABLED by default.
    ModelProxy handles all method interception (generate, forward, __call__).
    Enabling this hook would cause issues with models that call themselves
    internally (e.g., during generate()).

    This function is kept for API compatibility but does nothing.
    """
    # DISABLED: ModelProxy handles interception, enabling this causes issues
    # with internal model calls during generate()
    logger.debug("PyTorch hook disabled (ModelProxy handles interception)")


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
