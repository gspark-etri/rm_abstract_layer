"""
Hooks Module - Automatic Hooking System

Transparently intercept Transformers and PyTorch modules for backend routing.
Works with ModelProxy to enable Zero Code Change model execution.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Store original functions
_original_functions = {}


def activate_all_hooks(controller) -> None:
    """Activate all hooking systems"""
    from .transformers_hook import activate_transformers_hook
    from .pytorch_hook import activate_pytorch_hook

    activate_transformers_hook(controller)
    activate_pytorch_hook(controller)

    logger.debug("All hooks activated")


def deactivate_all_hooks() -> None:
    """Deactivate all hooking systems"""
    from .transformers_hook import deactivate_transformers_hook
    from .pytorch_hook import deactivate_pytorch_hook

    deactivate_transformers_hook()
    deactivate_pytorch_hook()

    logger.debug("All hooks deactivated")


__all__ = ["activate_all_hooks", "deactivate_all_hooks"]
