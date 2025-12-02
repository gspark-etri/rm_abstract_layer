"""
Hugging Face Transformers Hooking

Intercepts from_pretrained method to automatically prepare models for backends
and wrap them in ModelProxy for transparent method interception.
"""

from functools import wraps
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

# Store original functions
_original_from_pretrained = None
_controller = None


def activate_transformers_hook(controller) -> None:
    """Activate Transformers library hooking"""
    global _original_from_pretrained, _controller

    try:
        import transformers
        from transformers import PreTrainedModel

        _controller = controller
        _original_from_pretrained = PreTrainedModel.from_pretrained

        @classmethod
        @wraps(_original_from_pretrained.__func__)
        def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            """Load model and wrap in proxy for backend routing"""
            # Call original method
            model = _original_from_pretrained.__func__(
                cls, pretrained_model_name_or_path, *args, **kwargs
            )

            # Prepare model and wrap in proxy
            if _controller is not None:
                logger.debug(f"Intercepted model load: {pretrained_model_name_or_path}")
                model = _controller.prepare_model_with_proxy(model)

            return model

        PreTrainedModel.from_pretrained = patched_from_pretrained
        logger.debug("Transformers hook activated")

    except ImportError:
        logger.debug("Transformers not installed, skipping hook")


def deactivate_transformers_hook() -> None:
    """Deactivate Transformers library hooking"""
    global _original_from_pretrained, _controller

    if _original_from_pretrained is None:
        return

    try:
        from transformers import PreTrainedModel

        PreTrainedModel.from_pretrained = _original_from_pretrained
        _original_from_pretrained = None
        _controller = None
        logger.debug("Transformers hook deactivated")
    except ImportError:
        pass
