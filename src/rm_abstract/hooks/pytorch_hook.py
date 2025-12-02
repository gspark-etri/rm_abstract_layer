"""
PyTorch 모듈 후킹

nn.Module.__call__ 을 가로채서 추론 시 백엔드로 라우팅
"""

from functools import wraps
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

# 원본 함수 저장
_original_module_call = None
_controller = None


def activate_pytorch_hook(controller) -> None:
    """PyTorch nn.Module 후킹 활성화"""
    global _original_module_call, _controller

    try:
        import torch.nn as nn

        _controller = controller
        _original_module_call = nn.Module.__call__

        @wraps(_original_module_call)
        def patched_call(self, *args, **kwargs):
            """모델 호출을 가로채서 백엔드로 라우팅"""
            # 컨트롤러가 있고, 이 모델을 가로채야 하는 경우
            if _controller is not None and _controller.should_intercept(self):
                logger.debug(f"Intercepted model call: {type(self).__name__}")
                return _controller.execute(self, args[0] if args else kwargs, **kwargs)

            # 원본 메서드 호출
            return _original_module_call(self, *args, **kwargs)

        nn.Module.__call__ = patched_call
        logger.debug("PyTorch hook activated")

    except ImportError:
        logger.debug("PyTorch not installed, skipping hook")


def deactivate_pytorch_hook() -> None:
    """PyTorch nn.Module 후킹 비활성화"""
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
