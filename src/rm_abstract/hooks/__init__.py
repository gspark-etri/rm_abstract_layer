"""
Hooks module - 자동 후킹 시스템

Transformers, PyTorch 모듈을 투명하게 가로채서 백엔드 라우팅
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# 원본 함수 저장소
_original_functions = {}


def activate_all_hooks(controller) -> None:
    """모든 후킹 시스템 활성화"""
    from .transformers_hook import activate_transformers_hook
    from .pytorch_hook import activate_pytorch_hook

    activate_transformers_hook(controller)
    activate_pytorch_hook(controller)

    logger.debug("All hooks activated")


def deactivate_all_hooks() -> None:
    """모든 후킹 시스템 비활성화"""
    from .transformers_hook import deactivate_transformers_hook
    from .pytorch_hook import deactivate_pytorch_hook

    deactivate_transformers_hook()
    deactivate_pytorch_hook()

    logger.debug("All hooks deactivated")


__all__ = ["activate_all_hooks", "deactivate_all_hooks"]
