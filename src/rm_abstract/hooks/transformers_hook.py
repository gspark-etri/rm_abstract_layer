"""
Hugging Face Transformers 후킹

from_pretrained 메서드를 가로채서 모델 로드 시 자동으로 백엔드 준비
"""

from functools import wraps
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

# 원본 함수 저장
_original_from_pretrained = None
_controller = None


def activate_transformers_hook(controller) -> None:
    """Transformers 라이브러리 후킹 활성화"""
    global _original_from_pretrained, _controller

    try:
        import transformers
        from transformers import PreTrainedModel

        _controller = controller
        _original_from_pretrained = PreTrainedModel.from_pretrained

        @classmethod
        @wraps(_original_from_pretrained.__func__)
        def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            """모델 로드 후 자동으로 백엔드 준비"""
            # 원본 메서드 호출
            model = _original_from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)

            # 컨트롤러로 모델 준비
            if _controller is not None:
                logger.debug(f"Intercepted model load: {pretrained_model_name_or_path}")
                model = _controller.prepare_model(model)

            return model

        PreTrainedModel.from_pretrained = patched_from_pretrained
        logger.debug("Transformers hook activated")

    except ImportError:
        logger.debug("Transformers not installed, skipping hook")


def deactivate_transformers_hook() -> None:
    """Transformers 라이브러리 후킹 비활성화"""
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
