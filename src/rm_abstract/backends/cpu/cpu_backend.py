"""
CPU 백엔드

PyTorch CPU를 사용한 추론 (Fallback용)
"""

from typing import Any, Dict, Optional
import logging

from ...core.backend import Backend, DeviceType, DeviceInfo

logger = logging.getLogger(__name__)


class CPUBackend(Backend):
    """
    CPU 백엔드

    GPU/NPU 사용 불가시 Fallback으로 사용
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
        """CPU는 항상 사용 가능"""
        try:
            import torch
            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        """CPU 백엔드 초기화"""
        if self._initialized:
            return

        self._initialized = True
        logger.info("CPU backend initialized")

    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        모델 준비 (CPU로 이동)

        Args:
            model: PyTorch 모델
            model_config: 모델 설정

        Returns:
            CPU로 이동된 모델
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
        CPU에서 추론 실행

        Args:
            model: PyTorch 모델
            inputs: 입력 데이터
            **kwargs: 추가 옵션

        Returns:
            추론 결과
        """
        import torch

        with torch.no_grad():
            # 입력이 텐서인 경우 CPU로 이동
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to("cpu")
            elif isinstance(inputs, dict):
                inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # generate 메서드가 있으면 사용 (LLM용)
            if hasattr(model, "generate"):
                return model.generate(**inputs if isinstance(inputs, dict) else inputs, **kwargs)
            else:
                return model(inputs)

    def get_device_info(self) -> DeviceInfo:
        """CPU 디바이스 정보 반환"""
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
        """CPU 리소스 정리"""
        super().cleanup()
