"""
FuriosaAI RNGD NPU 백엔드

Furiosa SDK를 사용한 RNGD NPU 추론 지원
"""

from typing import Any, Dict, Optional
import logging

from ..base import NPUBackendBase
from ....core.backend import DeviceType, DeviceInfo

logger = logging.getLogger(__name__)


class FuriosaBackend(NPUBackendBase):
    """
    FuriosaAI RNGD NPU 백엔드

    Furiosa SDK를 통해 RNGD NPU에서 LLM 추론 수행
    """

    def __init__(
        self,
        device_id: int = 0,
        cache_dir: Optional[str] = None,
        compile_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(device_id, cache_dir, compile_options, **kwargs)
        self._runner = None

    @property
    def name(self) -> str:
        return "furiosa"

    @property
    def compiled_model_extension(self) -> str:
        return "enf"

    def is_available(self) -> bool:
        """Furiosa SDK 및 NPU 사용 가능 여부 확인"""
        try:
            from furiosa import runtime

            # NPU 디바이스 확인
            devices = runtime.list_devices()
            return len(devices) > self.device_id
        except ImportError:
            logger.debug("Furiosa SDK not installed")
            return False
        except Exception as e:
            logger.debug(f"Furiosa NPU not available: {e}")
            return False

    def initialize(self) -> None:
        """Furiosa 백엔드 초기화"""
        if self._initialized:
            return

        try:
            from furiosa import runtime

            # 디바이스 확인
            devices = runtime.list_devices()
            if self.device_id >= len(devices):
                raise RuntimeError(f"Furiosa device {self.device_id} not found. Available: {len(devices)}")

            self._initialized = True
            logger.info(f"Furiosa backend initialized on device {self.device_id}")

        except Exception as e:
            logger.error(f"Failed to initialize Furiosa backend: {e}")
            raise

    def compile_model(self, model: Any, **kwargs) -> Any:
        """
        모델을 FuriosaAI RNGD NPU용으로 컴파일

        Args:
            model: ONNX 모델 경로 또는 모델 객체
            **kwargs: 컴파일 옵션

        Returns:
            컴파일된 Furiosa 모델 (ENF 포맷)
        """
        from furiosa import compiler

        # 컴파일 옵션 설정
        batch_size = kwargs.get("batch_size", 1)
        target = kwargs.get("target", "rngd")

        logger.info(f"Compiling model for RNGD NPU (target={target}, batch_size={batch_size})")

        # ONNX 경로인 경우
        if isinstance(model, str):
            compiled = compiler.compile(
                model,
                target=target,
                batch_size=batch_size,
            )
        else:
            raise ValueError("Furiosa compiler requires ONNX model path")

        return compiled

    def load_compiled_model(self, path: str) -> Any:
        """컴파일된 Furiosa 모델 로드"""
        logger.info(f"Loading compiled Furiosa model from: {path}")

        with open(path, "rb") as f:
            return f.read()

    def save_compiled_model(self, model: Any, path: str) -> None:
        """컴파일된 Furiosa 모델 저장"""
        with open(path, "wb") as f:
            f.write(model)
        logger.info(f"Compiled Furiosa model saved to: {path}")

    def _execute_on_npu(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        FuriosaAI RNGD NPU에서 추론 실행

        Args:
            model: 컴파일된 Furiosa 모델 (ENF 바이너리)
            inputs: 입력 데이터
            **kwargs: 추가 옵션

        Returns:
            추론 결과
        """
        from furiosa import runtime
        import numpy as np

        # 입력 데이터 변환
        if hasattr(inputs, "numpy"):
            inputs = inputs.numpy()
        elif isinstance(inputs, dict):
            inputs = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in inputs.items()}

        # Runner 생성 및 실행
        if self._runner is None:
            self._runner = runtime.create_runner(model, device=f"npu{self.device_id}")

        # 입력이 딕셔너리인 경우
        if isinstance(inputs, dict):
            outputs = self._runner.run(**inputs)
        else:
            outputs = self._runner.run(inputs)

        return outputs

    def get_device_info(self) -> DeviceInfo:
        """FuriosaAI NPU 디바이스 정보 반환"""
        try:
            from furiosa import runtime

            devices = runtime.list_devices()
            if self.device_id < len(devices):
                device = devices[self.device_id]
                return DeviceInfo(
                    device_type=DeviceType.NPU,
                    device_id=self.device_id,
                    name="FuriosaAI RNGD",
                    vendor="FuriosaAI",
                    extra={
                        "device_info": str(device),
                    },
                )
        except Exception:
            pass

        return DeviceInfo(
            device_type=DeviceType.NPU,
            device_id=self.device_id,
            name="FuriosaAI RNGD",
            vendor="FuriosaAI",
        )

    def cleanup(self) -> None:
        """Furiosa 리소스 정리"""
        if self._runner is not None:
            try:
                self._runner.close()
            except Exception:
                pass
            self._runner = None
        super().cleanup()
