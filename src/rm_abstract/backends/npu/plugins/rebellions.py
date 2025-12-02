"""
Rebellions ATOM NPU 백엔드

RBLN SDK를 사용한 Rebellions ATOM NPU 추론 지원
"""

from typing import Any, Dict, Optional
import logging

from ..base import NPUBackendBase
from ....core.backend import DeviceType, DeviceInfo

logger = logging.getLogger(__name__)


class RBLNBackend(NPUBackendBase):
    """
    Rebellions ATOM NPU 백엔드

    RBLN SDK를 통해 ATOM NPU에서 LLM 추론 수행
    """

    def __init__(
        self,
        device_id: int = 0,
        cache_dir: Optional[str] = None,
        compile_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(device_id, cache_dir, compile_options, **kwargs)
        self._runtime = None

    @property
    def name(self) -> str:
        return "rbln"

    @property
    def compiled_model_extension(self) -> str:
        return "rbln"

    def is_available(self) -> bool:
        """Rebellions SDK 및 NPU 사용 가능 여부 확인"""
        try:
            import rebel
            # NPU 디바이스 확인
            devices = rebel.get_devices()
            return len(devices) > self.device_id
        except ImportError:
            logger.debug("Rebellions SDK (rebel) not installed")
            return False
        except Exception as e:
            logger.debug(f"Rebellions NPU not available: {e}")
            return False

    def initialize(self) -> None:
        """Rebellions 백엔드 초기화"""
        if self._initialized:
            return

        try:
            import rebel

            # 디바이스 확인
            devices = rebel.get_devices()
            if self.device_id >= len(devices):
                raise RuntimeError(f"RBLN device {self.device_id} not found. Available: {len(devices)}")

            self._initialized = True
            logger.info(f"Rebellions backend initialized on device {self.device_id}")

        except Exception as e:
            logger.error(f"Failed to initialize Rebellions backend: {e}")
            raise

    def compile_model(self, model: Any, **kwargs) -> Any:
        """
        모델을 Rebellions NPU용으로 컴파일

        Args:
            model: ONNX 모델 경로 또는 모델 객체
            **kwargs: 컴파일 옵션

        Returns:
            컴파일된 RBLN 모델
        """
        import rebel

        # 컴파일 옵션 설정
        optimization_level = kwargs.get("optimization_level", 3)
        precision = kwargs.get("precision", "fp16")

        logger.info(f"Compiling model for ATOM NPU (opt_level={optimization_level}, precision={precision})")

        # ONNX 경로인 경우
        if isinstance(model, str):
            compiled = rebel.compile_from_onnx(
                model,
                target="atom",
                optimization_level=optimization_level,
            )
        else:
            # PyTorch 모델인 경우 직접 컴파일
            compiled = rebel.compile(
                model,
                target="atom",
                optimization_level=optimization_level,
            )

        return compiled

    def load_compiled_model(self, path: str) -> Any:
        """컴파일된 RBLN 모델 로드"""
        import rebel

        logger.info(f"Loading compiled RBLN model from: {path}")
        return rebel.load(path)

    def save_compiled_model(self, model: Any, path: str) -> None:
        """컴파일된 RBLN 모델 저장"""
        import rebel

        rebel.save(model, path)
        logger.info(f"Compiled RBLN model saved to: {path}")

    def _execute_on_npu(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Rebellions NPU에서 추론 실행

        Args:
            model: 컴파일된 RBLN 모델
            inputs: 입력 데이터
            **kwargs: 추가 옵션

        Returns:
            추론 결과
        """
        import rebel
        import numpy as np

        # 입력 데이터 변환
        if hasattr(inputs, "numpy"):
            inputs = inputs.numpy()
        elif isinstance(inputs, dict):
            inputs = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in inputs.items()}

        # 런타임 생성 및 실행
        if self._runtime is None:
            self._runtime = rebel.Runtime(device_id=self.device_id)

        outputs = self._runtime.run(model, inputs)

        return outputs

    def get_device_info(self) -> DeviceInfo:
        """Rebellions NPU 디바이스 정보 반환"""
        try:
            import rebel

            devices = rebel.get_devices()
            if self.device_id < len(devices):
                device = devices[self.device_id]
                return DeviceInfo(
                    device_type=DeviceType.NPU,
                    device_id=self.device_id,
                    name=f"Rebellions ATOM",
                    vendor="Rebellions",
                    extra={
                        "sdk_version": getattr(rebel, "__version__", "unknown"),
                    },
                )
        except Exception:
            pass

        return DeviceInfo(
            device_type=DeviceType.NPU,
            device_id=self.device_id,
            name="Rebellions ATOM",
            vendor="Rebellions",
        )

    def cleanup(self) -> None:
        """Rebellions 리소스 정리"""
        self._runtime = None
        super().cleanup()
