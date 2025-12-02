"""
Backend 추상 베이스 클래스

모든 디바이스 백엔드(GPU, NPU, CPU)가 구현해야 하는 인터페이스 정의
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from enum import Enum


class DeviceType(Enum):
    """디바이스 타입"""
    GPU = "gpu"
    NPU = "npu"
    CPU = "cpu"
    AUTO = "auto"


@dataclass
class DeviceInfo:
    """디바이스 정보"""
    device_type: DeviceType
    device_id: int
    name: str
    vendor: Optional[str] = None
    memory_total: Optional[int] = None  # bytes
    memory_available: Optional[int] = None  # bytes
    extra: Optional[Dict[str, Any]] = None


class Backend(ABC):
    """
    Backend 추상 베이스 클래스

    모든 디바이스 백엔드가 구현해야 하는 공통 인터페이스
    """

    def __init__(self, device_id: int = 0, **kwargs):
        """
        Args:
            device_id: 디바이스 인덱스 (0, 1, 2, ...)
            **kwargs: 백엔드별 추가 옵션
        """
        self.device_id = device_id
        self.options = kwargs
        self._initialized = False

    @property
    @abstractmethod
    def device_type(self) -> DeviceType:
        """디바이스 타입 반환"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """백엔드 이름 반환 (예: 'CUDA', 'RBLN', 'Furiosa')"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        해당 백엔드 사용 가능 여부 확인

        Returns:
            사용 가능하면 True
        """
        ...

    @abstractmethod
    def initialize(self) -> None:
        """
        백엔드 초기화

        디바이스 연결, 런타임 초기화 등 수행
        """
        ...

    @abstractmethod
    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        모델 준비

        - GPU: 모델을 GPU로 이동 (passthrough)
        - NPU: ONNX 변환 → NPU 컴파일 → 컴파일된 모델 반환
        - CPU: 모델 그대로 반환 (passthrough)

        Args:
            model: PyTorch 모델 또는 모델 경로
            model_config: 모델 관련 설정 (배치 크기, 정밀도 등)

        Returns:
            준비된 모델 (백엔드에서 실행 가능한 형태)
        """
        ...

    @abstractmethod
    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        추론 실행

        Args:
            model: prepare_model()로 준비된 모델
            inputs: 입력 데이터
            **kwargs: 추가 실행 옵션

        Returns:
            추론 결과
        """
        ...

    @abstractmethod
    def get_device_info(self) -> DeviceInfo:
        """
        디바이스 정보 반환

        Returns:
            DeviceInfo 객체
        """
        ...

    def cleanup(self) -> None:
        """
        백엔드 정리

        리소스 해제, 메모리 정리 등
        """
        self._initialized = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device_id={self.device_id})"


class NPUBackend(Backend, ABC):
    """
    NPU 백엔드 공통 베이스 클래스

    NPU 특화 기능 (컴파일, 캐싱) 포함
    """

    def __init__(
        self,
        device_id: int = 0,
        cache_dir: Optional[str] = None,
        compile_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(device_id, **kwargs)
        self.cache_dir = cache_dir
        self.compile_options = compile_options or {}
        self._compiled_models: Dict[str, Any] = {}

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.NPU

    @abstractmethod
    def compile_model(self, model: Any, **kwargs) -> Any:
        """
        모델을 NPU용으로 컴파일

        Args:
            model: ONNX 모델 또는 PyTorch 모델
            **kwargs: 컴파일 옵션

        Returns:
            컴파일된 모델
        """
        ...

    @abstractmethod
    def load_compiled_model(self, path: str) -> Any:
        """
        컴파일된 모델 로드

        Args:
            path: 컴파일된 모델 파일 경로

        Returns:
            로드된 컴파일 모델
        """
        ...

    @abstractmethod
    def save_compiled_model(self, model: Any, path: str) -> None:
        """
        컴파일된 모델 저장

        Args:
            model: 컴파일된 모델
            path: 저장 경로
        """
        ...

    def get_cache_key(self, model: Any, config: Optional[Dict[str, Any]] = None) -> str:
        """
        모델 캐시 키 생성

        Args:
            model: 모델 객체
            config: 모델 설정

        Returns:
            캐시 키 문자열
        """
        import hashlib

        # 모델 이름/경로와 설정을 조합하여 해시 생성
        model_name = getattr(model, 'name_or_path', str(type(model).__name__))
        config_str = str(sorted(config.items())) if config else ""

        key_string = f"{model_name}_{self.name}_{config_str}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
