"""
DeviceFlowController - 디바이스 플로우 컨트롤러

디바이스 선택, 백엔드 관리, 모델 후킹을 담당하는 핵심 컨트롤러
"""

from typing import Any, Dict, Optional, Type
import logging

from .backend import Backend, DeviceType, DeviceInfo
from .config import Config

logger = logging.getLogger(__name__)


class DeviceFlowController:
    """
    디바이스 플로우 컨트롤러

    - 디바이스 문자열 파싱 및 백엔드 선택
    - 모델 자동 후킹 관리
    - 백엔드 간 전환 지원
    """

    # 등록된 백엔드 클래스
    _backend_registry: Dict[str, Type[Backend]] = {}

    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self.config = config
        self._backend: Optional[Backend] = None
        self._hooks_activated = False
        self._prepared_models: Dict[int, Any] = {}

        # 백엔드 초기화
        self._initialize_backend()

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[Backend]) -> None:
        """
        백엔드 등록

        Args:
            name: 백엔드 이름 (예: 'gpu', 'rbln', 'furiosa')
            backend_class: Backend 클래스
        """
        cls._backend_registry[name.lower()] = backend_class

    @classmethod
    def get_available_backends(cls) -> Dict[str, bool]:
        """
        사용 가능한 백엔드 목록 반환

        Returns:
            {백엔드명: 사용가능여부} 딕셔너리
        """
        result = {}
        for name, backend_class in cls._backend_registry.items():
            try:
                backend = backend_class(device_id=0)
                result[name] = backend.is_available()
            except Exception:
                result[name] = False
        return result

    def _initialize_backend(self) -> None:
        """디바이스에 맞는 백엔드 초기화"""
        device_type = self.config.device_type
        device_id = self.config.device_id

        if device_type == "auto":
            self._backend = self._auto_select_backend()
        else:
            self._backend = self._create_backend(device_type, device_id)

        if self._backend is not None:
            self._backend.initialize()

    def _auto_select_backend(self) -> Optional[Backend]:
        """자동으로 최적의 백엔드 선택 (NPU > GPU > CPU)"""
        # 우선순위: NPU > GPU > CPU
        priority_order = ["rbln", "furiosa", "gpu", "cpu"]

        for backend_name in priority_order:
            if backend_name in self._backend_registry:
                try:
                    backend = self._create_backend(backend_name, 0)
                    if backend and backend.is_available():
                        logger.info(f"Auto-selected backend: {backend_name}")
                        return backend
                except Exception as e:
                    logger.debug(f"Backend {backend_name} not available: {e}")

        logger.warning("No backend available")
        return None

    def _create_backend(self, device_type: str, device_id: int) -> Optional[Backend]:
        """백엔드 인스턴스 생성"""
        backend_class = self._backend_registry.get(device_type.lower())

        if backend_class is None:
            raise ValueError(f"Unknown device type: {device_type}")

        # NPU 백엔드인 경우 추가 옵션 전달
        if device_type in ["rbln", "furiosa"]:
            return backend_class(
                device_id=device_id,
                cache_dir=self.config.cache_dir,
                compile_options=self.config.compile_options,
            )
        else:
            return backend_class(device_id=device_id)

    @property
    def backend(self) -> Optional[Backend]:
        """현재 백엔드 반환"""
        return self._backend

    @property
    def device_name(self) -> str:
        """현재 디바이스 이름 반환"""
        if self._backend is None:
            return "none"
        return f"{self._backend.name}:{self._backend.device_id}"

    def switch_device(self, device: str) -> None:
        """
        런타임에 디바이스 전환

        Args:
            device: 새로운 디바이스 지정
        """
        # 기존 백엔드 정리
        if self._backend:
            self._backend.cleanup()

        # 새 설정으로 업데이트
        self.config.device = device
        self._prepared_models.clear()

        # 새 백엔드 초기화
        self._initialize_backend()

        logger.info(f"Switched to device: {self.device_name}")

    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        모델 준비 (백엔드에 맞게 변환/컴파일)

        Args:
            model: 원본 모델
            model_config: 모델 설정

        Returns:
            준비된 모델
        """
        if self._backend is None:
            logger.warning("No backend available, returning original model")
            return model

        model_id = id(model)

        # 이미 준비된 모델인지 확인
        if model_id in self._prepared_models:
            return self._prepared_models[model_id]

        # 백엔드에서 모델 준비
        prepared = self._backend.prepare_model(model, model_config)
        self._prepared_models[model_id] = prepared

        return prepared

    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        추론 실행

        Args:
            model: 준비된 모델
            inputs: 입력 데이터
            **kwargs: 추가 옵션

        Returns:
            추론 결과
        """
        if self._backend is None:
            raise RuntimeError("No backend available")

        return self._backend.execute(model, inputs, **kwargs)

    def should_intercept(self, model: Any) -> bool:
        """
        해당 모델의 호출을 가로챌지 결정

        Args:
            model: 모델 객체

        Returns:
            가로채야 하면 True
        """
        # 이미 준비된 모델이면 가로채기
        return id(model) in self._prepared_models

    def get_device_info(self) -> Dict[str, Any]:
        """현재 디바이스 정보 반환"""
        if self._backend is None:
            return {"status": "no_backend"}

        info = self._backend.get_device_info()
        return {
            "device_type": info.device_type.value,
            "device_id": info.device_id,
            "name": info.name,
            "vendor": info.vendor,
            "memory_total": info.memory_total,
            "memory_available": info.memory_available,
            "extra": info.extra,
        }

    def activate_hooks(self) -> None:
        """후킹 시스템 활성화"""
        if self._hooks_activated:
            return

        from ..hooks import activate_all_hooks
        activate_all_hooks(self)
        self._hooks_activated = True

    def deactivate_hooks(self) -> None:
        """후킹 시스템 비활성화"""
        if not self._hooks_activated:
            return

        from ..hooks import deactivate_all_hooks
        deactivate_all_hooks()
        self._hooks_activated = False

    def cleanup(self) -> None:
        """리소스 정리"""
        self.deactivate_hooks()
        if self._backend:
            self._backend.cleanup()
        self._prepared_models.clear()
