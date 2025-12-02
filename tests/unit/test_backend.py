"""Backend 추상 클래스 테스트"""

import pytest
from rm_abstract.core.backend import Backend, NPUBackend, DeviceType, DeviceInfo


class MockBackend(Backend):
    """테스트용 Mock 백엔드"""

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CPU

    @property
    def name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return True

    def initialize(self) -> None:
        self._initialized = True

    def prepare_model(self, model, model_config=None):
        return model

    def execute(self, model, inputs, **kwargs):
        return {"output": "mock_result"}

    def get_device_info(self) -> DeviceInfo:
        return DeviceInfo(
            device_type=DeviceType.CPU,
            device_id=0,
            name="MockDevice",
        )


class TestBackend:
    """Backend ABC 테스트"""

    def test_mock_backend_creation(self):
        """Mock 백엔드 생성 테스트"""
        backend = MockBackend(device_id=0)
        assert backend.device_id == 0
        assert backend.name == "mock"

    def test_backend_is_available(self):
        """is_available 메서드 테스트"""
        backend = MockBackend()
        assert backend.is_available() is True

    def test_backend_initialize(self):
        """initialize 메서드 테스트"""
        backend = MockBackend()
        assert backend._initialized is False
        backend.initialize()
        assert backend._initialized is True

    def test_backend_prepare_model(self):
        """prepare_model 메서드 테스트"""
        backend = MockBackend()
        model = {"type": "test_model"}
        prepared = backend.prepare_model(model)
        assert prepared == model

    def test_backend_execute(self):
        """execute 메서드 테스트"""
        backend = MockBackend()
        result = backend.execute(None, None)
        assert result == {"output": "mock_result"}

    def test_backend_get_device_info(self):
        """get_device_info 메서드 테스트"""
        backend = MockBackend()
        info = backend.get_device_info()
        assert info.device_type == DeviceType.CPU
        assert info.name == "MockDevice"

    def test_backend_cleanup(self):
        """cleanup 메서드 테스트"""
        backend = MockBackend()
        backend.initialize()
        assert backend._initialized is True
        backend.cleanup()
        assert backend._initialized is False


class TestDeviceInfo:
    """DeviceInfo 데이터클래스 테스트"""

    def test_device_info_creation(self):
        """DeviceInfo 생성 테스트"""
        info = DeviceInfo(
            device_type=DeviceType.GPU,
            device_id=0,
            name="NVIDIA A100",
            vendor="NVIDIA",
            memory_total=80 * 1024**3,
        )
        assert info.device_type == DeviceType.GPU
        assert info.device_id == 0
        assert info.name == "NVIDIA A100"
        assert info.vendor == "NVIDIA"

    def test_device_info_optional_fields(self):
        """DeviceInfo 선택적 필드 테스트"""
        info = DeviceInfo(
            device_type=DeviceType.NPU,
            device_id=0,
            name="ATOM",
        )
        assert info.vendor is None
        assert info.memory_total is None
        assert info.extra is None


class TestDeviceType:
    """DeviceType Enum 테스트"""

    def test_device_types(self):
        """디바이스 타입 값 테스트"""
        assert DeviceType.GPU.value == "gpu"
        assert DeviceType.NPU.value == "npu"
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.AUTO.value == "auto"
