"""DeviceFlowController 테스트"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from rm_abstract.core.controller import DeviceFlowController
from rm_abstract.core.config import Config
from rm_abstract.core.backend import Backend, DeviceType, DeviceInfo


class MockBackend(Backend):
    """테스트용 Mock 백엔드"""

    def __init__(self, device_id=0, available=True, **kwargs):
        super().__init__(device_id, **kwargs)
        self._available = available

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.CPU

    @property
    def name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return self._available

    def initialize(self) -> None:
        self._initialized = True

    def prepare_model(self, model, model_config=None):
        return f"prepared_{model}"

    def execute(self, model, inputs, **kwargs):
        return f"executed_{model}_{inputs}"

    def get_device_info(self) -> DeviceInfo:
        return DeviceInfo(
            device_type=DeviceType.CPU,
            device_id=self.device_id,
            name="MockDevice",
        )


class TestDeviceFlowController:
    """DeviceFlowController 테스트"""

    def setup_method(self):
        """테스트 전 백엔드 레지스트리 초기화"""
        DeviceFlowController._backend_registry.clear()

    def test_register_backend(self):
        """백엔드 등록 테스트"""
        DeviceFlowController.register_backend("mock", MockBackend)
        assert "mock" in DeviceFlowController._backend_registry

    def test_get_available_backends(self):
        """사용 가능한 백엔드 조회 테스트"""
        DeviceFlowController.register_backend("mock", MockBackend)
        available = DeviceFlowController.get_available_backends()
        assert "mock" in available
        assert available["mock"] is True

    def test_controller_creation_with_mock_backend(self):
        """컨트롤러 생성 테스트"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        assert controller.backend is not None
        assert controller.backend.name == "mock"

    def test_controller_device_name(self):
        """디바이스 이름 테스트"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        assert controller.device_name == "mock:0"

    def test_controller_prepare_model(self):
        """모델 준비 테스트"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        model = "test_model"
        prepared = controller.prepare_model(model)

        assert prepared == "prepared_test_model"

    def test_controller_execute(self):
        """추론 실행 테스트"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        result = controller.execute("model", "inputs")
        assert result == "executed_model_inputs"

    def test_controller_get_device_info(self):
        """디바이스 정보 조회 테스트"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        info = controller.get_device_info()
        assert info["device_type"] == "cpu"
        assert info["name"] == "MockDevice"

    def test_controller_switch_device(self):
        """디바이스 전환 테스트"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        assert controller.backend.device_id == 0

        controller.switch_device("mock:1")
        assert controller.backend.device_id == 1

    def test_controller_no_backend_available(self):
        """백엔드 없을 때 테스트"""
        config = Config(device="nonexistent:0")

        with pytest.raises(ValueError):
            DeviceFlowController(config)

    def test_controller_auto_select(self):
        """자동 백엔드 선택 테스트"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="auto")
        controller = DeviceFlowController(config)

        # mock이 유일하게 등록된 백엔드이므로 선택되어야 함
        assert controller.backend is not None

    def test_controller_cleanup(self):
        """리소스 정리 테스트"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        controller.prepare_model("model1")
        assert len(controller._prepared_models) == 1

        controller.cleanup()
        assert len(controller._prepared_models) == 0
