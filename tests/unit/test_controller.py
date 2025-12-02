"""DeviceFlowController tests"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from rm_abstract.core.controller import DeviceFlowController
from rm_abstract.core.config import Config
from rm_abstract.core.backend import Backend, DeviceType, DeviceInfo


class MockBackend(Backend):
    """Mock backend for testing"""

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
    """DeviceFlowController tests"""

    def setup_method(self):
        """Clear backend registry before each test"""
        DeviceFlowController._backend_registry.clear()

    def test_register_backend(self):
        """Test backend registration"""
        DeviceFlowController.register_backend("mock", MockBackend)
        assert "mock" in DeviceFlowController._backend_registry

    def test_get_available_backends(self):
        """Test getting available backends"""
        DeviceFlowController.register_backend("mock", MockBackend)
        available = DeviceFlowController.get_available_backends()
        assert "mock" in available
        assert available["mock"] is True

    def test_controller_creation_with_mock_backend(self):
        """Test controller creation"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        assert controller.backend is not None
        assert controller.backend.name == "mock"

    def test_controller_device_name(self):
        """Test device name"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        assert controller.device_name == "mock:0"

    def test_controller_prepare_model(self):
        """Test model preparation"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        model = "test_model"
        prepared = controller.prepare_model(model)

        assert prepared == "prepared_test_model"

    def test_controller_execute(self):
        """Test inference execution"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        result = controller.execute("model", "inputs")
        assert result == "executed_model_inputs"

    def test_controller_get_device_info(self):
        """Test device info retrieval"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        info = controller.get_device_info()
        assert info["device_type"] == "cpu"
        assert info["name"] == "MockDevice"

    def test_controller_switch_device(self):
        """Test device switching"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        assert controller.backend.device_id == 0

        controller.switch_device("mock:1")
        assert controller.backend.device_id == 1

    def test_controller_no_backend_available(self):
        """Test when no backend is available"""
        config = Config(device="nonexistent:0")

        with pytest.raises(ValueError):
            DeviceFlowController(config)

    def test_controller_auto_select(self):
        """Auto backend selection test"""
        # Register as 'cpu' since auto-select looks for specific names
        # in priority order: rbln, furiosa, gpu, cpu
        DeviceFlowController.register_backend("cpu", MockBackend)

        config = Config(device="auto")
        controller = DeviceFlowController(config)

        # cpu backend should be selected since it's in the priority list
        assert controller.backend is not None
        assert controller.backend.name == "mock"

    def test_controller_cleanup(self):
        """Test resource cleanup"""
        DeviceFlowController.register_backend("mock", MockBackend)

        config = Config(device="mock:0")
        controller = DeviceFlowController(config)

        controller.prepare_model("model1")
        assert len(controller._prepared_models) == 1

        controller.cleanup()
        assert len(controller._prepared_models) == 0
