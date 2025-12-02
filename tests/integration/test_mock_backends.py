"""Mock 기반 백엔드 통합 테스트

실제 하드웨어 없이 백엔드 인터페이스 동작 검증
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys


class TestCPUBackend:
    """CPU 백엔드 테스트 (실제 동작)"""

    def test_cpu_backend_is_available(self):
        """CPU 백엔드 가용성 테스트"""
        from rm_abstract.backends.cpu.cpu_backend import CPUBackend

        backend = CPUBackend()
        # PyTorch가 설치되어 있으면 True
        assert backend.is_available() is True

    def test_cpu_backend_initialize(self):
        """CPU 백엔드 초기화 테스트"""
        from rm_abstract.backends.cpu.cpu_backend import CPUBackend

        backend = CPUBackend()
        backend.initialize()
        assert backend._initialized is True

    def test_cpu_backend_device_info(self):
        """CPU 백엔드 디바이스 정보 테스트"""
        from rm_abstract.backends.cpu.cpu_backend import CPUBackend
        from rm_abstract.core.backend import DeviceType

        backend = CPUBackend()
        info = backend.get_device_info()

        assert info.device_type == DeviceType.CPU
        assert info.device_id == 0


class TestGPUBackendMock:
    """GPU 백엔드 Mock 테스트"""

    def test_gpu_backend_not_available_without_cuda(self):
        """CUDA 없을 때 GPU 백엔드 비가용 테스트"""
        with patch.dict(sys.modules, {'vllm': None}):
            from rm_abstract.backends.gpu.vllm_backend import VLLMBackend

            backend = VLLMBackend()
            # vLLM import 실패 시 False
            assert backend.is_available() is False

    @patch('torch.cuda.is_available', return_value=False)
    def test_gpu_backend_no_cuda(self, mock_cuda):
        """CUDA 미지원 환경 테스트"""
        from rm_abstract.backends.gpu.vllm_backend import VLLMBackend

        backend = VLLMBackend()
        assert backend.is_available() is False


class TestNPUBackendMock:
    """NPU 백엔드 Mock 테스트"""

    def test_rebellions_backend_not_available_without_sdk(self):
        """Rebellions SDK 없을 때 테스트"""
        # rebel 모듈이 없으면 import 실패
        with patch.dict(sys.modules, {'rebel': None}):
            try:
                from rm_abstract.backends.npu.plugins.rebellions import RBLNBackend
                backend = RBLNBackend()
                assert backend.is_available() is False
            except ImportError:
                # SDK 없으면 import 자체가 실패할 수 있음
                pass

    def test_furiosa_backend_not_available_without_sdk(self):
        """Furiosa SDK 없을 때 테스트"""
        with patch.dict(sys.modules, {'furiosa': None, 'furiosa.runtime': None}):
            try:
                from rm_abstract.backends.npu.plugins.furiosa import FuriosaBackend
                backend = FuriosaBackend()
                assert backend.is_available() is False
            except ImportError:
                pass


class TestBackendRegistry:
    """백엔드 레지스트리 테스트"""

    def test_register_all_backends(self):
        """모든 백엔드 등록 테스트"""
        from rm_abstract.core.controller import DeviceFlowController
        from rm_abstract.backends.registry import register_all_backends

        # 레지스트리 초기화
        DeviceFlowController._backend_registry.clear()

        # 백엔드 등록
        register_all_backends()

        # CPU 백엔드는 항상 등록되어야 함
        assert "cpu" in DeviceFlowController._backend_registry


class TestInitFunction:
    """rm_abstract.init() 함수 테스트"""

    def test_init_with_cpu(self):
        """CPU로 초기화 테스트"""
        import rm_abstract
        from rm_abstract.core.controller import DeviceFlowController

        # 레지스트리 초기화 및 CPU 백엔드 등록
        DeviceFlowController._backend_registry.clear()
        from rm_abstract.backends.cpu.cpu_backend import CPUBackend
        DeviceFlowController.register_backend("cpu", CPUBackend)

        controller = rm_abstract.init(device="cpu", verbose=False)

        assert controller is not None
        assert controller.device_name == "cpu:0"

    def test_init_returns_controller(self):
        """init()이 컨트롤러를 반환하는지 테스트"""
        import rm_abstract
        from rm_abstract.core.controller import DeviceFlowController

        DeviceFlowController._backend_registry.clear()
        from rm_abstract.backends.cpu.cpu_backend import CPUBackend
        DeviceFlowController.register_backend("cpu", CPUBackend)

        controller = rm_abstract.init(device="cpu", verbose=False)

        assert isinstance(controller, DeviceFlowController)

    def test_get_controller(self):
        """get_controller() 테스트"""
        import rm_abstract
        from rm_abstract.core.controller import DeviceFlowController

        DeviceFlowController._backend_registry.clear()
        from rm_abstract.backends.cpu.cpu_backend import CPUBackend
        DeviceFlowController.register_backend("cpu", CPUBackend)

        rm_abstract.init(device="cpu", verbose=False)
        controller = rm_abstract.get_controller()

        assert controller is not None
