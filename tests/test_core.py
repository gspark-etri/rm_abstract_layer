"""
Core functionality tests
"""

import pytest


class TestInitialization:
    """Test rm_abstract initialization"""
    
    def test_import(self):
        """Test that rm_abstract can be imported"""
        import rm_abstract
        assert rm_abstract is not None
    
    def test_version(self):
        """Test version is defined"""
        import rm_abstract
        assert hasattr(rm_abstract, "__version__")
        assert rm_abstract.__version__ is not None
    
    def test_init_auto(self):
        """Test auto device initialization"""
        import rm_abstract
        
        controller = rm_abstract.init(device="auto", verbose=False)
        assert controller is not None
    
    def test_init_cpu(self):
        """Test CPU device initialization"""
        import rm_abstract
        
        controller = rm_abstract.init(device="cpu", verbose=False)
        assert controller is not None
        assert "cpu" in controller.device_name.lower()
    
    def test_get_controller(self):
        """Test get_controller returns initialized controller"""
        import rm_abstract
        
        rm_abstract.init(device="cpu", verbose=False)
        controller = rm_abstract.get_controller()
        assert controller is not None


class TestDeviceSwitching:
    """Test device switching functionality"""
    
    def test_switch_to_cpu(self):
        """Test switching to CPU"""
        import rm_abstract
        
        rm_abstract.init(device="auto", verbose=False)
        rm_abstract.switch_device("cpu")
        
        controller = rm_abstract.get_controller()
        assert "cpu" in controller.device_name.lower()
    
    def test_get_device_info(self):
        """Test get_device_info returns valid info"""
        import rm_abstract
        
        rm_abstract.init(device="cpu", verbose=False)
        info = rm_abstract.get_device_info()
        
        assert isinstance(info, dict)
        assert "device_type" in info
        assert "device_id" in info


class TestBackends:
    """Test backend availability"""
    
    def test_get_available_backends(self):
        """Test getting available backends"""
        import rm_abstract
        
        backends = rm_abstract.get_available_backends()
        
        assert isinstance(backends, dict)
        assert "cpu" in backends
        # CPU should always be available
        assert backends["cpu"] == True
    
    def test_gpu_backend_detection(self):
        """Test GPU backend detection"""
        import rm_abstract
        
        backends = rm_abstract.get_available_backends()
        
        # Check GPU backend exists (may or may not be available)
        assert "gpu" in backends
        assert isinstance(backends["gpu"], bool)


class TestSystemInfo:
    """Test system information module"""
    
    def test_get_system_info(self):
        """Test get_system_info returns valid info"""
        import rm_abstract
        
        info = rm_abstract.get_system_info()
        
        assert info is not None
        assert hasattr(info, "gpus")
        assert hasattr(info, "cpu")
        assert hasattr(info, "backends")
    
    def test_get_quick_status(self):
        """Test get_quick_status returns dict"""
        import rm_abstract
        
        status = rm_abstract.get_quick_status()
        
        assert isinstance(status, dict)
        assert "cpu_backend" in status

