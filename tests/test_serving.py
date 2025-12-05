"""
Serving engines tests
"""

import pytest


class TestServingEngines:
    """Test serving engine availability"""
    
    def test_get_available_engines(self):
        """Test getting available serving engines"""
        from rm_abstract.serving import get_available_engines
        
        engines = get_available_engines()
        
        assert isinstance(engines, dict)
        assert "vllm" in engines
        assert "triton" in engines
        assert "torchserve" in engines
    
    def test_serving_config(self):
        """Test ServingConfig creation"""
        from rm_abstract.serving import ServingConfig, ServingEngineType, DeviceTarget
        
        config = ServingConfig(
            engine=ServingEngineType.VLLM,
            device=DeviceTarget.GPU,
            model_name="gpt2",
        )
        
        assert config.engine == ServingEngineType.VLLM
        assert config.device == DeviceTarget.GPU
        assert config.model_name == "gpt2"


class TestVLLMEngine:
    """Test vLLM serving engine"""
    
    @pytest.fixture
    def vllm_available(self):
        """Check if vLLM is available"""
        try:
            import vllm
            return True
        except ImportError:
            return False
    
    def test_create_vllm_engine(self, vllm_available):
        """Test creating vLLM engine"""
        if not vllm_available:
            pytest.skip("vLLM not installed")
        
        from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
        
        config = ServingConfig(
            engine=ServingEngineType.VLLM,
            device=DeviceTarget.GPU,
        )
        
        engine = create_serving_engine(config)
        assert engine is not None
        assert engine.name == "vLLM"


class TestTritonEngine:
    """Test Triton serving engine"""
    
    @pytest.fixture
    def triton_available(self):
        """Check if Triton client is available"""
        try:
            import tritonclient.http
            return True
        except ImportError:
            return False
    
    def test_create_triton_engine(self, triton_available):
        """Test creating Triton engine"""
        if not triton_available:
            pytest.skip("Triton client not installed")
        
        from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
        
        config = ServingConfig(
            engine=ServingEngineType.TRITON,
            device=DeviceTarget.GPU,
        )
        
        engine = create_serving_engine(config)
        assert engine is not None
        assert engine.name == "Triton"
    
    def test_triton_config_generation(self, triton_available, tmp_path):
        """Test Triton config generation"""
        if not triton_available:
            pytest.skip("Triton client not installed")
        
        from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
        import os
        
        config = ServingConfig(
            engine=ServingEngineType.TRITON,
            device=DeviceTarget.GPU,
        )
        
        engine = create_serving_engine(config)
        engine.setup_model_repository(str(tmp_path))
        engine.load_model("gpt2", triton_model_name="test_model")
        
        config_path = tmp_path / "test_model" / "config.pbtxt"
        assert config_path.exists()


class TestTorchServeEngine:
    """Test TorchServe engine"""
    
    @pytest.fixture
    def torchserve_available(self):
        """Check if TorchServe is available"""
        try:
            import subprocess
            result = subprocess.run(
                ["torch-model-archiver", "--help"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except:
            return False
    
    def test_create_torchserve_engine(self, torchserve_available):
        """Test creating TorchServe engine"""
        if not torchserve_available:
            pytest.skip("TorchServe not installed")
        
        from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
        
        config = ServingConfig(
            engine=ServingEngineType.TORCHSERVE,
            device=DeviceTarget.GPU,
        )
        
        engine = create_serving_engine(config)
        assert engine is not None
        assert engine.name == "TorchServe"
    
    def test_mar_creation(self, torchserve_available, tmp_path):
        """Test .mar file creation"""
        if not torchserve_available:
            pytest.skip("TorchServe not installed")
        
        from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
        import os
        
        config = ServingConfig(
            engine=ServingEngineType.TORCHSERVE,
            device=DeviceTarget.GPU,
        )
        
        engine = create_serving_engine(config)
        engine._model_store = str(tmp_path)
        
        mar_path = engine.create_model_archive("gpt2", archive_name="test_model")
        
        assert mar_path is not None
        assert os.path.exists(mar_path)
        assert mar_path.endswith(".mar")

