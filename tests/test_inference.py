"""
Inference tests
"""

import pytest


class TestCPUInference:
    """Test CPU inference"""
    
    def test_load_model(self, test_model_name):
        """Test loading a model on CPU"""
        import rm_abstract
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        rm_abstract.init(device="cpu", verbose=False)
        
        tokenizer = AutoTokenizer.from_pretrained(test_model_name)
        model = AutoModelForCausalLM.from_pretrained(test_model_name)
        
        assert model is not None
        assert tokenizer is not None
    
    def test_generate_text(self, test_model_name, test_prompt):
        """Test text generation on CPU"""
        import rm_abstract
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        rm_abstract.init(device="cpu", verbose=False)
        
        tokenizer = AutoTokenizer.from_pretrained(test_model_name)
        model = AutoModelForCausalLM.from_pretrained(test_model_name)
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        assert len(generated_text) > len(test_prompt)
        assert test_prompt in generated_text


class TestGPUInference:
    """Test GPU inference (skipped if no GPU)"""
    
    @pytest.fixture
    def gpu_available(self):
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def test_gpu_backend_available(self, gpu_available):
        """Test GPU backend availability"""
        if not gpu_available:
            pytest.skip("No GPU available")
        
        import rm_abstract
        
        backends = rm_abstract.get_available_backends()
        assert backends.get("gpu", False)
    
    def test_init_gpu(self, gpu_available):
        """Test GPU initialization"""
        if not gpu_available:
            pytest.skip("No GPU available")
        
        import rm_abstract
        
        controller = rm_abstract.init(device="gpu:0", verbose=False)
        assert "gpu" in controller.device_name.lower()


class TestDeviceSwitchInference:
    """Test inference after device switching"""
    
    def test_switch_and_generate(self, test_model_name, test_prompt):
        """Test generation after switching devices"""
        import rm_abstract
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Start with auto (may be GPU or CPU)
        rm_abstract.init(device="auto", verbose=False)
        
        # Switch to CPU
        rm_abstract.switch_device("cpu")
        
        # Load and generate
        tokenizer = AutoTokenizer.from_pretrained(test_model_name)
        model = AutoModelForCausalLM.from_pretrained(test_model_name)
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert len(generated_text) > len(test_prompt)

