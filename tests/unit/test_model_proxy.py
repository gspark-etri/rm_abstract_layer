"""
Tests for ModelProxy class
"""

import pytest
from unittest.mock import MagicMock, patch


class TestModelProxy:
    """Test cases for ModelProxy"""

    def test_proxy_creation(self):
        """Test ModelProxy can be created"""
        from rm_abstract.core.model_proxy import ModelProxy

        # Create mock objects
        original_model = MagicMock()
        compiled_model = MagicMock()
        backend = MagicMock()
        controller = MagicMock()

        # Create proxy
        proxy = ModelProxy(
            original_model=original_model,
            compiled_model=compiled_model,
            backend=backend,
            controller=controller,
        )

        assert proxy is not None
        assert proxy.unwrap() is original_model
        assert proxy.get_compiled_model() is compiled_model
        assert proxy.get_backend() is backend

    def test_proxy_generate_intercepts(self):
        """Test that generate() is intercepted and routed to backend"""
        from rm_abstract.core.model_proxy import ModelProxy

        # Create mocks
        original_model = MagicMock()
        compiled_model = MagicMock()
        backend = MagicMock()
        backend.execute.return_value = "generated output"
        controller = MagicMock()

        proxy = ModelProxy(
            original_model=original_model,
            compiled_model=compiled_model,
            backend=backend,
            controller=controller,
        )

        # Call generate
        result = proxy.generate("test input", max_tokens=100)

        # Verify backend.execute was called
        backend.execute.assert_called_once()
        call_args = backend.execute.call_args
        assert call_args[0][0] is compiled_model
        assert call_args[1]["_proxy_method"] == "generate"
        assert call_args[1]["original_model"] is original_model
        assert call_args[1]["max_tokens"] == 100

    def test_proxy_forward_intercepts(self):
        """Test that forward() is intercepted and routed to backend"""
        from rm_abstract.core.model_proxy import ModelProxy

        original_model = MagicMock()
        compiled_model = MagicMock()
        backend = MagicMock()
        backend.execute.return_value = MagicMock()
        controller = MagicMock()

        proxy = ModelProxy(
            original_model=original_model,
            compiled_model=compiled_model,
            backend=backend,
            controller=controller,
        )

        # Call forward
        proxy.forward("test input")

        # Verify backend.execute was called with forward method
        backend.execute.assert_called_once()
        call_args = backend.execute.call_args
        assert call_args[1]["_proxy_method"] == "forward"

    def test_proxy_call_intercepts(self):
        """Test that __call__() is intercepted and routed to backend"""
        from rm_abstract.core.model_proxy import ModelProxy

        original_model = MagicMock()
        compiled_model = MagicMock()
        backend = MagicMock()
        backend.execute.return_value = MagicMock()
        controller = MagicMock()

        proxy = ModelProxy(
            original_model=original_model,
            compiled_model=compiled_model,
            backend=backend,
            controller=controller,
        )

        # Call proxy directly
        proxy("test input")

        # Verify backend.execute was called with __call__ method
        backend.execute.assert_called_once()
        call_args = backend.execute.call_args
        assert call_args[1]["_proxy_method"] == "__call__"

    def test_proxy_attribute_passthrough(self):
        """Test that attributes are passed through to original model"""
        from rm_abstract.core.model_proxy import ModelProxy

        # Create mock with config attribute
        original_model = MagicMock()
        original_model.config.hidden_size = 768
        original_model.device = "cpu"
        original_model.dtype = "float32"

        proxy = ModelProxy(
            original_model=original_model,
            compiled_model=MagicMock(),
            backend=MagicMock(),
            controller=MagicMock(),
        )

        # Access attributes through proxy
        assert proxy.config.hidden_size == 768

    def test_proxy_repr(self):
        """Test proxy string representation"""
        from rm_abstract.core.model_proxy import ModelProxy

        class MockModel:
            pass

        class MockBackend:
            pass

        proxy = ModelProxy(
            original_model=MockModel(),
            compiled_model=MagicMock(),
            backend=MockBackend(),
            controller=MagicMock(),
        )

        repr_str = repr(proxy)
        assert "ModelProxy" in repr_str
        assert "MockModel" in repr_str
        assert "MockBackend" in repr_str

    def test_proxy_fallback_on_error(self):
        """Test that proxy falls back to original model on backend error"""
        from rm_abstract.core.model_proxy import ModelProxy

        original_model = MagicMock()
        original_model.generate.return_value = "fallback result"

        backend = MagicMock()
        backend.execute.side_effect = Exception("Backend error")

        proxy = ModelProxy(
            original_model=original_model,
            compiled_model=MagicMock(),
            backend=backend,
            controller=MagicMock(),
        )

        # Should fall back to original model
        result = proxy.generate("test")
        assert result == "fallback result"
        original_model.generate.assert_called_once()


class TestCreateModelProxy:
    """Test cases for create_model_proxy factory function"""

    def test_factory_creates_proxy(self):
        """Test that factory function creates ModelProxy correctly"""
        from rm_abstract.core.model_proxy import create_model_proxy, ModelProxy

        proxy = create_model_proxy(
            original_model=MagicMock(),
            compiled_model=MagicMock(),
            backend=MagicMock(),
            controller=MagicMock(),
        )

        assert isinstance(proxy, ModelProxy)
