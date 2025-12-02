"""
ModelProxy - Transparent Model Wrapper

Wraps model objects using the Proxy pattern to intercept all method calls
(generate, forward, __call__) and route them to the appropriate backend.

This enables "Zero Code Change" - existing code that calls model.generate()
will transparently be routed to NPU/GPU backends without modification.
"""

from typing import Any, Dict, Optional, Callable, Set
import logging

logger = logging.getLogger(__name__)


class ModelProxy:
    """
    Transparent Proxy for AI Models

    Wraps the original model and intercepts key methods to route
    execution to the configured backend.

    Features:
    - Intercepts generate(), forward(), __call__() methods
    - Passes through all other attributes to original model
    - Maintains compatibility with existing code
    - Supports both HuggingFace and PyTorch interfaces
    """

    # Methods to intercept and route to backend
    _INTERCEPTED_METHODS: Set[str] = {
        "generate",
        "forward",
        "__call__",
    }

    def __init__(
        self,
        original_model: Any,
        compiled_model: Any,
        backend: Any,
        controller: Any,
    ):
        """
        Args:
            original_model: The original PyTorch/HuggingFace model
            compiled_model: The prepared model for the backend (compiled NPU model, etc.)
            backend: The backend instance for execution
            controller: The DeviceFlowController instance
        """
        # Use object.__setattr__ to avoid triggering our __setattr__
        object.__setattr__(self, "_original", original_model)
        object.__setattr__(self, "_compiled", compiled_model)
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_controller", controller)
        object.__setattr__(
            self,
            "_proxy_attrs",
            {"_original", "_compiled", "_backend", "_controller", "_proxy_attrs"},
        )

        logger.debug(
            f"Created ModelProxy for {type(original_model).__name__} "
            f"with backend {type(backend).__name__}"
        )

    def generate(self, *args, **kwargs) -> Any:
        """
        Intercept generate() method for text generation models.

        This is the primary method used by HuggingFace LLMs for inference.
        """
        logger.debug(f"ModelProxy.generate() called")
        return self._execute_on_backend("generate", *args, **kwargs)

    def forward(self, *args, **kwargs) -> Any:
        """
        Intercept forward() method for PyTorch models.

        This is the standard PyTorch forward pass.
        """
        logger.debug(f"ModelProxy.forward() called")
        return self._execute_on_backend("forward", *args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Intercept __call__() for direct model invocation.

        model(inputs) is equivalent to model.forward(inputs) in PyTorch.
        """
        logger.debug(f"ModelProxy.__call__() called")
        return self._execute_on_backend("__call__", *args, **kwargs)

    def _execute_on_backend(self, method_name: str, *args, **kwargs) -> Any:
        """
        Execute the method on the backend.

        Args:
            method_name: Name of the intercepted method
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from backend execution
        """
        # Extract inputs from args/kwargs
        inputs = args[0] if args else kwargs.get("input_ids", kwargs.get("inputs", None))

        # Add method hint for backend
        kwargs["_proxy_method"] = method_name

        try:
            # Execute through backend
            result = self._backend.execute(
                self._compiled, inputs, original_model=self._original, **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Backend execution failed for {method_name}: {e}")
            # Optionally fall back to original model
            if hasattr(self._original, method_name):
                logger.warning(f"Falling back to original model.{method_name}()")
                original_method = getattr(self._original, method_name)
                return original_method(*args, **kwargs)
            raise

    def __getattr__(self, name: str) -> Any:
        """
        Pass through attribute access to the original model.

        This ensures compatibility with code that accesses model attributes
        like model.config, model.device, model.dtype, etc.
        """
        # Check if it's a proxy internal attribute first
        proxy_attrs = object.__getattribute__(self, "_proxy_attrs")
        if name in proxy_attrs:
            return object.__getattribute__(self, name)

        # Get original model
        original = object.__getattribute__(self, "_original")

        # Pass through to original model
        return getattr(original, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Pass through attribute setting to the original model.
        """
        proxy_attrs = object.__getattribute__(self, "_proxy_attrs")
        if name in proxy_attrs:
            object.__setattr__(self, name, value)
        else:
            original = object.__getattribute__(self, "_original")
            setattr(original, name, value)

    def __repr__(self) -> str:
        original = object.__getattribute__(self, "_original")
        backend = object.__getattribute__(self, "_backend")
        return f"ModelProxy({type(original).__name__}, " f"backend={type(backend).__name__})"

    def __str__(self) -> str:
        return self.__repr__()

    # Support for isinstance checks and type introspection
    @property
    def __class__(self):
        """
        Return the original model's class for isinstance() checks.

        This allows code like `isinstance(model, PreTrainedModel)` to work.
        """
        # Return proxy class - users who need original class can use unwrap()
        return type(self)

    def unwrap(self) -> Any:
        """
        Return the original unwrapped model.

        Useful when direct access to the original model is needed.
        """
        return object.__getattribute__(self, "_original")

    def get_compiled_model(self) -> Any:
        """
        Return the compiled/prepared model.

        Useful for debugging or direct backend access.
        """
        return object.__getattribute__(self, "_compiled")

    def get_backend(self) -> Any:
        """
        Return the backend instance.
        """
        return object.__getattribute__(self, "_backend")

    # Support for common model properties that should work transparently
    @property
    def config(self) -> Any:
        """Pass through config attribute."""
        return self._original.config

    @property
    def device(self) -> Any:
        """Pass through device attribute."""
        return getattr(self._original, "device", None)

    @property
    def dtype(self) -> Any:
        """Pass through dtype attribute."""
        return getattr(self._original, "dtype", None)

    # Support iteration (some models are iterable)
    def __iter__(self):
        return iter(self._original)

    # Support len() if original model supports it
    def __len__(self):
        return len(self._original)

    # Support pickling (for multiprocessing)
    def __getstate__(self):
        return {
            "original": self._original,
            "compiled": self._compiled,
            "backend": self._backend,
            "controller": self._controller,
        }

    def __setstate__(self, state):
        object.__setattr__(self, "_original", state["original"])
        object.__setattr__(self, "_compiled", state["compiled"])
        object.__setattr__(self, "_backend", state["backend"])
        object.__setattr__(self, "_controller", state["controller"])
        object.__setattr__(
            self,
            "_proxy_attrs",
            {"_original", "_compiled", "_backend", "_controller", "_proxy_attrs"},
        )


def create_model_proxy(
    original_model: Any,
    compiled_model: Any,
    backend: Any,
    controller: Any,
) -> ModelProxy:
    """
    Factory function to create a ModelProxy.

    Args:
        original_model: The original PyTorch/HuggingFace model
        compiled_model: The prepared model for the backend
        backend: The backend instance
        controller: The DeviceFlowController instance

    Returns:
        ModelProxy wrapping the model
    """
    return ModelProxy(
        original_model=original_model,
        compiled_model=compiled_model,
        backend=backend,
        controller=controller,
    )
