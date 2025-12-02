"""
NPU Backend Common Base Class

Includes NPU-specific features (compilation, caching, ONNX conversion)
"""

from abc import abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import logging
import os

from ...core.backend import NPUBackend, DeviceType, DeviceInfo

logger = logging.getLogger(__name__)


class NPUBackendBase(NPUBackend):
    """
    NPU Backend Common Base Class

    All NPU vendor backends inherit and implement this class.
    """

    def __init__(
        self,
        device_id: int = 0,
        cache_dir: Optional[str] = None,
        compile_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(device_id, cache_dir, compile_options, **kwargs)

        # Set cache directory
        if self.cache_dir is None:
            self.cache_dir = os.path.join(Path.home(), ".rm_abstract", "cache", self.name)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Prepare model (ONNX conversion → NPU compilation)

        1. Check cache
        2. If no cache: ONNX conversion → NPU compilation → save to cache
        3. Return compiled model

        Args:
            model: PyTorch model or model path
            model_config: Model configuration

        Returns:
            Compiled model executable on NPU
        """
        config = model_config or {}
        cache_key = self.get_cache_key(model, config)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.{self.compiled_model_extension}")

        # 1. Check cache
        if os.path.exists(cache_path):
            logger.info(f"Loading compiled model from cache: {cache_path}")
            return self.load_compiled_model(cache_path)

        # 2. Convert to ONNX
        logger.info("Converting model to ONNX...")
        onnx_path = self._convert_to_onnx(model, config)

        # 3. Compile for NPU
        logger.info(f"Compiling model for {self.name}...")
        compiled_model = self.compile_model(onnx_path, **self.compile_options)

        # 4. Save to cache
        logger.info(f"Saving compiled model to cache: {cache_path}")
        self.save_compiled_model(compiled_model, cache_path)

        return compiled_model

    @property
    @abstractmethod
    def compiled_model_extension(self) -> str:
        """Compiled model file extension (e.g., 'rbln', 'enf')"""
        ...

    def _convert_to_onnx(self, model: Any, config: Dict[str, Any]) -> str:
        """
        Convert PyTorch model to ONNX

        Args:
            model: PyTorch model
            config: Conversion configuration

        Returns:
            ONNX file path
        """
        import torch

        cache_key = self.get_cache_key(model, config)
        onnx_path = os.path.join(self.cache_dir, f"{cache_key}.onnx")

        # Reuse if ONNX file already exists
        if os.path.exists(onnx_path):
            logger.debug(f"Using existing ONNX file: {onnx_path}")
            return onnx_path

        # Create dummy input
        batch_size = config.get("batch_size", 1)
        seq_length = config.get("seq_length", 128)

        # Create dummy input based on model type
        if hasattr(model, "config"):
            vocab_size = getattr(model.config, "vocab_size", 32000)
        else:
            vocab_size = 32000

        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input_ids"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "output": {0: "batch_size", 1: "sequence"},
            },
            opset_version=config.get("opset_version", 14),
        )

        logger.info(f"ONNX model exported to: {onnx_path}")
        return onnx_path

    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute inference on NPU

        Args:
            model: Compiled NPU model
            inputs: Input data
            **kwargs: Additional options (may include proxy metadata)

        Returns:
            Inference result
        """
        # Extract proxy metadata (set by ModelProxy)
        proxy_method = kwargs.pop("_proxy_method", None)
        original_model = kwargs.pop("original_model", None)

        # Handle different method types from proxy
        if proxy_method == "generate":
            return self._execute_generate(model, inputs, original_model, **kwargs)
        elif proxy_method in ("forward", "__call__"):
            return self._execute_on_npu(model, inputs, **kwargs)
        else:
            # Default to standard NPU execution
            return self._execute_on_npu(model, inputs, **kwargs)

    def _execute_generate(self, model: Any, inputs: Any, original_model: Any, **kwargs) -> Any:
        """
        Execute generate() for text generation models.

        For NPU, this typically involves:
        1. Tokenization (using original model's tokenizer)
        2. Iterative token generation on NPU
        3. Decoding output tokens

        Args:
            model: Compiled NPU model
            inputs: Input data (token IDs or text)
            original_model: Original HuggingFace model (for tokenizer access)
            **kwargs: Generation parameters (max_length, temperature, etc.)
        """
        import torch

        # Extract generation parameters
        max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_length", 128))

        # Get input as tensor
        if hasattr(inputs, "input_ids"):
            input_ids = inputs.input_ids
        elif isinstance(inputs, dict) and "input_ids" in inputs:
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs

        # Ensure tensor type
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)

        # Simple greedy generation loop on NPU
        generated = input_ids
        for _ in range(max_new_tokens):
            # Run single forward pass on NPU
            outputs = self._execute_on_npu(model, generated, **kwargs)

            # Get next token (greedy: argmax of last position)
            if hasattr(outputs, "logits"):
                next_token_logits = outputs.logits[:, -1, :]
            elif isinstance(outputs, dict) and "logits" in outputs:
                next_token_logits = outputs["logits"][:, -1, :]
            else:
                # Assume outputs is raw logits tensor
                if hasattr(outputs, "shape") and len(outputs.shape) >= 2:
                    next_token_logits = outputs[:, -1, :]
                else:
                    next_token_logits = outputs

            if not isinstance(next_token_logits, torch.Tensor):
                next_token_logits = torch.tensor(next_token_logits)

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS token
            if original_model is not None and hasattr(original_model, "config"):
                eos_token_id = getattr(original_model.config, "eos_token_id", None)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

        return generated

    @abstractmethod
    def _execute_on_npu(self, model: Any, inputs: Any, **kwargs) -> Any:
        """Execute NPU inference (vendor-specific)"""
        ...
