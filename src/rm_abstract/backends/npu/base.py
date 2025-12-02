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
            **kwargs: Additional options

        Returns:
            Inference result
        """
        return self._execute_on_npu(model, inputs, **kwargs)

    @abstractmethod
    def _execute_on_npu(self, model: Any, inputs: Any, **kwargs) -> Any:
        """Execute NPU inference (vendor-specific)"""
        ...
