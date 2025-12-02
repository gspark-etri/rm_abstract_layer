"""
vLLM-based GPU Backend

High-performance LLM inference using vLLM on GPU
"""

from typing import Any, Dict, Optional, List
import logging

from ...core.backend import Backend, DeviceType, DeviceInfo

logger = logging.getLogger(__name__)


class VLLMBackend(Backend):
    """
    vLLM-based GPU Backend

    Utilizes vLLM's advanced features like Continuous Batching, Tensor Parallel, etc.
    """

    def __init__(self, device_id: int = 0, **kwargs):
        super().__init__(device_id, **kwargs)
        self._llm_engine = None
        self._model_name: Optional[str] = None

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.GPU

    @property
    def name(self) -> str:
        return "gpu"

    def is_available(self) -> bool:
        """Check CUDA and vLLM availability"""
        try:
            import torch

            if not torch.cuda.is_available():
                return False

            # Check vLLM import
            import vllm

            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        """Initialize GPU backend"""
        if self._initialized:
            return

        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")

            # Set specified GPU
            torch.cuda.set_device(self.device_id)
            self._initialized = True
            logger.info(f"GPU backend initialized on cuda:{self.device_id}")

        except Exception as e:
            logger.error(f"Failed to initialize GPU backend: {e}")
            raise

    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Prepare model (create vLLM engine)

        Args:
            model: Model name/path or PyTorch model
            model_config: vLLM configuration options

        Returns:
            vLLM LLM instance
        """
        config = model_config or {}

        # Extract model name
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "name_or_path"):
            model_name = model.name_or_path
        elif hasattr(model, "config") and hasattr(model.config, "name_or_path"):
            model_name = model.config.name_or_path
        else:
            raise ValueError("Cannot determine model name/path")

        try:
            from vllm import LLM

            self._llm_engine = LLM(
                model=model_name,
                tensor_parallel_size=config.get("tensor_parallel_size", 1),
                dtype=config.get("dtype", "auto"),
                gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
                trust_remote_code=config.get("trust_remote_code", True),
            )
            self._model_name = model_name

            logger.info(f"vLLM engine created for model: {model_name}")
            return self._llm_engine

        except Exception as e:
            logger.error(f"Failed to create vLLM engine: {e}")
            raise

    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute inference with vLLM

        Args:
            model: vLLM LLM instance
            inputs: Prompt string or list
            **kwargs: SamplingParams options

        Returns:
            Generated text
        """
        from vllm import SamplingParams

        # Configure SamplingParams
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            max_tokens=kwargs.get("max_tokens", 256),
        )

        # Process input
        if isinstance(inputs, str):
            prompts = [inputs]
        elif isinstance(inputs, list):
            prompts = inputs
        else:
            prompts = [str(inputs)]

        # Execute inference
        outputs = model.generate(prompts, sampling_params)

        return outputs

    def get_device_info(self) -> DeviceInfo:
        """Return GPU device information"""
        try:
            import torch

            props = torch.cuda.get_device_properties(self.device_id)
            return DeviceInfo(
                device_type=DeviceType.GPU,
                device_id=self.device_id,
                name=props.name,
                vendor="NVIDIA",
                memory_total=props.total_memory,
                memory_available=torch.cuda.memory_reserved(self.device_id),
                extra={
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                },
            )
        except Exception:
            return DeviceInfo(
                device_type=DeviceType.GPU,
                device_id=self.device_id,
                name="Unknown GPU",
            )

    def cleanup(self) -> None:
        """Cleanup GPU resources"""
        self._llm_engine = None
        self._model_name = None

        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

        super().cleanup()
