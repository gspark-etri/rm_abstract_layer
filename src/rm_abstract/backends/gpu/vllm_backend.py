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
            inputs: Prompt string, list, or token IDs (can be None if in kwargs)
            **kwargs: SamplingParams options (may include proxy metadata)

        Returns:
            Generated text
        """
        from vllm import SamplingParams

        # Extract proxy metadata (set by ModelProxy)
        proxy_method = kwargs.pop("_proxy_method", None)
        original_model = kwargs.pop("original_model", None)

        # Handle do_sample parameter (HuggingFace style)
        do_sample = kwargs.pop("do_sample", True)
        temperature = kwargs.get("temperature", 0.7 if do_sample else 0.0)
        if not do_sample:
            temperature = 0.0

        # Configure SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=kwargs.get("top_p", 1.0 if not do_sample else 0.95),
            max_tokens=kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256)),
        )

        # Process input - handle various input formats
        prompts = self._prepare_inputs(inputs, original_model, **kwargs)

        # Execute inference
        outputs = model.generate(prompts, sampling_params)

        # For generate() calls, return in a format compatible with HuggingFace
        if proxy_method == "generate":
            return self._convert_to_hf_format(outputs, inputs, original_model, **kwargs)

        return outputs

    def _prepare_inputs(
        self, inputs: Any, original_model: Any = None, **kwargs
    ) -> list:
        """
        Prepare inputs for vLLM.

        Handles various input formats:
        - String prompts
        - List of prompts
        - Token IDs (from tokenizer)
        - Dict with input_ids
        - None (inputs in kwargs)

        Args:
            inputs: Input data in various formats (can be None)
            original_model: Original HF model (for tokenizer access)
            **kwargs: May contain input_ids if inputs is None

        Returns:
            List of prompt strings
        """
        # If inputs is None, try to get from kwargs
        if inputs is None:
            inputs = kwargs.get("input_ids", kwargs.get("inputs", None))
            if inputs is None:
                raise ValueError("No inputs provided")

        # Already a string prompt
        if isinstance(inputs, str):
            return [inputs]

        # List of strings
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            return inputs

        # Token IDs - need to decode back to text
        # This happens when user code uses tokenizer + model.generate()
        if hasattr(inputs, "input_ids"):
            token_ids = inputs.input_ids
        elif isinstance(inputs, dict) and "input_ids" in inputs:
            token_ids = inputs["input_ids"]
        else:
            token_ids = inputs

        # Try to decode token IDs to text
        try:
            import torch

            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()

            # Try to get tokenizer from original model or use a default
            tokenizer = self._get_tokenizer(original_model)

            if tokenizer is not None:
                # Decode token IDs to text
                if isinstance(token_ids[0], list):
                    # Batch of sequences
                    return [
                        tokenizer.decode(ids, skip_special_tokens=False)
                        for ids in token_ids
                    ]
                else:
                    # Single sequence
                    return [tokenizer.decode(token_ids, skip_special_tokens=False)]

        except Exception as e:
            logger.warning(f"Failed to decode token IDs: {e}")

        # Fallback: convert to string
        return [str(inputs)]

    def _get_tokenizer(self, original_model: Any = None):
        """Get tokenizer from original model or model name."""
        tokenizer = None

        if original_model is not None:
            # Try various ways to get tokenizer
            if hasattr(original_model, "tokenizer"):
                tokenizer = original_model.tokenizer
            elif hasattr(original_model, "config"):
                model_name = getattr(original_model.config, "name_or_path", None)
                if model_name:
                    try:
                        from transformers import AutoTokenizer

                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                    except Exception:
                        pass

        # Fallback to stored model name
        if tokenizer is None and self._model_name:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            except Exception:
                pass

        return tokenizer

    def _convert_to_hf_format(
        self,
        vllm_outputs: Any,
        original_inputs: Any,
        original_model: Any = None,
        **kwargs,
    ) -> Any:
        """
        Convert vLLM outputs to HuggingFace-compatible format.

        When code expects model.generate() to return token IDs,
        we need to tokenize the output text.

        Args:
            vllm_outputs: vLLM RequestOutput objects
            original_inputs: Original inputs (to detect expected format)
            original_model: Original HF model (for tokenizer access)
            **kwargs: Additional parameters

        Returns:
            Token IDs tensor if input was tokens, otherwise vLLM outputs
        """
        # If input was string, return vLLM outputs as-is
        if isinstance(original_inputs, str):
            return vllm_outputs

        # If input was token IDs, convert output text back to token IDs
        try:
            import torch

            # Get tokenizer
            tokenizer = self._get_tokenizer(original_model)
            if tokenizer is None:
                return vllm_outputs

            # Extract generated text from vLLM outputs (including prompt)
            all_token_ids = []
            for output in vllm_outputs:
                # vLLM output includes prompt + generated tokens
                prompt_token_ids = output.prompt_token_ids
                for completion in output.outputs:
                    # Combine prompt + generated token IDs
                    generated_token_ids = list(prompt_token_ids) + list(
                        completion.token_ids
                    )
                    all_token_ids.append(generated_token_ids)

            # Convert to tensor
            if len(all_token_ids) == 1:
                return torch.tensor([all_token_ids[0]])
            else:
                # Pad sequences to same length
                max_len = max(len(ids) for ids in all_token_ids)
                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
                padded = [
                    ids + [pad_token_id] * (max_len - len(ids)) for ids in all_token_ids
                ]
                return torch.tensor(padded)

        except Exception as e:
            logger.warning(f"Failed to convert to HF format: {e}")
            return vllm_outputs

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
