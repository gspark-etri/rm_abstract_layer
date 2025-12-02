"""
vLLM 기반 GPU 백엔드

GPU에서 vLLM을 사용한 고성능 LLM 추론 지원
"""

from typing import Any, Dict, Optional, List
import logging

from ...core.backend import Backend, DeviceType, DeviceInfo

logger = logging.getLogger(__name__)


class VLLMBackend(Backend):
    """
    vLLM 기반 GPU 백엔드

    vLLM의 Continuous Batching, Tensor Parallel 등 고급 기능 활용
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
        """CUDA 및 vLLM 사용 가능 여부 확인"""
        try:
            import torch
            if not torch.cuda.is_available():
                return False

            # vLLM 임포트 확인
            import vllm
            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        """GPU 백엔드 초기화"""
        if self._initialized:
            return

        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")

            # 지정된 GPU 설정
            torch.cuda.set_device(self.device_id)
            self._initialized = True
            logger.info(f"GPU backend initialized on cuda:{self.device_id}")

        except Exception as e:
            logger.error(f"Failed to initialize GPU backend: {e}")
            raise

    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        모델 준비 (vLLM 엔진 생성)

        Args:
            model: 모델 이름/경로 또는 PyTorch 모델
            model_config: vLLM 설정 옵션

        Returns:
            vLLM LLM 인스턴스
        """
        config = model_config or {}

        # 모델 이름 추출
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
        vLLM으로 추론 실행

        Args:
            model: vLLM LLM 인스턴스
            inputs: 프롬프트 문자열 또는 리스트
            **kwargs: SamplingParams 옵션

        Returns:
            생성된 텍스트
        """
        from vllm import SamplingParams

        # SamplingParams 설정
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            max_tokens=kwargs.get("max_tokens", 256),
        )

        # 입력 처리
        if isinstance(inputs, str):
            prompts = [inputs]
        elif isinstance(inputs, list):
            prompts = inputs
        else:
            prompts = [str(inputs)]

        # 추론 실행
        outputs = model.generate(prompts, sampling_params)

        return outputs

    def get_device_info(self) -> DeviceInfo:
        """GPU 디바이스 정보 반환"""
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
        """GPU 리소스 정리"""
        self._llm_engine = None
        self._model_name = None

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        super().cleanup()
