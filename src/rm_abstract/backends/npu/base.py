"""
NPU 백엔드 공통 베이스 클래스

NPU 특화 기능 (컴파일, 캐싱, ONNX 변환) 포함
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
    NPU 백엔드 공통 베이스 클래스

    모든 NPU 벤더 백엔드가 상속받아 구현
    """

    def __init__(
        self,
        device_id: int = 0,
        cache_dir: Optional[str] = None,
        compile_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(device_id, cache_dir, compile_options, **kwargs)

        # 캐시 디렉토리 설정
        if self.cache_dir is None:
            self.cache_dir = os.path.join(Path.home(), ".rm_abstract", "cache", self.name)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        모델 준비 (ONNX 변환 → NPU 컴파일)

        1. 캐시 확인
        2. 캐시 없으면 ONNX 변환 → NPU 컴파일 → 캐시 저장
        3. 컴파일된 모델 반환

        Args:
            model: PyTorch 모델 또는 모델 경로
            model_config: 모델 설정

        Returns:
            NPU에서 실행 가능한 컴파일된 모델
        """
        config = model_config or {}
        cache_key = self.get_cache_key(model, config)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.{self.compiled_model_extension}")

        # 1. 캐시 확인
        if os.path.exists(cache_path):
            logger.info(f"Loading compiled model from cache: {cache_path}")
            return self.load_compiled_model(cache_path)

        # 2. ONNX 변환
        logger.info("Converting model to ONNX...")
        onnx_path = self._convert_to_onnx(model, config)

        # 3. NPU 컴파일
        logger.info(f"Compiling model for {self.name}...")
        compiled_model = self.compile_model(onnx_path, **self.compile_options)

        # 4. 캐시 저장
        logger.info(f"Saving compiled model to cache: {cache_path}")
        self.save_compiled_model(compiled_model, cache_path)

        return compiled_model

    @property
    @abstractmethod
    def compiled_model_extension(self) -> str:
        """컴파일된 모델 파일 확장자 (예: 'rbln', 'enf')"""
        ...

    def _convert_to_onnx(self, model: Any, config: Dict[str, Any]) -> str:
        """
        PyTorch 모델을 ONNX로 변환

        Args:
            model: PyTorch 모델
            config: 변환 설정

        Returns:
            ONNX 파일 경로
        """
        import torch

        cache_key = self.get_cache_key(model, config)
        onnx_path = os.path.join(self.cache_dir, f"{cache_key}.onnx")

        # 이미 ONNX 파일이 있으면 재사용
        if os.path.exists(onnx_path):
            logger.debug(f"Using existing ONNX file: {onnx_path}")
            return onnx_path

        # 더미 입력 생성
        batch_size = config.get("batch_size", 1)
        seq_length = config.get("seq_length", 128)

        # 모델 타입에 따라 더미 입력 생성
        if hasattr(model, "config"):
            vocab_size = getattr(model.config, "vocab_size", 32000)
        else:
            vocab_size = 32000

        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))

        # ONNX 내보내기
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
        NPU에서 추론 실행

        Args:
            model: 컴파일된 NPU 모델
            inputs: 입력 데이터
            **kwargs: 추가 옵션

        Returns:
            추론 결과
        """
        return self._execute_on_npu(model, inputs, **kwargs)

    @abstractmethod
    def _execute_on_npu(self, model: Any, inputs: Any, **kwargs) -> Any:
        """벤더별 NPU 추론 실행"""
        ...
