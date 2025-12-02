"""ONNX 변환 유틸리티"""

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def convert_to_onnx(
    model: Any,
    output_path: str,
    input_shape: Optional[Tuple[int, ...]] = None,
    opset_version: int = 14,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> str:
    """
    PyTorch 모델을 ONNX로 변환

    Args:
        model: PyTorch 모델
        output_path: 출력 ONNX 파일 경로
        input_shape: 입력 텐서 shape (기본값: (1, 128))
        opset_version: ONNX opset 버전
        dynamic_axes: 동적 축 설정

    Returns:
        ONNX 파일 경로
    """
    import torch

    if input_shape is None:
        input_shape = (1, 128)

    # 더미 입력 생성
    if hasattr(model, "config"):
        vocab_size = getattr(model.config, "vocab_size", 32000)
    else:
        vocab_size = 32000

    dummy_input = torch.randint(0, vocab_size, input_shape)

    # 기본 동적 축 설정
    if dynamic_axes is None:
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence"},
            "output": {0: "batch_size", 1: "sequence"},
        }

    # 출력 디렉토리 생성
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ONNX 내보내기
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info(f"Model exported to ONNX: {output_path}")
    return output_path


def optimize_onnx(
    input_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    ONNX 모델 최적화

    Args:
        input_path: 입력 ONNX 파일 경로
        output_path: 출력 ONNX 파일 경로 (None이면 덮어쓰기)

    Returns:
        최적화된 ONNX 파일 경로
    """
    try:
        import onnx
        from onnxsim import simplify

        if output_path is None:
            output_path = input_path

        # ONNX 모델 로드
        model = onnx.load(input_path)

        # 모델 단순화
        model_simp, check = simplify(model)

        if check:
            onnx.save(model_simp, output_path)
            logger.info(f"ONNX model optimized: {output_path}")
        else:
            logger.warning("ONNX simplification failed, keeping original")
            if output_path != input_path:
                import shutil
                shutil.copy(input_path, output_path)

        return output_path

    except ImportError:
        logger.warning("onnxsim not installed, skipping optimization")
        return input_path
