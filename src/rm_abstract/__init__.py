"""
RM Abstract Layer - 이종 AI 반도체 통합 호환 라이브러리

기존 GPU 추론 스크립트를 코드 수정 없이 NPU/GPU 어디서든 실행 가능하도록 하는 추상화 레이어
"""

from typing import Optional, Dict, Any
import os

from .core.controller import DeviceFlowController
from .core.config import Config

__version__ = "0.1.0"
__all__ = ["init", "switch_device", "get_device_info", "get_controller"]

# Global controller instance
_global_controller: Optional[DeviceFlowController] = None


def init(
    device: str = "auto",
    cache_dir: Optional[str] = None,
    compile_options: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> DeviceFlowController:
    """
    RM Abstract Layer 초기화

    Args:
        device: 디바이스 지정
            - "auto": 자동 선택 (NPU > GPU > CPU 순)
            - "gpu:0", "gpu:1": 특정 GPU
            - "rbln:0": Rebellions ATOM NPU
            - "furiosa:0": FuriosaAI RNGD NPU
            - "cpu": CPU
        cache_dir: NPU 컴파일 캐시 디렉토리 (기본값: ~/.rm_abstract/cache)
        compile_options: NPU 컴파일 옵션
        verbose: 컴파일 진행상황 출력 여부

    Returns:
        DeviceFlowController 인스턴스

    Example:
        >>> import rm_abstract
        >>> rm_abstract.init(device="rbln:0")
        >>> # 이후 기존 코드 그대로 사용
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    """
    global _global_controller

    # 환경 변수에서 설정 로드
    device = os.environ.get("RM_DEVICE", device)
    cache_dir = os.environ.get("RM_CACHE_DIR", cache_dir)

    # Config 생성
    config = Config(
        device=device,
        cache_dir=cache_dir,
        compile_options=compile_options or {},
        verbose=verbose,
    )

    # Controller 생성 및 초기화
    _global_controller = DeviceFlowController(config)
    _global_controller.activate_hooks()

    if verbose:
        print(f"[RM Abstract] Initialized with device: {_global_controller.device_name}")

    return _global_controller


def switch_device(device: str) -> None:
    """
    런타임에 디바이스 전환

    Args:
        device: 새로운 디바이스 지정
    """
    global _global_controller

    if _global_controller is None:
        raise RuntimeError("RM Abstract Layer not initialized. Call init() first.")

    _global_controller.switch_device(device)


def get_device_info() -> Dict[str, Any]:
    """
    현재 디바이스 정보 반환

    Returns:
        디바이스 정보 딕셔너리
    """
    global _global_controller

    if _global_controller is None:
        raise RuntimeError("RM Abstract Layer not initialized. Call init() first.")

    return _global_controller.get_device_info()


def get_controller() -> Optional[DeviceFlowController]:
    """
    현재 글로벌 컨트롤러 반환

    Returns:
        DeviceFlowController 인스턴스 또는 None
    """
    return _global_controller
