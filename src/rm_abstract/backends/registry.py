"""
백엔드 레지스트리

사용 가능한 백엔드를 자동으로 탐지하고 등록
"""

import logging

logger = logging.getLogger(__name__)


def register_all_backends() -> None:
    """모든 사용 가능한 백엔드 등록"""
    from ..core.controller import DeviceFlowController

    # GPU 백엔드 등록
    try:
        from .gpu.vllm_backend import VLLMBackend

        DeviceFlowController.register_backend("gpu", VLLMBackend)
        logger.debug("Registered GPU (vLLM) backend")
    except ImportError as e:
        logger.debug(f"GPU backend not available: {e}")

    # CPU 백엔드 등록
    try:
        from .cpu.cpu_backend import CPUBackend

        DeviceFlowController.register_backend("cpu", CPUBackend)
        logger.debug("Registered CPU backend")
    except ImportError as e:
        logger.debug(f"CPU backend not available: {e}")

    # Rebellions NPU 백엔드 등록
    try:
        from .npu.plugins.rebellions import RBLNBackend

        DeviceFlowController.register_backend("rbln", RBLNBackend)
        logger.debug("Registered Rebellions (RBLN) backend")
    except ImportError as e:
        logger.debug(f"Rebellions backend not available: {e}")

    # FuriosaAI NPU 백엔드 등록
    try:
        from .npu.plugins.furiosa import FuriosaBackend

        DeviceFlowController.register_backend("furiosa", FuriosaBackend)
        logger.debug("Registered FuriosaAI backend")
    except ImportError as e:
        logger.debug(f"FuriosaAI backend not available: {e}")
