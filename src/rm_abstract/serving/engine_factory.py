"""
Serving Engine Factory

Creates and manages serving engines with unified interface.
"""

from typing import Dict, List, Optional, Type
import logging

from .base import ServingEngine, ServingEngineType, ServingConfig, DeviceTarget
from .vllm_engine import VLLMServingEngine
from .triton_engine import TritonServingEngine
from .torchserve_engine import TorchServeEngine

logger = logging.getLogger(__name__)

# Registry of available engines
_ENGINE_REGISTRY: Dict[ServingEngineType, Type[ServingEngine]] = {
    ServingEngineType.VLLM: VLLMServingEngine,
    ServingEngineType.TRITON: TritonServingEngine,
    ServingEngineType.TORCHSERVE: TorchServeEngine,
}


def get_available_engines() -> Dict[str, Dict]:
    """
    Get all available serving engines and their status
    
    Returns:
        Dictionary of engine info with availability status
    """
    engines = {}
    
    for engine_type, engine_class in _ENGINE_REGISTRY.items():
        available = engine_class.is_available()
        supported_devices = engine_class.supported_devices()
        
        engines[engine_type.value] = {
            "name": engine_class.__name__,
            "available": available,
            "supported_devices": [d.value for d in supported_devices],
            "description": engine_class.__doc__.split("\n")[1].strip() if engine_class.__doc__ else "",
        }
    
    return engines


def create_serving_engine(
    config: Optional[ServingConfig] = None,
    engine: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> ServingEngine:
    """
    Create a serving engine instance
    
    Args:
        config: ServingConfig object (if provided, other args are ignored)
        engine: Engine type ("vllm", "triton", "torchserve")
        device: Target device ("gpu", "cpu", "npu_rbln")
        **kwargs: Additional configuration options
        
    Returns:
        Configured ServingEngine instance
        
    Example:
        # Using config object
        config = ServingConfig(
            engine=ServingEngineType.VLLM,
            device=DeviceTarget.GPU,
            model_name="gpt2",
        )
        engine = create_serving_engine(config)
        
        # Using keyword arguments
        engine = create_serving_engine(
            engine="vllm",
            device="gpu",
            model_name="gpt2",
            port=8000,
        )
    """
    if config is None:
        # Build config from kwargs
        engine_type = ServingEngineType(engine or "vllm")
        device_target = DeviceTarget(device or "gpu")
        
        config = ServingConfig(
            engine=engine_type,
            device=device_target,
            model_name=kwargs.pop("model_name", ""),
            model_path=kwargs.pop("model_path", None),
            host=kwargs.pop("host", "0.0.0.0"),
            port=kwargs.pop("port", 8000),
            device_id=kwargs.pop("device_id", 0),
            tensor_parallel_size=kwargs.pop("tensor_parallel_size", 1),
            max_batch_size=kwargs.pop("max_batch_size", 32),
            max_seq_len=kwargs.pop("max_seq_len", 2048),
            extra_options=kwargs,
        )
    
    # Get engine class
    engine_class = _ENGINE_REGISTRY.get(config.engine)
    
    if engine_class is None:
        raise ValueError(f"Unknown engine type: {config.engine}")
    
    if not engine_class.is_available():
        raise RuntimeError(f"Engine {config.engine.value} is not available")
    
    # Check device support
    supported_devices = engine_class.supported_devices()
    if config.device not in supported_devices:
        raise ValueError(
            f"Device {config.device.value} not supported by {config.engine.value}. "
            f"Supported: {[d.value for d in supported_devices]}"
        )
    
    return engine_class(config)


def auto_select_engine(
    device: Optional[str] = None,
    prefer_engine: Optional[str] = None,
) -> ServingEngine:
    """
    Automatically select the best available serving engine
    
    Args:
        device: Preferred device (gpu, cpu, npu_rbln)
        prefer_engine: Preferred engine if available
        
    Returns:
        Best available ServingEngine
    """
    # Priority order for engines
    priority = [
        ServingEngineType.VLLM,
        ServingEngineType.TRITON,
        ServingEngineType.TORCHSERVE,
    ]
    
    # If prefer_engine specified, move it to front
    if prefer_engine:
        try:
            preferred = ServingEngineType(prefer_engine)
            priority.remove(preferred)
            priority.insert(0, preferred)
        except ValueError:
            pass
    
    # Determine device
    if device:
        device_target = DeviceTarget(device)
    else:
        # Auto-detect best device
        device_target = _auto_detect_device()
    
    # Find first available engine that supports the device
    for engine_type in priority:
        engine_class = _ENGINE_REGISTRY.get(engine_type)
        if engine_class is None:
            continue
        
        if not engine_class.is_available():
            continue
        
        if device_target not in engine_class.supported_devices():
            continue
        
        config = ServingConfig(
            engine=engine_type,
            device=device_target,
        )
        
        logger.info(f"Auto-selected engine: {engine_type.value} on {device_target.value}")
        return engine_class(config)
    
    raise RuntimeError("No suitable serving engine available")


def _auto_detect_device() -> DeviceTarget:
    """Auto-detect the best available device"""
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            return DeviceTarget.GPU
    except ImportError:
        pass
    
    # Check RBLN NPU
    try:
        import rebel
        devices = rebel.get_devices()
        if devices:
            return DeviceTarget.NPU_RBLN
    except ImportError:
        try:
            import rbln
            devices = rbln.get_devices()
            if devices:
                return DeviceTarget.NPU_RBLN
        except ImportError:
            pass
    
    # Fallback to CPU
    return DeviceTarget.CPU


def list_engines() -> None:
    """Print available engines (for CLI)"""
    engines = get_available_engines()
    
    print("=" * 60)
    print("Available Serving Engines")
    print("=" * 60)
    
    for name, info in engines.items():
        status = "✓ Available" if info["available"] else "✗ Not available"
        print(f"\n{name}:")
        print(f"  Status: {status}")
        print(f"  Devices: {', '.join(info['supported_devices'])}")
        if info["description"]:
            print(f"  Description: {info['description']}")

