"""
System Information Module

Provides comprehensive information about available resources,
serving engines, and backends on the current system.

Usage:
    # Python API
    from rm_abstract import get_system_info, print_system_info
    info = get_system_info()
    print_system_info()
    
    # CLI
    python -m rm_abstract.system_info
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
import platform
import os

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU device information"""
    id: int
    name: str
    memory_total_gb: float
    memory_free_gb: float
    compute_capability: str
    driver_version: str = ""
    cuda_version: str = ""


@dataclass
class NPUInfo:
    """NPU device information"""
    id: int
    name: str
    vendor: str
    sdk_version: str = ""
    available: bool = True


@dataclass
class CPUInfo:
    """CPU information"""
    name: str
    cores: int
    threads: int
    memory_total_gb: float
    memory_free_gb: float


@dataclass
class BackendInfo:
    """Backend availability information"""
    name: str
    display_name: str
    available: bool
    device_type: str
    version: str = ""
    description: str = ""


@dataclass
class ServingEngineInfo:
    """Serving engine availability information"""
    name: str
    display_name: str
    available: bool
    supported_devices: List[str]
    version: str = ""
    description: str = ""


@dataclass
class SystemInfo:
    """Complete system information"""
    # Hardware
    gpus: List[GPUInfo] = field(default_factory=list)
    npus: List[NPUInfo] = field(default_factory=list)
    cpu: Optional[CPUInfo] = None
    
    # Software
    backends: List[BackendInfo] = field(default_factory=list)
    serving_engines: List[ServingEngineInfo] = field(default_factory=list)
    
    # System
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "hardware": {
                "gpus": [vars(g) for g in self.gpus],
                "npus": [vars(n) for n in self.npus],
                "cpu": vars(self.cpu) if self.cpu else None,
            },
            "software": {
                "backends": [vars(b) for b in self.backends],
                "serving_engines": [vars(e) for e in self.serving_engines],
            },
            "system": {
                "os_name": self.os_name,
                "os_version": self.os_version,
                "python_version": self.python_version,
            },
        }


def _detect_gpus() -> List[GPUInfo]:
    """Detect available NVIDIA GPUs"""
    gpus = []
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda or ""
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                
                # Get memory info
                total_mem = props.total_memory / (1024**3)
                try:
                    free_mem = (props.total_memory - torch.cuda.memory_reserved(i)) / (1024**3)
                except:
                    free_mem = total_mem
                
                gpus.append(GPUInfo(
                    id=i,
                    name=props.name,
                    memory_total_gb=round(total_mem, 2),
                    memory_free_gb=round(free_mem, 2),
                    compute_capability=f"{props.major}.{props.minor}",
                    cuda_version=cuda_version,
                ))
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")
    
    return gpus


def _detect_npus() -> List[NPUInfo]:
    """Detect available NPUs (Rebellions, FuriosaAI)"""
    npus = []
    
    # Check Rebellions ATOM
    try:
        import rebel
        devices = rebel.get_devices()
        for i, dev in enumerate(devices):
            npus.append(NPUInfo(
                id=i,
                name="Rebellions ATOM",
                vendor="Rebellions",
                sdk_version=getattr(rebel, "__version__", "unknown"),
            ))
    except ImportError:
        try:
            import rbln
            devices = rbln.get_devices()
            for i, dev in enumerate(devices):
                npus.append(NPUInfo(
                    id=i,
                    name="Rebellions ATOM",
                    vendor="Rebellions",
                    sdk_version=getattr(rbln, "__version__", "unknown"),
                ))
        except ImportError:
            pass
    except Exception as e:
        logger.debug(f"Rebellions NPU detection failed: {e}")
    
    # Check FuriosaAI
    try:
        import furiosa
        # FuriosaAI specific detection
        npus.append(NPUInfo(
            id=len(npus),
            name="FuriosaAI RNGD",
            vendor="FuriosaAI",
            sdk_version=getattr(furiosa, "__version__", "unknown"),
        ))
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"FuriosaAI NPU detection failed: {e}")
    
    return npus


def _detect_cpu() -> CPUInfo:
    """Detect CPU information"""
    import psutil
    
    try:
        mem = psutil.virtual_memory()
        
        return CPUInfo(
            name=platform.processor() or platform.machine(),
            cores=psutil.cpu_count(logical=False) or 1,
            threads=psutil.cpu_count(logical=True) or 1,
            memory_total_gb=round(mem.total / (1024**3), 2),
            memory_free_gb=round(mem.available / (1024**3), 2),
        )
    except ImportError:
        return CPUInfo(
            name=platform.processor() or platform.machine(),
            cores=os.cpu_count() or 1,
            threads=os.cpu_count() or 1,
            memory_total_gb=0,
            memory_free_gb=0,
        )


def _detect_backends() -> List[BackendInfo]:
    """Detect available backends"""
    backends = []
    
    # GPU (vLLM)
    try:
        import vllm
        import torch
        backends.append(BackendInfo(
            name="gpu",
            display_name="vLLM GPU Backend",
            available=torch.cuda.is_available(),
            device_type="GPU",
            version=getattr(vllm, "__version__", ""),
            description="High-performance LLM inference using vLLM on NVIDIA GPUs",
        ))
    except ImportError:
        backends.append(BackendInfo(
            name="gpu",
            display_name="vLLM GPU Backend",
            available=False,
            device_type="GPU",
            description="Not installed (pip install vllm)",
        ))
    
    # CPU (PyTorch)
    try:
        import torch
        backends.append(BackendInfo(
            name="cpu",
            display_name="PyTorch CPU Backend",
            available=True,
            device_type="CPU",
            version=torch.__version__,
            description="Fallback CPU inference using PyTorch",
        ))
    except ImportError:
        backends.append(BackendInfo(
            name="cpu",
            display_name="PyTorch CPU Backend",
            available=False,
            device_type="CPU",
            description="Not installed (pip install torch)",
        ))
    
    # Rebellions NPU
    rbln_available = False
    rbln_version = ""
    
    try:
        from optimum.rbln import RBLNModelForCausalLM
        rbln_available = True
        rbln_version = "optimum-rbln"
    except ImportError:
        pass
    
    try:
        import vllm
        # Check if vllm-rbln variant
        rbln_available = rbln_available or True  # vLLM can work with RBLN
    except ImportError:
        pass
    
    backends.append(BackendInfo(
        name="rbln",
        display_name="Rebellions ATOM NPU",
        available=rbln_available,
        device_type="NPU",
        version=rbln_version,
        description="Rebellions ATOM NPU backend (vLLM-RBLN or Optimum-RBLN)",
    ))
    
    # FuriosaAI NPU
    try:
        import furiosa
        backends.append(BackendInfo(
            name="furiosa",
            display_name="FuriosaAI RNGD NPU",
            available=True,
            device_type="NPU",
            version=getattr(furiosa, "__version__", ""),
            description="FuriosaAI RNGD NPU backend",
        ))
    except ImportError:
        backends.append(BackendInfo(
            name="furiosa",
            display_name="FuriosaAI RNGD NPU",
            available=False,
            device_type="NPU",
            description="Not installed",
        ))
    
    return backends


def _detect_serving_engines() -> List[ServingEngineInfo]:
    """Detect available serving engines"""
    engines = []
    
    # vLLM
    try:
        import vllm
        engines.append(ServingEngineInfo(
            name="vllm",
            display_name="vLLM",
            available=True,
            supported_devices=["GPU", "NPU (RBLN)", "CPU"],
            version=getattr(vllm, "__version__", ""),
            description="High-performance LLM serving with continuous batching",
        ))
    except ImportError:
        engines.append(ServingEngineInfo(
            name="vllm",
            display_name="vLLM",
            available=False,
            supported_devices=["GPU", "NPU (RBLN)", "CPU"],
            description="Not installed (pip install vllm)",
        ))
    
    # Triton
    try:
        import tritonclient.http
        engines.append(ServingEngineInfo(
            name="triton",
            display_name="Triton Inference Server",
            available=True,
            supported_devices=["GPU", "NPU (RBLN)", "CPU"],
            version="client installed",
            description="NVIDIA Triton for multi-model serving",
        ))
    except ImportError:
        engines.append(ServingEngineInfo(
            name="triton",
            display_name="Triton Inference Server",
            available=False,
            supported_devices=["GPU", "NPU (RBLN)", "CPU"],
            description="Not installed (pip install tritonclient[all])",
        ))
    
    # TorchServe
    try:
        import subprocess
        result = subprocess.run(
            ["torch-model-archiver", "--help"],
            capture_output=True,
            timeout=5,
        )
        ts_available = result.returncode == 0
    except:
        ts_available = False
    
    engines.append(ServingEngineInfo(
        name="torchserve",
        display_name="TorchServe",
        available=ts_available,
        supported_devices=["GPU", "NPU (RBLN)", "CPU"],
        description="PyTorch native model serving" if ts_available else "Not installed (pip install torchserve torch-model-archiver)",
    ))
    
    return engines


def get_system_info() -> SystemInfo:
    """
    Get comprehensive system information
    
    Returns:
        SystemInfo object with hardware, software, and system details
    """
    return SystemInfo(
        gpus=_detect_gpus(),
        npus=_detect_npus(),
        cpu=_detect_cpu(),
        backends=_detect_backends(),
        serving_engines=_detect_serving_engines(),
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
    )


def print_system_info(info: Optional[SystemInfo] = None) -> None:
    """
    Print system information in a formatted table
    
    Args:
        info: SystemInfo object (if None, will be detected)
    """
    if info is None:
        info = get_system_info()
    
    print()
    print("=" * 70)
    print("  RM Abstract Layer - System Information")
    print("=" * 70)
    
    # System Info
    print(f"\n📋 System")
    print(f"   OS: {info.os_name} {info.os_version}")
    print(f"   Python: {info.python_version}")
    
    # CPU
    if info.cpu:
        print(f"\n💻 CPU")
        print(f"   {info.cpu.name}")
        print(f"   Cores: {info.cpu.cores} ({info.cpu.threads} threads)")
        print(f"   Memory: {info.cpu.memory_free_gb:.1f} / {info.cpu.memory_total_gb:.1f} GB available")
    
    # GPUs
    print(f"\n🎮 GPUs ({len(info.gpus)} detected)")
    if info.gpus:
        for gpu in info.gpus:
            status = "✓" if gpu.memory_free_gb > 1 else "⚠"
            print(f"   {status} [{gpu.id}] {gpu.name}")
            print(f"      Memory: {gpu.memory_free_gb:.1f} / {gpu.memory_total_gb:.1f} GB")
            print(f"      CUDA: {gpu.cuda_version}, Compute: {gpu.compute_capability}")
    else:
        print("   ✗ No NVIDIA GPUs detected")
    
    # NPUs
    print(f"\n🔮 NPUs ({len(info.npus)} detected)")
    if info.npus:
        for npu in info.npus:
            print(f"   ✓ [{npu.id}] {npu.name} ({npu.vendor})")
            if npu.sdk_version:
                print(f"      SDK: {npu.sdk_version}")
    else:
        print("   ✗ No NPUs detected")
    
    # Backends
    print(f"\n⚙️  Backends")
    for backend in info.backends:
        status = "✓" if backend.available else "✗"
        print(f"   {status} {backend.display_name}")
        if backend.available and backend.version:
            print(f"      Version: {backend.version}")
        elif not backend.available and backend.description:
            print(f"      {backend.description}")
    
    # Serving Engines
    print(f"\n🚀 Serving Engines")
    for engine in info.serving_engines:
        status = "✓" if engine.available else "✗"
        print(f"   {status} {engine.display_name}")
        if engine.available:
            print(f"      Devices: {', '.join(engine.supported_devices)}")
            if engine.version:
                print(f"      Version: {engine.version}")
        else:
            print(f"      {engine.description}")
    
    # Summary
    print(f"\n📊 Summary")
    available_backends = sum(1 for b in info.backends if b.available)
    available_engines = sum(1 for e in info.serving_engines if e.available)
    
    print(f"   Hardware: {len(info.gpus)} GPU(s), {len(info.npus)} NPU(s)")
    print(f"   Backends: {available_backends}/{len(info.backends)} available")
    print(f"   Serving Engines: {available_engines}/{len(info.serving_engines)} available")
    
    # Recommendations
    print(f"\n💡 Recommendations")
    if info.gpus and any(b.name == "gpu" and b.available for b in info.backends):
        print("   • Use GPU backend with vLLM for best LLM performance")
    if info.npus:
        print("   • Use NPU backend for power-efficient inference")
    if not info.gpus and not info.npus:
        print("   • CPU-only mode available (limited performance)")
    
    print()
    print("=" * 70)


def get_quick_status() -> Dict[str, bool]:
    """
    Get quick availability status for all components
    
    Returns:
        Dictionary with availability status
    """
    info = get_system_info()
    
    return {
        "gpu_available": len(info.gpus) > 0,
        "npu_available": len(info.npus) > 0,
        "vllm_available": any(e.name == "vllm" and e.available for e in info.serving_engines),
        "triton_available": any(e.name == "triton" and e.available for e in info.serving_engines),
        "torchserve_available": any(e.name == "torchserve" and e.available for e in info.serving_engines),
        "gpu_backend": any(b.name == "gpu" and b.available for b in info.backends),
        "cpu_backend": any(b.name == "cpu" and b.available for b in info.backends),
        "rbln_backend": any(b.name == "rbln" and b.available for b in info.backends),
    }


# CLI entry point
if __name__ == "__main__":
    print_system_info()

