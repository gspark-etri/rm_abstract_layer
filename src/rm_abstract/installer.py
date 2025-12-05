"""
RM Abstract Layer - Installation Helper

Helps users install required dependencies for different components.

Usage:
    python -m rm_abstract.installer [component]
    
    # Or in Python
    from rm_abstract.installer import install_component, check_requirements
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Component(Enum):
    BASE = "base"
    GPU = "gpu"
    TRITON = "triton"
    TORCHSERVE = "torchserve"
    NPU_RBLN = "npu-rbln"
    ALL = "all"


@dataclass
class Dependency:
    name: str
    pip_package: str
    check_import: str
    description: str
    optional: bool = False


@dataclass
class SystemDependency:
    name: str
    check_command: str
    install_apt: str
    install_yum: str
    install_brew: str
    description: str


# Component dependencies
COMPONENT_DEPS: Dict[Component, List[Dependency]] = {
    Component.BASE: [
        Dependency("torch", "torch>=2.0.0", "torch", "PyTorch deep learning framework"),
        Dependency("transformers", "transformers>=4.30.0", "transformers", "HuggingFace Transformers"),
        Dependency("psutil", "psutil>=5.9.0", "psutil", "System utilities"),
    ],
    Component.GPU: [
        Dependency("vllm", "vllm>=0.4.0", "vllm", "High-performance LLM serving"),
    ],
    Component.TRITON: [
        Dependency("tritonclient", "tritonclient[all]>=2.40.0", "tritonclient", "Triton client library"),
        Dependency("onnx", "onnx>=1.14.0", "onnx", "ONNX model format", optional=True),
    ],
    Component.TORCHSERVE: [
        Dependency("torchserve", "torchserve>=0.8.0", "ts", "TorchServe"),
        Dependency("torch-model-archiver", "torch-model-archiver>=0.8.0", "model_archiver", "Model archiver"),
    ],
    Component.NPU_RBLN: [
        Dependency("rebel/rbln", "rebel-sdk", "rebel", "Rebellions SDK", optional=True),
    ],
}

SYSTEM_DEPS: Dict[str, SystemDependency] = {
    "java": SystemDependency(
        name="Java 11",
        check_command="java -version",
        install_apt="sudo apt-get install -y openjdk-11-jdk",
        install_yum="sudo yum install -y java-11-openjdk",
        install_brew="brew install openjdk@11",
        description="Required for TorchServe server",
    ),
    "docker": SystemDependency(
        name="Docker",
        check_command="docker --version",
        install_apt="curl -fsSL https://get.docker.com | sh",
        install_yum="sudo yum install -y docker && sudo systemctl start docker",
        install_brew="brew install --cask docker",
        description="Required for Triton server",
    ),
}


def detect_package_manager() -> str:
    """Detect system package manager"""
    if shutil.which("apt-get"):
        return "apt"
    elif shutil.which("yum"):
        return "yum"
    elif shutil.which("dnf"):
        return "dnf"
    elif shutil.which("brew"):
        return "brew"
    return "unknown"


def check_import(module_name: str) -> bool:
    """Check if a Python module can be imported"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def check_command(command: str) -> bool:
    """Check if a system command is available"""
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except:
        return False


def check_requirements(component: Component) -> Dict[str, bool]:
    """
    Check if requirements for a component are installed
    
    Args:
        component: Component to check
        
    Returns:
        Dictionary of {dependency_name: is_installed}
    """
    results = {}
    
    # Check base deps first if not checking base
    if component != Component.BASE:
        for dep in COMPONENT_DEPS[Component.BASE]:
            results[dep.name] = check_import(dep.check_import)
    
    # Check component specific deps
    if component == Component.ALL:
        for comp in Component:
            if comp != Component.ALL:
                for dep in COMPONENT_DEPS.get(comp, []):
                    results[dep.name] = check_import(dep.check_import)
    else:
        for dep in COMPONENT_DEPS.get(component, []):
            results[dep.name] = check_import(dep.check_import)
    
    return results


def check_system_requirements() -> Dict[str, bool]:
    """Check system dependencies"""
    results = {}
    for name, dep in SYSTEM_DEPS.items():
        results[name] = check_command(dep.check_command)
    return results


def install_pip_package(package: str, use_uv: bool = True) -> bool:
    """Install a pip package"""
    try:
        if use_uv and shutil.which("uv"):
            cmd = ["uv", "pip", "install", package]
        else:
            cmd = [sys.executable, "-m", "pip", "install", package]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error installing {package}: {e}")
        return False


def install_component(component: Component, verbose: bool = True) -> Tuple[int, int]:
    """
    Install dependencies for a component
    
    Args:
        component: Component to install
        verbose: Print progress
        
    Returns:
        Tuple of (installed_count, failed_count)
    """
    installed = 0
    failed = 0
    
    deps_to_install = []
    
    # Always include base
    deps_to_install.extend(COMPONENT_DEPS[Component.BASE])
    
    if component == Component.ALL:
        for comp in Component:
            if comp != Component.ALL and comp != Component.BASE:
                deps_to_install.extend(COMPONENT_DEPS.get(comp, []))
    elif component != Component.BASE:
        deps_to_install.extend(COMPONENT_DEPS.get(component, []))
    
    for dep in deps_to_install:
        if check_import(dep.check_import):
            if verbose:
                print(f"  ✓ {dep.name} already installed")
            continue
        
        if verbose:
            print(f"  Installing {dep.name}...", end=" ", flush=True)
        
        if install_pip_package(dep.pip_package):
            if verbose:
                print("✓")
            installed += 1
        else:
            if dep.optional:
                if verbose:
                    print("⚠ (optional)")
            else:
                if verbose:
                    print("✗")
                failed += 1
    
    return installed, failed


def print_installation_guide(component: Optional[Component] = None):
    """Print installation guide"""
    print()
    print("=" * 60)
    print("  RM Abstract Layer - Installation Guide")
    print("=" * 60)
    print()
    
    print("Components:")
    print("-" * 40)
    
    components_info = [
        (Component.BASE, "Base", "Core functionality"),
        (Component.GPU, "GPU (vLLM)", "High-performance GPU inference"),
        (Component.TRITON, "Triton", "Multi-model serving"),
        (Component.TORCHSERVE, "TorchServe", "PyTorch native serving"),
        (Component.NPU_RBLN, "Rebellions NPU", "ATOM NPU support"),
        (Component.ALL, "All", "Everything"),
    ]
    
    for comp, name, desc in components_info:
        status = check_requirements(comp)
        installed_count = sum(status.values())
        total_count = len(status)
        
        if total_count == 0:
            status_str = "✓"
        elif installed_count == total_count:
            status_str = "✓"
        elif installed_count > 0:
            status_str = f"⚠ ({installed_count}/{total_count})"
        else:
            status_str = "✗"
        
        print(f"  {status_str} {name}: {desc}")
    
    print()
    print("System Dependencies:")
    print("-" * 40)
    
    sys_status = check_system_requirements()
    for name, dep in SYSTEM_DEPS.items():
        status = "✓" if sys_status[name] else "✗"
        print(f"  {status} {dep.name}: {dep.description}")
    
    print()
    print("Installation Commands:")
    print("-" * 40)
    print("  # Python packages")
    print("  python -m rm_abstract.installer gpu        # GPU/vLLM")
    print("  python -m rm_abstract.installer triton     # Triton")
    print("  python -m rm_abstract.installer torchserve # TorchServe")
    print("  python -m rm_abstract.installer all        # Everything")
    print()
    print("  # System dependencies")
    print("  ./scripts/install_deps.sh java            # Java for TorchServe")
    print("  ./scripts/install_deps.sh docker          # Docker for Triton")
    print()


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RM Abstract Layer - Dependency Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m rm_abstract.installer              # Show status
  python -m rm_abstract.installer gpu          # Install GPU/vLLM
  python -m rm_abstract.installer triton       # Install Triton
  python -m rm_abstract.installer torchserve   # Install TorchServe
  python -m rm_abstract.installer all          # Install everything
        """,
    )
    
    parser.add_argument(
        "component",
        nargs="?",
        choices=["base", "gpu", "triton", "torchserve", "npu-rbln", "all", "status"],
        default="status",
        help="Component to install (default: status)",
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode",
    )
    
    args = parser.parse_args()
    
    if args.component == "status":
        print_installation_guide()
        return
    
    # Map string to enum
    component_map = {
        "base": Component.BASE,
        "gpu": Component.GPU,
        "triton": Component.TRITON,
        "torchserve": Component.TORCHSERVE,
        "npu-rbln": Component.NPU_RBLN,
        "all": Component.ALL,
    }
    
    component = component_map[args.component]
    
    print()
    print(f"Installing: {args.component}")
    print("-" * 40)
    
    installed, failed = install_component(component, verbose=not args.quiet)
    
    print()
    print("-" * 40)
    print(f"Installed: {installed}, Failed: {failed}")
    
    if failed > 0:
        print("\nSome packages failed to install. You may need to install them manually.")
        sys.exit(1)
    
    # Post-install instructions
    if component in [Component.TORCHSERVE, Component.ALL]:
        print("\n⚠ TorchServe server requires Java 11+")
        print("  Install: ./scripts/install_deps.sh java")
    
    if component in [Component.TRITON, Component.ALL]:
        print("\n⚠ Triton server requires Docker")
        print("  Install: ./scripts/install_deps.sh docker")
        print("  Start:   docker-compose -f docker/docker-compose.yml up triton")
    
    print("\nVerify installation:")
    print("  python -m rm_abstract.system_validator")


if __name__ == "__main__":
    main()

