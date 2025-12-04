"""
Installation Verification Script

Verifies that rm-abstract is properly installed and configured.
"""

import sys
import importlib.util
from typing import Dict, List, Tuple


def check_mark(success: bool) -> str:
    """Return check mark or X based on success"""
    return "[OK]" if success else "[FAIL]"


def check_python_version() -> Tuple[bool, str]:
    """Check Python version"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 9:
        return True, version_str
    return False, f"{version_str} (requires 3.9+)"


def check_package(package_name: str) -> Tuple[bool, str]:
    """Check if package is installed"""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False, "Not installed"

    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except Exception as e:
        return False, f"Error: {e}"


def check_core_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Check core dependencies"""
    deps = {
        "torch": "torch",
        "numpy": "numpy",
    }

    results = {}
    for name, package in deps.items():
        results[name] = check_package(package)

    return results


def check_optional_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Check optional backend dependencies"""
    deps = {
        "vLLM": "vllm",
        "Transformers": "transformers",
        "RBLN SDK": "rbln",
        "Furiosa SDK": "furiosa",
    }

    results = {}
    for name, package in deps.items():
        results[name] = check_package(package)

    return results


def check_rm_abstract() -> Tuple[bool, str]:
    """Check rm-abstract installation"""
    try:
        import rm_abstract
        version = rm_abstract.__version__
        return True, version
    except Exception as e:
        return False, f"Error: {e}"


def check_plugins() -> Dict[str, bool]:
    """Check available plugins"""
    try:
        import rm_abstract
        plugins = rm_abstract.list_plugins(available_only=False)

        availability = {}
        for name, info in plugins.items():
            availability[name] = info.get("available", False)

        return availability
    except Exception as e:
        return {"error": f"Failed to check plugins: {e}"}


def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            cuda_version = torch.version.cuda or "Unknown"
            return True, f"{device_count} device(s), {device_name}, CUDA {cuda_version}"
        return False, "CUDA not available"
    except Exception as e:
        return False, f"Error: {e}"


def run_basic_test() -> Tuple[bool, str]:
    """Run basic functionality test"""
    try:
        import rm_abstract

        # Test plugin listing
        plugins = rm_abstract.list_plugins(available_only=True)
        if not plugins:
            return False, "No plugins available"

        # Test initialization
        rm_abstract.init(device="auto", use_plugin_system=True, verbose=False)

        manager = rm_abstract.get_resource_manager()
        if manager is None:
            return False, "ResourceManager not created"

        if manager.active_plugin is None:
            return False, "No active plugin"

        return True, "All basic tests passed"

    except Exception as e:
        return False, f"Test failed: {e}"


def main():
    """Main verification function"""
    print("=" * 60)
    print("RM Abstract - Installation Verification")
    print("=" * 60)
    print()

    # Check Python version
    py_ok, py_version = check_python_version()
    print(f"{check_mark(py_ok)} Python: {py_version}")

    if not py_ok:
        print("\n[FAIL] Python 3.9+ required")
        sys.exit(1)

    # Check rm-abstract
    rm_ok, rm_version = check_rm_abstract()
    print(f"{check_mark(rm_ok)} RM Abstract: {rm_version}")

    if not rm_ok:
        print("\n[FAIL] RM Abstract not properly installed")
        print("Try: pip install -e .")
        sys.exit(1)

    print()

    # Check core dependencies
    print("Core Dependencies:")
    core_deps = check_core_dependencies()
    all_core_ok = True

    for name, (ok, version) in core_deps.items():
        print(f"  {check_mark(ok)} {name}: {version}")
        all_core_ok = all_core_ok and ok

    if not all_core_ok:
        print("\n[FAIL] Missing core dependencies")
        print("Try: pip install torch numpy")
        sys.exit(1)

    print()

    # Check optional dependencies
    print("Optional Dependencies:")
    opt_deps = check_optional_dependencies()

    for name, (ok, version) in opt_deps.items():
        status = version if ok else "Not installed"
        print(f"  {check_mark(ok)} {name}: {status}")

    print()

    # Check CUDA
    cuda_ok, cuda_info = check_cuda()
    print(f"GPU Support:")
    print(f"  {check_mark(cuda_ok)} CUDA: {cuda_info}")
    print()

    # Check plugins
    print("Available Plugins:")
    plugins = check_plugins()

    if "error" in plugins:
        print(f"  [FAIL] {plugins['error']}")
    else:
        plugin_count = sum(1 for available in plugins.values() if available)
        print(f"  Found {plugin_count} available plugin(s):")

        for name, available in plugins.items():
            print(f"    {check_mark(available)} {name}")

    print()

    # Run basic tests
    print("Running Basic Tests:")
    test_ok, test_msg = run_basic_test()
    print(f"  {check_mark(test_ok)} {test_msg}")

    print()
    print("=" * 60)

    if test_ok:
        print("[OK] Verification completed successfully!")
        print()
        print("Next steps:")
        print("  1. Try examples: python examples/simple_plugin_test.py")
        print("  2. Read docs: cat README.md")
        print("  3. Check plugins: python -c 'import rm_abstract; print(rm_abstract.list_plugins())'")
        sys.exit(0)
    else:
        print("[FAIL] Verification failed")
        print()
        print("Troubleshooting:")
        print("  1. Check INSTALL.md for detailed instructions")
        print("  2. Ensure all dependencies are installed")
        print("  3. Try: pip install -e '.[dev]'")
        sys.exit(1)


if __name__ == "__main__":
    main()
