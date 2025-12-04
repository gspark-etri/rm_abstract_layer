"""
Command Line Interface for RM Abstract

Provides convenient CLI commands for common operations.
"""

import sys
import argparse
from typing import Optional


def cmd_verify(args):
    """Run verification"""
    from .verify import main as verify_main
    verify_main()


def cmd_list_plugins(args):
    """List available plugins"""
    import rm_abstract

    plugins = rm_abstract.list_plugins(available_only=not args.all)

    if not plugins:
        print("No plugins found")
        return

    print(f"\n{'=' * 70}")
    print(f"Available Plugins ({len(plugins)})")
    print(f"{'=' * 70}\n")

    for name, info in sorted(plugins.items()):
        status = "[AVAILABLE]" if info["available"] else "[NOT AVAILABLE]"
        print(f"{status} {name}")
        print(f"  Name: {info['display_name']}")
        print(f"  Vendor: {info['vendor']}")
        print(f"  Version: {info['version']}")
        print(f"  Priority: {info['priority']}")
        print(f"  Device Types: {', '.join(info['device_types'])}")
        if info.get('description'):
            print(f"  Description: {info['description']}")
        print()


def cmd_info(args):
    """Show system information"""
    import rm_abstract
    import sys
    import platform

    print(f"\n{'=' * 70}")
    print("RM Abstract - System Information")
    print(f"{'=' * 70}\n")

    print(f"RM Abstract Version: {rm_abstract.__version__}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    # Check PyTorch
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"CUDA Devices: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch: Not installed")

    # List available plugins
    print(f"\n{'=' * 70}")
    print("Available Plugins:")
    print(f"{'=' * 70}\n")

    plugins = rm_abstract.list_plugins(available_only=True)
    if plugins:
        for name in sorted(plugins.keys()):
            print(f"  - {name}")
    else:
        print("  No plugins available")

    print()


def cmd_test(args):
    """Run basic tests"""
    print("\nRunning basic tests...\n")

    try:
        # Run simple plugin test
        import subprocess
        import os
        from pathlib import Path

        repo_root = Path(__file__).parent.parent.parent.parent
        test_file = repo_root / "examples" / "simple_plugin_test.py"

        if test_file.exists():
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
                sys.exit(1)
        else:
            print(f"Test file not found: {test_file}")
            print("Skipping tests")

    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


def cmd_example(args):
    """Run an example"""
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).parent.parent.parent.parent
    examples_dir = repo_root / "examples"

    example_map = {
        "plugin": "plugin_system_demo.py",
        "migration": "gpu_to_npu_migration.py",
        "simple": "simple_plugin_test.py",
    }

    example_file = example_map.get(args.example)
    if not example_file:
        print(f"Unknown example: {args.example}")
        print(f"Available examples: {', '.join(example_map.keys())}")
        sys.exit(1)

    example_path = examples_dir / example_file

    if not example_path.exists():
        print(f"Example not found: {example_path}")
        sys.exit(1)

    print(f"\nRunning example: {args.example}\n")
    result = subprocess.run([sys.executable, str(example_path)])
    sys.exit(result.returncode)


def cmd_init(args):
    """Initialize rm-abstract interactively"""
    import rm_abstract

    print("\n" + "=" * 70)
    print("RM Abstract - Interactive Initialization")
    print("=" * 70 + "\n")

    # Show available plugins
    plugins = rm_abstract.list_plugins(available_only=True)

    if not plugins:
        print("No plugins available!")
        print("Install at least one backend:")
        print("  pip install rm-abstract[gpu]        # For GPU support")
        print("  pip install rm-abstract[npu-rbln]   # For Rebellions NPU")
        sys.exit(1)

    print("Available plugins:")
    for i, (name, info) in enumerate(sorted(plugins.items()), 1):
        print(f"  {i}. {name} - {info['display_name']}")

    print(f"  {len(plugins) + 1}. auto (auto-select best)")
    print()

    # Get user choice
    try:
        choice = input("Select plugin (number or name): ").strip()

        if choice.isdigit():
            idx = int(choice) - 1
            if idx == len(plugins):
                device = "auto"
            elif 0 <= idx < len(plugins):
                device = list(plugins.keys())[idx]
            else:
                print("Invalid choice")
                sys.exit(1)
        else:
            device = choice

        # Initialize
        print(f"\nInitializing with device: {device}")
        rm_abstract.init(device=device, use_plugin_system=True, verbose=True)

        print("\n[OK] Initialization successful!")
        print("\nYou can now use rm-abstract in your code:")
        print(f"  import rm_abstract")
        print(f"  rm_abstract.init(device='{device}', use_plugin_system=True)")

    except KeyboardInterrupt:
        print("\n\nCancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FAIL] Initialization failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="rm-abstract",
        description="LLM Heterogeneous Resource Orchestrator CLI",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify installation"
    )
    verify_parser.set_defaults(func=cmd_verify)

    # list-plugins command
    list_parser = subparsers.add_parser(
        "list-plugins",
        help="List available plugins"
    )
    list_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all plugins (including unavailable)"
    )
    list_parser.set_defaults(func=cmd_list_plugins)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information"
    )
    info_parser.set_defaults(func=cmd_info)

    # test command
    test_parser = subparsers.add_parser(
        "test",
        help="Run basic tests"
    )
    test_parser.set_defaults(func=cmd_test)

    # example command
    example_parser = subparsers.add_parser(
        "example",
        help="Run an example"
    )
    example_parser.add_argument(
        "example",
        choices=["plugin", "migration", "simple"],
        help="Example to run"
    )
    example_parser.set_defaults(func=cmd_example)

    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize rm-abstract interactively"
    )
    init_parser.set_defaults(func=cmd_init)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
