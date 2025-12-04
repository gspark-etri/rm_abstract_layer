"""
Simple Plugin System Test

Quick test to verify plugin system is working correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rm_abstract


def test_list_plugins():
    """Test listing plugins"""
    print("=" * 60)
    print("Test: List Plugins")
    print("=" * 60)

    plugins = rm_abstract.list_plugins(available_only=True)

    print(f"\nFound {len(plugins)} available plugin(s):\n")

    for name, info in plugins.items():
        print(f"  [{name}]")
        print(f"    Display Name: {info['display_name']}")
        print(f"    Vendor: {info['vendor']}")
        print(f"    Priority: {info['priority']}")
        print(f"    Available: {info['available']}")
        print()

    return len(plugins) > 0


def test_plugin_initialization():
    """Test initializing with plugin system"""
    print("=" * 60)
    print("Test: Plugin Initialization")
    print("=" * 60)

    try:
        # Initialize with plugin system
        rm_abstract.init(device="auto", use_plugin_system=True, verbose=True)

        # Get resource manager
        manager = rm_abstract.get_resource_manager()

        if manager is None:
            print("[FAIL] Failed: No resource manager created")
            return False

        if manager.active_plugin is None:
            print("[FAIL] Failed: No active plugin")
            return False

        meta = manager.active_plugin.metadata()
        print(f"\n[OK] Success: Using plugin '{meta.name}'")
        print(f"   Display Name: {meta.display_name}")
        print(f"   Priority: {meta.priority.value}")

        # Test capabilities
        caps = manager.get_capabilities()
        print(f"\n   Capabilities:")
        for cap, value in caps.items():
            print(f"     {cap}: {value}")

        return True

    except Exception as e:
        print(f"[FAIL] Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_custom_plugin():
    """Test creating and using custom plugin"""
    print("\n" + "=" * 60)
    print("Test: Custom Plugin")
    print("=" * 60)

    try:
        from rm_abstract.core.plugin import (
            Plugin,
            PluginMetadata,
            PluginPriority,
            get_registry,
        )
        from typing import Any, Dict, Optional

        class DemoPlugin(Plugin):
            """Demo plugin for testing"""

            @classmethod
            def metadata(cls) -> PluginMetadata:
                return PluginMetadata(
                    name="demo",
                    display_name="Demo Plugin",
                    version="1.0.0",
                    vendor="Test",
                    priority=PluginPriority.NORMAL,
                    device_types=["demo"],
                    description="Demo plugin for testing",
                )

            def is_available(self) -> bool:
                return True

            def initialize(self) -> None:
                self._initialized = True

            def prepare_resource(
                self, resource: Any, config: Optional[Dict[str, Any]] = None
            ) -> Any:
                return f"prepared_{resource}"

            def execute(self, resource: Any, inputs: Any, **kwargs) -> Any:
                return f"result_for_{inputs}"

            def cleanup(self) -> None:
                self._initialized = False

        # Register
        get_registry().register(DemoPlugin)
        print("[OK] Custom plugin registered")

        # Verify it appears in list
        plugins = rm_abstract.list_plugins()
        if "demo" in plugins:
            print(f"[OK] Custom plugin found in registry: {plugins['demo']['display_name']}")
        else:
            print("[FAIL] Custom plugin not found in registry")
            return False

        # Try using it
        manager = rm_abstract.get_resource_manager()
        if manager:
            manager.switch_plugin("demo")
            print(f"[OK] Switched to custom plugin: {manager.device_name}")

            # Test operations
            resource = "test_model"
            prepared = manager.prepare_resource(resource)
            print(f"[OK] Prepare test: {prepared}")

            result = manager.execute(prepared, "test_input")
            print(f"[OK] Execute test: {result}")

            return True

    except Exception as e:
        print(f"[FAIL] Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RM Abstract Layer - Plugin System Tests")
    print("=" * 60 + "\n")

    results = []

    # Test 1: List plugins
    results.append(("List Plugins", test_list_plugins()))

    # Test 2: Initialize with plugin system
    results.append(("Plugin Initialization", test_plugin_initialization()))

    # Test 3: Custom plugin
    results.append(("Custom Plugin", test_custom_plugin()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
