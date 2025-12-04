"""
Plugin System Demo

Demonstrates the new plugin-based resource management system.
Shows how to:
1. List available plugins
2. Use the plugin system for model inference
3. Create custom plugins
4. Switch between plugins at runtime
"""

import rm_abstract


def demo_list_plugins():
    """Demo: List all available plugins"""
    print("=" * 60)
    print("Available Plugins")
    print("=" * 60)

    plugins = rm_abstract.list_plugins(available_only=True)

    if not plugins:
        print("No plugins available")
        return

    for name, info in plugins.items():
        print(f"\nPlugin: {name}")
        print(f"  Display Name: {info['display_name']}")
        print(f"  Vendor: {info['vendor']}")
        print(f"  Version: {info['version']}")
        print(f"  Priority: {info['priority']}")
        print(f"  Device Types: {', '.join(info['device_types'])}")
        print(f"  Available: {info['available']}")
        print(f"  Description: {info['description']}")


def demo_plugin_auto_selection():
    """Demo: Auto-select best plugin"""
    print("\n" + "=" * 60)
    print("Auto-Selection Demo")
    print("=" * 60)

    # Initialize with auto-selection using plugin system
    rm_abstract.init(device="auto", use_plugin_system=True, verbose=True)

    # Get the resource manager
    manager = rm_abstract.get_resource_manager()
    if manager:
        plugin = manager.active_plugin
        if plugin:
            meta = plugin.metadata()
            print(f"\nAuto-selected plugin: {meta.display_name}")
            print(f"  Name: {meta.name}")
            print(f"  Priority: {meta.priority.value}")

            # Get capabilities
            caps = manager.get_capabilities()
            print(f"\nCapabilities:")
            for cap, value in caps.items():
                print(f"  {cap}: {value}")


def demo_specific_plugin():
    """Demo: Use specific plugin"""
    print("\n" + "=" * 60)
    print("Specific Plugin Selection Demo")
    print("=" * 60)

    # Try different plugins in order of preference
    for device in ["gpu:0", "cpu"]:
        try:
            print(f"\nTrying device: {device}")
            rm_abstract.init(device=device, use_plugin_system=True, verbose=True)

            manager = rm_abstract.get_resource_manager()
            if manager and manager.active_plugin:
                print(f"Successfully initialized with: {manager.device_name}")
                break
        except Exception as e:
            print(f"Failed: {e}")


def demo_custom_plugin():
    """Demo: Creating a custom plugin"""
    print("\n" + "=" * 60)
    print("Custom Plugin Demo")
    print("=" * 60)

    from rm_abstract.core.plugin import Plugin, PluginMetadata, PluginPriority
    from rm_abstract.core.plugin import get_registry
    from typing import Any, Dict, Optional

    class DummyPlugin(Plugin):
        """Example custom plugin for demonstration"""

        @classmethod
        def metadata(cls) -> PluginMetadata:
            return PluginMetadata(
                name="dummy",
                display_name="Dummy Plugin",
                version="1.0.0",
                vendor="Demo",
                priority=PluginPriority.LOW,
                device_types=["custom"],
                requires=[],
                description="A dummy plugin for demonstration purposes",
            )

        def is_available(self) -> bool:
            return True

        def initialize(self) -> None:
            print("[DummyPlugin] Initialized")
            self._initialized = True

        def prepare_resource(
            self, resource: Any, config: Optional[Dict[str, Any]] = None
        ) -> Any:
            print(f"[DummyPlugin] Preparing resource: {type(resource).__name__}")
            return resource

        def execute(self, resource: Any, inputs: Any, **kwargs) -> Any:
            print(f"[DummyPlugin] Executing on resource with inputs")
            return f"Dummy output for {inputs}"

        def cleanup(self) -> None:
            print("[DummyPlugin] Cleaned up")
            self._initialized = False

    # Register custom plugin
    registry = get_registry()
    registry.register(DummyPlugin)
    print("Custom plugin registered!")

    # List plugins again
    plugins = rm_abstract.list_plugins()
    if "dummy" in plugins:
        print(f"\nDummy plugin found: {plugins['dummy']}")

    # Try to use the custom plugin
    manager = rm_abstract.get_resource_manager()
    if manager:
        try:
            manager.switch_plugin("dummy", device_id=0)
            print("Switched to dummy plugin")

            # Test resource preparation
            test_resource = "test_model"
            prepared = manager.prepare_resource(test_resource)
            print(f"Prepared resource: {prepared}")

            # Test execution
            result = manager.execute(prepared, "test_input")
            print(f"Execution result: {result}")

        except Exception as e:
            print(f"Error using dummy plugin: {e}")


def demo_plugin_switching():
    """Demo: Switch between plugins at runtime"""
    print("\n" + "=" * 60)
    print("Plugin Switching Demo")
    print("=" * 60)

    # Initialize with CPU
    rm_abstract.init(device="cpu", use_plugin_system=True, verbose=True)
    manager = rm_abstract.get_resource_manager()

    if not manager:
        print("No resource manager available")
        return

    print(f"\nCurrent plugin: {manager.device_name}")

    # Get list of available plugins
    plugins = rm_abstract.list_plugins(available_only=True)
    available_names = [name for name, info in plugins.items() if info["available"]]

    print(f"\nAvailable plugins for switching: {available_names}")

    # Try switching to each available plugin
    for plugin_name in available_names[:2]:  # Test first 2
        if plugin_name == manager.device_name:
            continue

        print(f"\nSwitching to: {plugin_name}")
        try:
            manager.switch_plugin(plugin_name)
            print(f"Successfully switched to: {manager.device_name}")

            # Show capabilities
            caps = manager.get_capabilities()
            print(f"Capabilities: {caps}")
        except Exception as e:
            print(f"Failed to switch: {e}")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("RM Abstract Layer - Plugin System Demo")
    print("=" * 60)

    try:
        # Demo 1: List plugins
        demo_list_plugins()

        # Demo 2: Auto-selection
        demo_plugin_auto_selection()

        # Demo 3: Specific plugin
        demo_specific_plugin()

        # Demo 4: Custom plugin
        demo_custom_plugin()

        # Demo 5: Plugin switching
        demo_plugin_switching()

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
