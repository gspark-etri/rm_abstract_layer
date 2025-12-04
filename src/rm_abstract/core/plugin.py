"""
Plugin System for Extensible Backend Management

Provides plugin discovery, registration, and lifecycle management
for any type of backend (GPU, NPU, CPU, or custom accelerators).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Type, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PluginPriority(Enum):
    """Plugin priority for auto-selection"""

    HIGHEST = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    LOWEST = 0


@dataclass
class PluginMetadata:
    """
    Plugin metadata and capabilities

    Provides information about the plugin's capabilities,
    requirements, and configuration.
    """

    name: str  # Unique plugin name (e.g., "vllm", "rbln", "furiosa")
    display_name: str  # Human-readable name
    version: str  # Plugin version
    vendor: Optional[str] = None  # Vendor name (e.g., "NVIDIA", "Rebellions")
    priority: PluginPriority = PluginPriority.NORMAL  # Auto-selection priority
    device_types: List[str] = field(
        default_factory=list
    )  # Supported device types ["gpu", "npu"]
    requires: List[str] = field(default_factory=list)  # Required Python packages
    description: str = ""  # Plugin description
    extra: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def check_requirements(self) -> bool:
        """
        Check if all required packages are available

        Returns:
            True if all requirements are met
        """
        import importlib.util

        for package in self.requires:
            if importlib.util.find_spec(package) is None:
                logger.debug(f"Plugin {self.name}: missing requirement {package}")
                return False
        return True


class Plugin(ABC):
    """
    Plugin Base Class

    All backend plugins must inherit from this class.
    Plugins are self-contained modules that can be loaded dynamically.
    """

    def __init__(self, **kwargs):
        """
        Initialize plugin

        Args:
            **kwargs: Plugin-specific configuration
        """
        self.config = kwargs
        self._initialized = False

    @classmethod
    @abstractmethod
    def metadata(cls) -> PluginMetadata:
        """
        Return plugin metadata

        Returns:
            PluginMetadata object
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if plugin is available on this system

        Should check for:
        - Required packages installed
        - Hardware devices present
        - Runtime/drivers available

        Returns:
            True if plugin can be used
        """
        ...

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize plugin resources

        Called once when plugin is first used.
        Should perform:
        - Device initialization
        - Runtime setup
        - Resource allocation
        """
        ...

    @abstractmethod
    def prepare_resource(
        self, resource: Any, config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Prepare resource for use

        Args:
            resource: Resource to prepare (model, data, etc.)
            config: Resource-specific configuration

        Returns:
            Prepared resource (compiled, optimized, etc.)
        """
        ...

    @abstractmethod
    def execute(self, resource: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute operation on resource

        Args:
            resource: Prepared resource
            inputs: Input data
            **kwargs: Execution options

        Returns:
            Execution result
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup plugin resources

        Should release:
        - Memory allocations
        - Device handles
        - File resources
        """
        ...

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return plugin capabilities

        Returns:
            Dictionary of capability flags
        """
        return {
            "can_compile": False,
            "can_cache": False,
            "supports_batch": False,
            "supports_streaming": False,
        }

    def __repr__(self) -> str:
        meta = self.metadata()
        return f"{self.__class__.__name__}(name={meta.name}, version={meta.version})"


class PluginRegistry:
    """
    Plugin Registry for Discovery and Management

    Manages plugin registration, discovery, and instantiation.
    Supports both manual registration and auto-discovery.
    """

    def __init__(self):
        self._plugins: Dict[str, Type[Plugin]] = {}
        self._instances: Dict[str, Plugin] = {}

    def register(self, plugin_class: Type[Plugin]) -> None:
        """
        Register a plugin class

        Args:
            plugin_class: Plugin class to register
        """
        metadata = plugin_class.metadata()

        # Check if requirements are met
        if not metadata.check_requirements():
            logger.warning(
                f"Plugin {metadata.name} requirements not met, skipping registration"
            )
            return

        if metadata.name in self._plugins:
            logger.warning(
                f"Plugin {metadata.name} already registered, overwriting"
            )

        self._plugins[metadata.name] = plugin_class
        logger.info(f"Registered plugin: {metadata.name} ({metadata.display_name})")

    def unregister(self, name: str) -> None:
        """
        Unregister a plugin

        Args:
            name: Plugin name
        """
        if name in self._instances:
            self._instances[name].cleanup()
            del self._instances[name]

        if name in self._plugins:
            del self._plugins[name]
            logger.info(f"Unregistered plugin: {name}")

    def get_plugin(self, name: str, **kwargs) -> Optional[Plugin]:
        """
        Get or create plugin instance

        Args:
            name: Plugin name
            **kwargs: Plugin configuration

        Returns:
            Plugin instance or None
        """
        # Return cached instance if exists
        if name in self._instances:
            return self._instances[name]

        # Create new instance
        plugin_class = self._plugins.get(name)
        if plugin_class is None:
            logger.error(f"Plugin not found: {name}")
            return None

        try:
            plugin = plugin_class(**kwargs)
            if not plugin.is_available():
                logger.warning(f"Plugin {name} not available on this system")
                return None

            self._instances[name] = plugin
            return plugin
        except Exception as e:
            logger.error(f"Failed to create plugin {name}: {e}")
            return None

    def list_plugins(self, available_only: bool = False) -> List[PluginMetadata]:
        """
        List all registered plugins

        Args:
            available_only: Only return available plugins

        Returns:
            List of plugin metadata
        """
        result = []
        for plugin_class in self._plugins.values():
            metadata = plugin_class.metadata()

            if available_only:
                try:
                    plugin = plugin_class()
                    if not plugin.is_available():
                        continue
                except Exception:
                    continue

            result.append(metadata)

        # Sort by priority (descending)
        result.sort(key=lambda m: m.priority.value, reverse=True)
        return result

    def auto_select(
        self, device_type: Optional[str] = None, **kwargs
    ) -> Optional[Plugin]:
        """
        Auto-select best available plugin

        Args:
            device_type: Filter by device type (e.g., "gpu", "npu")
            **kwargs: Plugin configuration

        Returns:
            Best available plugin or None
        """
        plugins = self.list_plugins(available_only=True)

        # Filter by device type if specified
        if device_type:
            plugins = [
                p
                for p in plugins
                if device_type in p.device_types or not p.device_types
            ]

        # Return highest priority plugin
        if plugins:
            best = plugins[0]
            logger.info(f"Auto-selected plugin: {best.name} (priority={best.priority})")
            return self.get_plugin(best.name, **kwargs)

        logger.warning("No suitable plugin found")
        return None

    def discover_plugins(self, package_path: str = "rm_abstract.backends") -> None:
        """
        Discover and register plugins from package

        Args:
            package_path: Python package path to search
        """
        import importlib
        import pkgutil

        try:
            package = importlib.import_module(package_path)
        except ImportError as e:
            logger.warning(f"Failed to import package {package_path}: {e}")
            return

        # Walk through package modules
        for _, name, ispkg in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            try:
                module = importlib.import_module(name)

                # Find Plugin subclasses in module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    # Check if it's a Plugin subclass (but not Plugin itself)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, Plugin)
                        and attr is not Plugin
                    ):
                        try:
                            self.register(attr)
                        except Exception as e:
                            logger.debug(
                                f"Skipped registering {attr_name} from {name}: {e}"
                            )
            except Exception as e:
                logger.debug(f"Failed to import {name}: {e}")

    def cleanup_all(self) -> None:
        """Cleanup all plugin instances"""
        for plugin in self._instances.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin: {e}")
        self._instances.clear()


# Global plugin registry
_global_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Get global plugin registry"""
    return _global_registry


def register_plugin(plugin_class: Type[Plugin]) -> None:
    """
    Register a plugin (decorator friendly)

    Args:
        plugin_class: Plugin class to register
    """
    _global_registry.register(plugin_class)


def plugin_decorator(metadata_kwargs: Dict[str, Any]) -> Callable:
    """
    Decorator for auto-registering plugins

    Usage:
        @plugin_decorator({"name": "my_plugin", "version": "1.0.0"})
        class MyPlugin(Plugin):
            ...
    """

    def decorator(plugin_class: Type[Plugin]) -> Type[Plugin]:
        register_plugin(plugin_class)
        return plugin_class

    return decorator
