"""
Resource Manager - Unified Resource Management System

Manages resources (models, data, devices) across different backends
using the plugin system. Handles resource lifecycle and allocation.
"""

from typing import Any, Dict, Optional, List
import logging

from .plugin import Plugin, PluginRegistry, get_registry, PluginMetadata
from .config import Config

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Resource Manager

    Central component for managing resources across different backends.
    Provides unified interface for resource preparation, execution, and cleanup.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize ResourceManager

        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.registry: PluginRegistry = get_registry()
        self._active_plugin: Optional[Plugin] = None
        self._prepared_resources: Dict[int, Any] = {}
        self._resource_metadata: Dict[int, Dict[str, Any]] = {}

    def initialize(self, auto_discover: bool = True) -> None:
        """
        Initialize resource manager

        Args:
            auto_discover: Automatically discover and register plugins
        """
        if auto_discover:
            logger.info("Auto-discovering plugins...")
            self.registry.discover_plugins("rm_abstract.backends")

        # Select and initialize plugin
        device = self.config.device
        if device == "auto":
            self._active_plugin = self.registry.auto_select()
        else:
            device_type, device_id = self._parse_device(device)
            self._active_plugin = self.registry.get_plugin(
                device_type,
                device_id=device_id,
                cache_dir=self.config.cache_dir,
                compile_options=self.config.compile_options,
            )

        if self._active_plugin:
            self._active_plugin.initialize()
            logger.info(f"Initialized with plugin: {self._active_plugin.metadata().name}")
        else:
            logger.warning("No plugin available")

    def _parse_device(self, device: str) -> tuple[str, int]:
        """
        Parse device string

        Args:
            device: Device string (e.g., "gpu:0", "rbln:1")

        Returns:
            (device_type, device_id) tuple
        """
        if ":" in device:
            device_type, device_id_str = device.split(":", 1)
            try:
                device_id = int(device_id_str)
            except ValueError:
                device_id = 0
        else:
            device_type = device
            device_id = 0

        return device_type.lower(), device_id

    @property
    def active_plugin(self) -> Optional[Plugin]:
        """Get active plugin"""
        return self._active_plugin

    @property
    def device_name(self) -> str:
        """Get current device name"""
        if self._active_plugin is None:
            return "none"
        meta = self._active_plugin.metadata()
        return meta.name

    def prepare_resource(
        self, resource: Any, config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Prepare resource for use

        Args:
            resource: Resource to prepare (model, data, etc.)
            config: Resource-specific configuration

        Returns:
            Prepared resource
        """
        if self._active_plugin is None:
            logger.warning("No active plugin, returning original resource")
            return resource

        resource_id = id(resource)

        # Check if already prepared
        if resource_id in self._prepared_resources:
            logger.debug(f"Resource {resource_id} already prepared, returning cached")
            return self._prepared_resources[resource_id]

        # Prepare resource using active plugin
        try:
            prepared = self._active_plugin.prepare_resource(resource, config)
            self._prepared_resources[resource_id] = prepared
            self._resource_metadata[resource_id] = {
                "plugin": self._active_plugin.metadata().name,
                "config": config,
                "original_type": type(resource).__name__,
            }
            logger.info(
                f"Prepared resource {type(resource).__name__} "
                f"using {self._active_plugin.metadata().name}"
            )
            return prepared
        except Exception as e:
            logger.error(f"Failed to prepare resource: {e}")
            raise

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
        if self._active_plugin is None:
            raise RuntimeError("No active plugin")

        return self._active_plugin.execute(resource, inputs, **kwargs)

    def switch_plugin(
        self, plugin_name: str, device_id: int = 0, **kwargs
    ) -> None:
        """
        Switch to a different plugin at runtime

        Args:
            plugin_name: Name of plugin to switch to
            device_id: Device ID
            **kwargs: Plugin-specific configuration
        """
        # Cleanup current plugin
        if self._active_plugin:
            self._active_plugin.cleanup()

        # Clear cached resources
        self._prepared_resources.clear()
        self._resource_metadata.clear()

        # Get new plugin
        self._active_plugin = self.registry.get_plugin(
            plugin_name,
            device_id=device_id,
            cache_dir=self.config.cache_dir,
            compile_options=self.config.compile_options,
            **kwargs,
        )

        if self._active_plugin:
            self._active_plugin.initialize()
            logger.info(f"Switched to plugin: {plugin_name}")
        else:
            logger.error(f"Failed to switch to plugin: {plugin_name}")

    def list_available_plugins(self) -> List[PluginMetadata]:
        """
        List all available plugins

        Returns:
            List of plugin metadata
        """
        return self.registry.list_plugins(available_only=True)

    def get_plugin_info(self, plugin_name: str) -> Optional[PluginMetadata]:
        """
        Get information about a specific plugin

        Args:
            plugin_name: Plugin name

        Returns:
            Plugin metadata or None
        """
        plugin = self.registry.get_plugin(plugin_name)
        if plugin:
            return plugin.metadata()
        return None

    def get_resource_info(self, resource: Any) -> Optional[Dict[str, Any]]:
        """
        Get information about a prepared resource

        Args:
            resource: Resource object

        Returns:
            Resource metadata or None
        """
        resource_id = id(resource)
        return self._resource_metadata.get(resource_id)

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of active plugin

        Returns:
            Capability dictionary
        """
        if self._active_plugin is None:
            return {}
        return self._active_plugin.get_capabilities()

    def cleanup(self) -> None:
        """Cleanup all resources and plugins"""
        if self._active_plugin:
            self._active_plugin.cleanup()

        self._prepared_resources.clear()
        self._resource_metadata.clear()

        logger.info("ResourceManager cleaned up")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        return False


# Global resource manager instance
_global_manager: Optional[ResourceManager] = None


def get_manager() -> Optional[ResourceManager]:
    """Get global resource manager instance"""
    return _global_manager


def set_manager(manager: ResourceManager) -> None:
    """Set global resource manager instance"""
    global _global_manager
    _global_manager = manager
