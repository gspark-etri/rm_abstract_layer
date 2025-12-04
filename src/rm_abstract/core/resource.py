"""
Resource Model - Heterogeneous Resource Abstraction

Defines resources (GPU, NPU, PIM, CPU, Remote) with their capabilities,
build requirements, and runtime characteristics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Type of computational resource"""

    GPU = "gpu"
    NPU = "npu"
    PIM = "pim"  # Processing-In-Memory
    CPU = "cpu"
    REMOTE = "remote"  # Remote LLM service
    HYBRID = "hybrid"  # Hybrid resources (e.g., GPU+PIM)


@dataclass
class Resource:
    """
    Computational Resource

    Represents any type of compute resource (GPU, NPU, PIM, etc.)
    with its physical and logical characteristics.
    """

    id: str  # Unique identifier (e.g., "gpu-0", "npu-rbln-1")
    type: ResourceType
    attributes: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Common attributes (stored in attributes dict):
    # - vendor: str (e.g., "NVIDIA", "Rebellions", "FuriosaAI")
    # - device_id: int
    # - memory_gb: float
    # - bandwidth_gbps: float
    # - pcie_gen: int
    # - compute_capability: str
    # - driver_version: str

    def __post_init__(self):
        """Validate resource configuration"""
        if not self.id:
            raise ValueError("Resource id cannot be empty")

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get resource attribute with default"""
        return self.attributes.get(key, default)

    def has_tag(self, tag: str) -> bool:
        """Check if resource has specific tag"""
        return tag in self.tags

    def matches_tags(self, tags: List[str]) -> bool:
        """Check if resource matches all given tags"""
        return all(tag in self.tags for tag in tags)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "attributes": self.attributes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Resource":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            type=ResourceType(data["type"]),
            attributes=data.get("attributes", {}),
            tags=data.get("tags", []),
        )


@dataclass
class Capability:
    """
    Resource Capability

    Describes what a resource can do and its performance characteristics.
    """

    max_batch_size: int = 1
    max_seq_len: int = 2048
    support_kv_cache: bool = True
    support_streaming: bool = False
    dtype: List[str] = field(default_factory=lambda: ["fp16"])
    optimized_for: List[str] = field(
        default_factory=list
    )  # ["throughput", "latency", "energy"]

    # Advanced capabilities
    support_flash_attention: bool = False
    support_paged_attention: bool = False
    support_tensor_parallel: bool = False
    support_pipeline_parallel: bool = False

    # Build/Compilation capabilities
    requires_precompile: bool = False
    support_dynamic_shape: bool = True
    support_quantization: List[str] = field(
        default_factory=list
    )  # ["int8", "int4", "fp8"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "support_kv_cache": self.support_kv_cache,
            "support_streaming": self.support_streaming,
            "dtype": self.dtype,
            "optimized_for": self.optimized_for,
            "support_flash_attention": self.support_flash_attention,
            "support_paged_attention": self.support_paged_attention,
            "support_tensor_parallel": self.support_tensor_parallel,
            "support_pipeline_parallel": self.support_pipeline_parallel,
            "requires_precompile": self.requires_precompile,
            "support_dynamic_shape": self.support_dynamic_shape,
            "support_quantization": self.support_quantization,
        }


@dataclass
class BuildProfile:
    """
    Build/Compilation Profile

    Describes how to compile a model for a specific resource.
    Especially important for NPU/PIM that require pre-compilation.
    """

    target_resource_type: ResourceType
    compiler: str  # "tensorrt", "vendor_npu_compiler", "onnx_runtime", etc.
    compiler_version: str
    flags: Dict[str, Any] = field(default_factory=dict)
    interface: str = "python"  # "python", "cli", "c-api", "config-file"

    # Common flags (stored in flags dict):
    # - precision: str ("fp16", "int8", etc.)
    # - optimization_level: int (0-3)
    # - max_seq_len: int
    # - max_batch_size: int
    # - enable_kv_cache: bool
    # - quantization: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "target_resource_type": self.target_resource_type.value,
            "compiler": self.compiler,
            "compiler_version": self.compiler_version,
            "flags": self.flags,
            "interface": self.interface,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildProfile":
        """Create from dictionary"""
        return cls(
            target_resource_type=ResourceType(data["target_resource_type"]),
            compiler=data["compiler"],
            compiler_version=data["compiler_version"],
            flags=data.get("flags", {}),
            interface=data.get("interface", "python"),
        )


@dataclass
class BuildArtifact:
    """
    Build Artifact

    Represents a compiled/prepared model artifact for a specific resource.
    """

    id: str
    model_id: str  # Original model identifier
    build_profile: BuildProfile
    path: str  # Local path or remote URI
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Common metadata (stored in metadata dict):
    # - checksum: str
    # - created_at: str (ISO timestamp)
    # - size_bytes: int
    # - status: str ("valid", "expired", "corrupted")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "build_profile": self.build_profile.to_dict(),
            "path": self.path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildArtifact":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            model_id=data["model_id"],
            build_profile=BuildProfile.from_dict(data["build_profile"]),
            path=data["path"],
            metadata=data.get("metadata", {}),
        )


class ResourcePool:
    """
    Resource Pool

    Manages a collection of available resources and their capabilities.
    """

    def __init__(self):
        self._resources: Dict[str, Resource] = {}
        self._capabilities: Dict[str, Capability] = {}

    def add_resource(self, resource: Resource, capability: Optional[Capability] = None):
        """Add a resource to the pool"""
        self._resources[resource.id] = resource
        if capability:
            self._capabilities[resource.id] = capability
        logger.debug(f"Added resource: {resource.id} ({resource.type.value})")

    def remove_resource(self, resource_id: str):
        """Remove a resource from the pool"""
        if resource_id in self._resources:
            del self._resources[resource_id]
        if resource_id in self._capabilities:
            del self._capabilities[resource_id]
        logger.debug(f"Removed resource: {resource_id}")

    def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get resource by ID"""
        return self._resources.get(resource_id)

    def get_capability(self, resource_id: str) -> Optional[Capability]:
        """Get capability for resource"""
        return self._capabilities.get(resource_id)

    def list_resources(
        self,
        resource_type: Optional[ResourceType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Resource]:
        """
        List resources with optional filtering

        Args:
            resource_type: Filter by resource type
            tags: Filter by tags (must match all)

        Returns:
            List of matching resources
        """
        resources = list(self._resources.values())

        if resource_type:
            resources = [r for r in resources if r.type == resource_type]

        if tags:
            resources = [r for r in resources if r.matches_tags(tags)]

        return resources

    def find_best_resource(
        self,
        resource_type: Optional[ResourceType] = None,
        min_memory_gb: Optional[float] = None,
        required_tags: Optional[List[str]] = None,
    ) -> Optional[Resource]:
        """
        Find best resource matching criteria

        Args:
            resource_type: Required resource type
            min_memory_gb: Minimum memory requirement
            required_tags: Required tags

        Returns:
            Best matching resource or None
        """
        candidates = self.list_resources(resource_type=resource_type, tags=required_tags)

        if min_memory_gb:
            candidates = [
                r
                for r in candidates
                if r.get_attribute("memory_gb", 0) >= min_memory_gb
            ]

        if not candidates:
            return None

        # Sort by memory (descending) as tie-breaker
        candidates.sort(key=lambda r: r.get_attribute("memory_gb", 0), reverse=True)
        return candidates[0]

    def __len__(self) -> int:
        """Return number of resources"""
        return len(self._resources)

    def __iter__(self):
        """Iterate over resources"""
        return iter(self._resources.values())
