"""
Binary Compiler and Runtime Adapters

Adapters for integrating closed-source, binary-only NPU/PIM stacks
that only provide CLI tools or C API interfaces.
"""

import subprocess
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path
import tempfile

from .resource import BuildProfile, BuildArtifact

logger = logging.getLogger(__name__)


class BinaryCompilerAdapter(ABC):
    """
    Binary Compiler Adapter

    Wraps closed-source compilers that are provided as:
    - CLI executables
    - Configuration files
    - Limited C APIs

    Handles the black-box compilation process for NPU/PIM devices.
    """

    def __init__(self, build_profile: BuildProfile):
        """
        Initialize compiler adapter

        Args:
            build_profile: Build profile with compiler settings
        """
        self.build_profile = build_profile
        self.compiler_path = self._find_compiler()

    @abstractmethod
    def _find_compiler(self) -> Optional[str]:
        """
        Find compiler executable or library

        Returns:
            Path to compiler or None if not found
        """
        ...

    @abstractmethod
    def compile(
        self, input_path: str, output_path: str, **kwargs
    ) -> BuildArtifact:
        """
        Compile model to target format

        Args:
            input_path: Input model path (e.g., ONNX file)
            output_path: Output artifact path
            **kwargs: Additional compilation options

        Returns:
            BuildArtifact representing compiled model

        Raises:
            RuntimeError: If compilation fails
        """
        ...

    def validate_input(self, input_path: str) -> bool:
        """
        Validate input model format

        Args:
            input_path: Path to input model

        Returns:
            True if valid
        """
        return os.path.exists(input_path)

    def check_availability(self) -> bool:
        """
        Check if compiler is available

        Returns:
            True if compiler can be used
        """
        return self.compiler_path is not None and os.path.exists(self.compiler_path)


class CLICompilerAdapter(BinaryCompilerAdapter):
    """
    CLI-based Compiler Adapter

    For compilers provided as command-line tools.

    Example:
        npu_compile --input model.onnx --output engine.bin \
                    --config config.yaml --precision fp16
    """

    def __init__(
        self,
        build_profile: BuildProfile,
        cli_name: str,
        search_paths: Optional[List[str]] = None,
    ):
        """
        Initialize CLI compiler adapter

        Args:
            build_profile: Build profile
            cli_name: Name of CLI executable
            search_paths: Additional paths to search for executable
        """
        self.cli_name = cli_name
        self.search_paths = search_paths or []
        super().__init__(build_profile)

    def _find_compiler(self) -> Optional[str]:
        """Find CLI compiler executable"""
        # Check PATH
        import shutil

        compiler = shutil.which(self.cli_name)
        if compiler:
            return compiler

        # Check additional search paths
        for path in self.search_paths:
            candidate = os.path.join(path, self.cli_name)
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                return candidate

        logger.warning(f"CLI compiler '{self.cli_name}' not found")
        return None

    def compile(
        self, input_path: str, output_path: str, **kwargs
    ) -> BuildArtifact:
        """Compile using CLI tool"""
        if not self.check_availability():
            raise RuntimeError(f"Compiler '{self.cli_name}' not available")

        if not self.validate_input(input_path):
            raise ValueError(f"Input model not found: {input_path}")

        # Build command
        cmd = self._build_command(input_path, output_path, **kwargs)

        # Execute compilation
        logger.info(f"Running compilation: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=kwargs.get("timeout", 600),  # 10 min default
            )
            logger.info(f"Compilation successful: {output_path}")
            logger.debug(f"Compiler output: {result.stdout}")

            # Create artifact
            artifact = BuildArtifact(
                id=f"artifact_{os.path.basename(output_path)}",
                model_id=kwargs.get("model_id", "unknown"),
                build_profile=self.build_profile,
                path=output_path,
                metadata={
                    "compiler_stdout": result.stdout,
                    "compiler_stderr": result.stderr,
                    "size_bytes": os.path.getsize(output_path),
                },
            )
            return artifact

        except subprocess.TimeoutExpired as e:
            logger.error(f"Compilation timeout after {e.timeout}s")
            raise RuntimeError(f"Compilation timeout: {e}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Compilation failed: {e.stderr}")
            raise RuntimeError(f"Compilation failed: {e.stderr}")

    def _build_command(
        self, input_path: str, output_path: str, **kwargs
    ) -> List[str]:
        """
        Build CLI command

        Override this for specific compiler command formats.

        Args:
            input_path: Input model path
            output_path: Output path
            **kwargs: Additional options

        Returns:
            Command as list of strings
        """
        cmd = [self.compiler_path, "--input", input_path, "--output", output_path]

        # Add flags from build profile
        flags = self.build_profile.flags
        for key, value in flags.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

        return cmd


class ConfigFileCompilerAdapter(BinaryCompilerAdapter):
    """
    Config-file based Compiler Adapter

    For compilers that use configuration files instead of CLI arguments.

    Example:
        npu_compile --config compile_config.json
    """

    def __init__(
        self,
        build_profile: BuildProfile,
        cli_name: str,
        config_format: str = "json",
    ):
        """
        Initialize config-file compiler adapter

        Args:
            build_profile: Build profile
            cli_name: Name of CLI executable
            config_format: Config file format ("json", "yaml", "toml")
        """
        self.cli_name = cli_name
        self.config_format = config_format
        super().__init__(build_profile)

    def _find_compiler(self) -> Optional[str]:
        """Find compiler executable"""
        import shutil

        return shutil.which(self.cli_name)

    def compile(
        self, input_path: str, output_path: str, **kwargs
    ) -> BuildArtifact:
        """Compile using config file"""
        if not self.check_availability():
            raise RuntimeError(f"Compiler '{self.cli_name}' not available")

        # Create temporary config file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f".{self.config_format}",
            delete=False,
        ) as f:
            config = self._create_config(input_path, output_path, **kwargs)
            if self.config_format == "json":
                json.dump(config, f, indent=2)
            else:
                raise NotImplementedError(f"Format {self.config_format} not supported")
            config_path = f.name

        try:
            # Run compiler with config file
            cmd = [self.compiler_path, "--config", config_path]
            logger.info(f"Running compilation with config: {config_path}")

            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=600
            )

            logger.info(f"Compilation successful: {output_path}")

            artifact = BuildArtifact(
                id=f"artifact_{os.path.basename(output_path)}",
                model_id=kwargs.get("model_id", "unknown"),
                build_profile=self.build_profile,
                path=output_path,
                metadata={
                    "config_file": config_path,
                    "compiler_stdout": result.stdout,
                },
            )
            return artifact

        finally:
            # Cleanup temp config
            try:
                os.unlink(config_path)
            except Exception:
                pass

    def _create_config(
        self, input_path: str, output_path: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Create configuration dictionary

        Override for specific compiler config formats.

        Args:
            input_path: Input model path
            output_path: Output path
            **kwargs: Additional options

        Returns:
            Configuration dictionary
        """
        return {
            "input": input_path,
            "output": output_path,
            **self.build_profile.flags,
        }


class BinaryRuntimeAdapter(ABC):
    """
    Binary Runtime Adapter

    Wraps closed-source runtimes that provide limited C APIs
    for loading and executing compiled models.
    """

    def __init__(
        self,
        engine_path: str,
        device_id: int = 0,
        runtime_so: Optional[str] = None,
    ):
        """
        Initialize runtime adapter

        Args:
            engine_path: Path to compiled engine/model
            device_id: Device ID to use
            runtime_so: Path to runtime shared library
        """
        self.engine_path = engine_path
        self.device_id = device_id
        self.runtime_so = runtime_so
        self._handle = None

    @abstractmethod
    def load(self) -> None:
        """
        Load compiled model into runtime

        Raises:
            RuntimeError: If loading fails
        """
        ...

    @abstractmethod
    def run(self, inputs: Any, **kwargs) -> Any:
        """
        Execute inference

        Args:
            inputs: Input data
            **kwargs: Runtime options

        Returns:
            Inference results

        Raises:
            RuntimeError: If execution fails
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources"""
        ...

    def __enter__(self):
        """Context manager entry"""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.unload()
        return False


class DummyRuntimeAdapter(BinaryRuntimeAdapter):
    """
    Dummy Runtime Adapter for testing

    Simulates a binary-only runtime without actual device.
    """

    def load(self) -> None:
        """Load (dummy)"""
        logger.info(f"[DummyRuntime] Loading engine: {self.engine_path}")
        if not os.path.exists(self.engine_path):
            raise RuntimeError(f"Engine not found: {self.engine_path}")
        self._handle = "dummy_handle"

    def run(self, inputs: Any, **kwargs) -> Any:
        """Run (dummy)"""
        if self._handle is None:
            raise RuntimeError("Runtime not loaded")
        logger.info(f"[DummyRuntime] Running inference on device {self.device_id}")
        return {"output": f"dummy_result_for_{inputs}"}

    def unload(self) -> None:
        """Unload (dummy)"""
        logger.info("[DummyRuntime] Unloading engine")
        self._handle = None
