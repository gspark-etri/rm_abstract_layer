"""
Setup script for LLM Heterogeneous Resource Orchestrator
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read version
def get_version():
    """Get version from __init__.py"""
    init_file = Path(__file__).parent / "src" / "rm_abstract" / "__init__.py"
    with open(init_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"


# Read long description
def get_long_description():
    """Get long description from README"""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Core dependencies (always required)
CORE_DEPS = [
    "torch>=2.0.0",
    "numpy>=1.20.0",
]

# Optional dependencies for different backends
EXTRAS = {
    # GPU backend dependencies
    "gpu": [
        "vllm>=0.2.0",
        "transformers>=4.30.0",
    ],
    # NPU backend dependencies
    "npu-rbln": [
        "rbln>=0.1.0",  # Rebellions SDK
    ],
    "npu-furiosa": [
        "furiosa-sdk>=0.1.0",  # FuriosaAI SDK
    ],
    # Development dependencies
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "ruff>=0.1.0",
        "mypy>=1.0.0",
    ],
    # Documentation dependencies
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "myst-parser>=0.18.0",
    ],
}

# All optional dependencies
EXTRAS["all"] = list(set(sum(EXTRAS.values(), [])))


setup(
    name="rm-abstract",
    version=get_version(),
    author="RM Abstract Team",
    author_email="contact@rmabstract.dev",
    description="LLM Heterogeneous Resource Orchestrator - GPU/NPU/PIM Meta-Serving Layer",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rm_abstract_layer",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/rm_abstract_layer/issues",
        "Source": "https://github.com/yourusername/rm_abstract_layer",
        "Documentation": "https://rm-abstract.readthedocs.io",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=CORE_DEPS,
    extras_require=EXTRAS,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="llm gpu npu pim inference serving orchestration heterogeneous",
    entry_points={
        "console_scripts": [
            "rm-abstract=rm_abstract.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
