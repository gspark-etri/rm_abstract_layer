"""
Serving Engines Demo

Demonstrates how to use different serving engines:
- vLLM: High-performance LLM serving
- Triton: NVIDIA Triton Inference Server
- TorchServe: PyTorch native serving

Each engine supports multiple devices:
- GPU: NVIDIA CUDA
- NPU (RBLN): Rebellions ATOM
- CPU: Fallback

Usage:
    python examples/serving_engines_demo.py
    
Requirements:
    pip install vllm transformers torch
    # For Triton: pip install tritonclient[all]
    # For TorchServe: pip install torchserve torch-model-archiver
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_available_engines():
    """Check which serving engines are available"""
    from rm_abstract.serving import get_available_engines
    
    print("=" * 60)
    print("Available Serving Engines")
    print("=" * 60)
    
    engines = get_available_engines()
    
    for name, info in engines.items():
        status = "✓ Available" if info["available"] else "✗ Not available"
        print(f"\n  {name}:")
        print(f"    Status: {status}")
        print(f"    Devices: {', '.join(info['supported_devices'])}")
    
    print()
    return engines


def demo_vllm_engine():
    """Demo: vLLM Serving Engine"""
    from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
    
    print("=" * 60)
    print("Demo 1: vLLM Serving Engine")
    print("=" * 60)
    
    try:
        # Create vLLM engine
        config = ServingConfig(
            engine=ServingEngineType.VLLM,
            device=DeviceTarget.GPU,
            model_name="gpt2",
            port=8000,
        )
        
        engine = create_serving_engine(config)
        print(f"\n  Engine: {engine.name}")
        print(f"  Device: {config.device.value}")
        
        # Load model
        print("\n  Loading model...")
        engine.load_model("gpt2")
        
        # Run inference (max_tokens is for inference, not model loading)
        prompt = "The future of AI is"
        print(f"\n  [Input]  \"{prompt}\"")
        
        output = engine.infer(prompt, max_tokens=30, temperature=0.7)
        print(f"  [Output] \"{output}\"")
        
        print("\n  ✓ vLLM demo completed!")
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
    
    print()


def demo_vllm_with_rbln():
    """Demo: vLLM with RBLN NPU"""
    from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
    
    print("=" * 60)
    print("Demo 2: vLLM with RBLN NPU")
    print("=" * 60)
    
    try:
        config = ServingConfig(
            engine=ServingEngineType.VLLM,
            device=DeviceTarget.NPU_RBLN,  # Use Rebellions NPU
            model_name="gpt2",
            device_id=0,
        )
        
        engine = create_serving_engine(config)
        print(f"\n  Engine: {engine.name}")
        print(f"  Device: {config.device.value}")
        
        # Load model (compiled for RBLN)
        print("\n  Loading model for RBLN NPU...")
        engine.load_model("gpt2")
        
        # Run inference
        prompt = "Machine learning enables"
        print(f"\n  [Input]  \"{prompt}\"")
        
        output = engine.infer(prompt, max_tokens=30)
        print(f"  [Output] \"{output}\"")
        
        print("\n  ✓ vLLM-RBLN demo completed!")
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        print("  Note: RBLN NPU requires Rebellions SDK and hardware")
    
    print()


def demo_triton_setup():
    """Demo: Triton Inference Server Setup"""
    from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
    
    print("=" * 60)
    print("Demo 3: Triton Inference Server")
    print("=" * 60)
    
    try:
        config = ServingConfig(
            engine=ServingEngineType.TRITON,
            device=DeviceTarget.GPU,
            model_name="llm_model",
            port=8000,
        )
        
        engine = create_serving_engine(config)
        print(f"\n  Engine: {engine.name}")
        print(f"  Device: {config.device.value}")
        
        # Setup model repository
        print("\n  Setting up model repository...")
        engine.setup_model_repository("/tmp/triton_models")
        
        # Prepare model (creates config and handler)
        print("  Preparing model for Triton...")
        engine.load_model("gpt2", triton_model_name="gpt2_llm")
        
        print("\n  ✓ Triton model prepared!")
        print("  To start server, run:")
        print("    docker run --gpus=1 --rm -p8000:8000 \\")
        print("      -v /tmp/triton_models:/models \\")
        print("      nvcr.io/nvidia/tritonserver:latest \\")
        print("      tritonserver --model-repository=/models")
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
    
    print()


def demo_torchserve_setup():
    """Demo: TorchServe Setup"""
    from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
    
    print("=" * 60)
    print("Demo 4: TorchServe")
    print("=" * 60)
    
    try:
        config = ServingConfig(
            engine=ServingEngineType.TORCHSERVE,
            device=DeviceTarget.GPU,
            model_name="gpt2_model",
            port=8080,
        )
        
        engine = create_serving_engine(config)
        print(f"\n  Engine: {engine.name}")
        print(f"  Device: {config.device.value}")
        
        # Create model archive
        print("\n  Creating model archive (.mar)...")
        mar_path = engine.create_model_archive("gpt2", archive_name="gpt2_model")
        print(f"  Archive created: {mar_path}")
        
        print("\n  ✓ TorchServe model archive created!")
        print("  To start server, run:")
        print(f"    torchserve --start --model-store {engine._model_store} --models all")
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        if "torch-model-archiver" in str(e):
            print("  Install TorchServe: pip install torchserve torch-model-archiver")
    
    print()


def demo_engine_comparison():
    """Demo: Compare different engines"""
    from rm_abstract.serving import get_available_engines, ServingEngineType
    
    print("=" * 60)
    print("Demo 5: Engine Comparison")
    print("=" * 60)
    
    comparison = """
    ┌─────────────────┬────────────────┬─────────────────┬──────────────┐
    │ Feature         │ vLLM           │ Triton          │ TorchServe   │
    ├─────────────────┼────────────────┼─────────────────┼──────────────┤
    │ Performance     │ ⭐⭐⭐ (Best)   │ ⭐⭐⭐           │ ⭐⭐          │
    │ Ease of Use     │ ⭐⭐⭐          │ ⭐⭐             │ ⭐⭐⭐        │
    │ Multi-model     │ ⭐             │ ⭐⭐⭐ (Best)    │ ⭐⭐⭐        │
    │ GPU Support     │ ✓              │ ✓               │ ✓            │
    │ RBLN NPU        │ ✓              │ ✓               │ ✓            │
    │ OpenAI API      │ ✓ (Built-in)   │ ⚠ (Custom)      │ ⚠ (Custom)   │
    │ Batching        │ Continuous     │ Dynamic         │ Static       │
    └─────────────────┴────────────────┴─────────────────┴──────────────┘
    
    Recommendations:
    - LLM Serving: vLLM (best performance for text generation)
    - Multi-model: Triton (supports multiple models simultaneously)
    - Simple Setup: TorchServe (PyTorch native, easy to get started)
    - RBLN NPU: All three support Rebellions ATOM NPU
    """
    
    print(comparison)


def demo_switch_engines():
    """Demo: Switching between engines"""
    from rm_abstract.serving import (
        create_serving_engine, 
        ServingConfig, 
        ServingEngineType, 
        DeviceTarget,
        get_available_engines,
    )
    
    print("=" * 60)
    print("Demo 6: Switching Between Engines")
    print("=" * 60)
    
    engines = get_available_engines()
    model_name = "gpt2"
    prompt = "Hello, I am"
    
    print(f"\n  Model: {model_name}")
    print(f"  Prompt: \"{prompt}\"")
    print()
    
    # Test each available engine
    for engine_type in [ServingEngineType.VLLM, ServingEngineType.TORCHSERVE]:
        engine_info = engines.get(engine_type.value)
        
        if not engine_info or not engine_info["available"]:
            print(f"  [{engine_type.value}] ✗ Not available")
            continue
        
        try:
            config = ServingConfig(
                engine=engine_type,
                device=DeviceTarget.GPU if "gpu" in engine_info["supported_devices"] else DeviceTarget.CPU,
                model_name=model_name,
            )
            
            engine = create_serving_engine(config)
            engine.load_model(model_name)
            
            output = engine.infer(prompt, max_tokens=20)
            print(f"  [{engine_type.value}] Output: \"{output[:50]}...\"")
            
        except Exception as e:
            print(f"  [{engine_type.value}] ✗ Error: {e}")
    
    print()


def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("  Serving Engines Demo - RM Abstract Layer")
    print("=" * 60 + "\n")
    
    # Check available engines
    engines = check_available_engines()
    
    # Run demos based on availability
    if engines.get("vllm", {}).get("available"):
        demo_vllm_engine()
    
    # RBLN demo (will show error if not available)
    demo_vllm_with_rbln()
    
    # Setup demos (don't require running servers)
    if engines.get("triton", {}).get("available"):
        demo_triton_setup()
    
    if engines.get("torchserve", {}).get("available"):
        demo_torchserve_setup()
    
    # Comparison
    demo_engine_comparison()
    
    print("=" * 60)
    print("All demos completed!")
    print("=" * 60)


# IMPORTANT: Required for multiprocessing
if __name__ == "__main__":
    main()

