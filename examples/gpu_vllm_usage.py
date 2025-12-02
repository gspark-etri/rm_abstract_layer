"""
GPU (vLLM) Usage Example

Demonstrates how to use RM Abstract Layer with GPU/vLLM backend.
This example shows the "Zero Code Change" principle - existing HuggingFace
code works transparently with vLLM acceleration.

Requirements:
    pip install torch vllm transformers

Usage:
    python examples/gpu_vllm_usage.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import rm_abstract

# Check available backends first
print("=" * 60)
print("Checking available backends...")
print("=" * 60)

backends = rm_abstract.get_available_backends()
for name, available in backends.items():
    status = "✓ Available" if available else "✗ Not available"
    print(f"  {name}: {status}")

print()

# Example 1: Basic GPU usage with HuggingFace-style code
print("=" * 60)
print("Example 1: Zero Code Change - HuggingFace style")
print("=" * 60)

try:
    # Initialize with GPU
    # If GPU/vLLM is not available, falls back to CPU
    rm_abstract.init(device="auto", verbose=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model - automatically wrapped in ModelProxy
    model_name = "gpt2"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"Model type: {type(model)}")

    # Standard HuggingFace generate() call
    # This is transparently routed to the backend (vLLM or CPU)
    inputs = tokenizer("The future of AI is", return_tensors="pt")
    print(f"\nInput: 'The future of AI is'")

    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: {generated_text}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()

print()

# Example 2: Check device info
print("=" * 60)
print("Example 2: Device Information")
print("=" * 60)

try:
    info = rm_abstract.get_device_info()
    print(f"  Device Type: {info.get('device_type', 'N/A')}")
    print(f"  Device ID: {info.get('device_id', 'N/A')}")
    print(f"  Name: {info.get('name', 'N/A')}")
    print(f"  Vendor: {info.get('vendor', 'N/A')}")
    if info.get("memory_total"):
        print(f"  Memory: {info['memory_total'] / 1e9:.1f} GB")
except Exception as e:
    print(f"  Could not get device info: {e}")

print()

# Example 3: Device switching
print("=" * 60)
print("Example 3: Runtime Device Switching")
print("=" * 60)

try:
    controller = rm_abstract.get_controller()
    if controller:
        print(f"  Current device: {controller.device_name}")

        # You can switch devices at runtime
        # rm_abstract.switch_device("cpu")
        # print(f"  Switched to: {controller.device_name}")
        print("  (Device switching available via rm_abstract.switch_device())")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=" * 60)
print("Done!")
print("=" * 60)
