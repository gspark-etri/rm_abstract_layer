"""
Basic Usage Example

Demonstrates basic usage of RM Abstract Layer.
"""

# 1. Initialize RM Abstract Layer (add at the top of existing code)
import rm_abstract

# Device settings:
# - "auto": Auto-select (NPU > GPU > CPU)
# - "gpu:0": First GPU
# - "rbln:0": Rebellions ATOM NPU
# - "furiosa:0": FuriosaAI RNGD NPU
# - "cpu": CPU
rm_abstract.init(device="auto", verbose=True)

# 2. Use existing code as-is below
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model (automatically prepared for the backend)
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Execute inference
inputs = tokenizer("Hello, I am a language model,", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Check device info
print("\nDevice Info:")
print(rm_abstract.get_device_info())
