"""
Device Switching Example

Demonstrates how to switch devices at runtime.
"""

import rm_abstract
from transformers import AutoModelForCausalLM, AutoTokenizer

# Start with GPU
rm_abstract.init(device="gpu:0", verbose=True)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("=" * 50)
print("Running on GPU...")
print("=" * 50)

model = AutoModelForCausalLM.from_pretrained(model_name)
inputs = tokenizer("GPU inference test:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Switch to NPU
print("\n" + "=" * 50)
print("Switching to NPU...")
print("=" * 50)

rm_abstract.switch_device("rbln:0")

# Inference on NPU with same code
model = AutoModelForCausalLM.from_pretrained(model_name)
inputs = tokenizer("NPU inference test:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Check device info
print("\nCurrent Device:")
print(rm_abstract.get_device_info())
