"""
디바이스 전환 예시

런타임에 디바이스를 전환하는 방법을 보여줍니다.
"""

import rm_abstract
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPU로 시작
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

# NPU로 전환
print("\n" + "=" * 50)
print("Switching to NPU...")
print("=" * 50)

rm_abstract.switch_device("rbln:0")

# 동일한 코드로 NPU에서 추론
model = AutoModelForCausalLM.from_pretrained(model_name)
inputs = tokenizer("NPU inference test:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 디바이스 정보 확인
print("\nCurrent Device:")
print(rm_abstract.get_device_info())
