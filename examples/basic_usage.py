"""
기본 사용 예시

RM Abstract Layer의 기본적인 사용 방법을 보여줍니다.
"""

# 1. RM Abstract Layer 초기화 (기존 코드 상단에 추가)
import rm_abstract

# 디바이스 설정:
# - "auto": 자동 선택 (NPU > GPU > CPU)
# - "gpu:0": 첫 번째 GPU
# - "rbln:0": Rebellions ATOM NPU
# - "furiosa:0": FuriosaAI RNGD NPU
# - "cpu": CPU
rm_abstract.init(device="auto", verbose=True)

# 2. 이하 기존 코드 그대로 사용
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델 로드 (자동으로 백엔드에 맞게 준비됨)
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 추론 실행
inputs = tokenizer("Hello, I am a language model,", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 디바이스 정보 확인
print("\nDevice Info:")
print(rm_abstract.get_device_info())
