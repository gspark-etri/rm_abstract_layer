# 빠른 시작 가이드

예제 중심으로 RM Abstract Layer 사용법을 알아봅니다.

---

## 🎯 5분 만에 시작하기

### 1. 설치

```bash
pip install -e ".[gpu]"
```

### 2. 시스템 확인

```bash
python -m rm_abstract.system_validator --quick
```

### 3. 첫 번째 예제

```python
import rm_abstract
from transformers import AutoModelForCausalLM, AutoTokenizer

# 초기화
rm_abstract.init(device="auto")

# 모델 로드 및 추론
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer("Hello, I am", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0]))
```

---

## 📚 예제 모음

### 예제 1: GPU/vLLM 사용

```python
import rm_abstract
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPU 초기화
rm_abstract.init(device="gpu:0", verbose=True)

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 텍스트 생성
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 예제 2: CPU로 전환

```python
import rm_abstract
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPU로 시작
rm_abstract.init(device="gpu:0")
print(f"현재 디바이스: {rm_abstract.get_controller().device_name}")

# CPU로 전환
rm_abstract.switch_device("cpu")
print(f"전환 후: {rm_abstract.get_controller().device_name}")

# CPU에서 추론
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("Hello", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

### 예제 3: 시스템 정보 확인

```python
import rm_abstract

# 전체 시스템 정보
rm_abstract.print_system_info()

# 상세 정보 가져오기
info = rm_abstract.get_system_info()
print(f"GPU 개수: {len(info.gpus)}")
print(f"NPU 개수: {len(info.npus)}")

# 사용 가능한 백엔드
backends = rm_abstract.get_available_backends()
for name, available in backends.items():
    status = "✓" if available else "✗"
    print(f"  {status} {name}")
```

### 예제 4: REST API 서버

```python
# 서버 시작 (별도 터미널)
# python -m rm_abstract.api --port 8000

import requests

# 텍스트 생성
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "gpt2",
        "prompt": "Hello, I am",
        "max_tokens": 30,
    }
)
print(response.json()["choices"][0]["text"])

# 채팅
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt2",
        "messages": [
            {"role": "user", "content": "What is AI?"}
        ],
        "max_tokens": 50,
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

### 예제 5: 서빙 엔진 사용

```python
from rm_abstract.serving import (
    create_serving_engine,
    ServingConfig,
    ServingEngineType,
    DeviceTarget,
)

# vLLM 엔진
config = ServingConfig(
    engine=ServingEngineType.VLLM,
    device=DeviceTarget.GPU,
)
engine = create_serving_engine(config)
engine.load_model("gpt2")
output = engine.infer("Hello, I am", max_tokens=30)
print(output)
```

---

## 🔧 디바이스 옵션

```python
import rm_abstract

# 자동 선택 (NPU > GPU > CPU)
rm_abstract.init(device="auto")

# 특정 GPU
rm_abstract.init(device="gpu:0")
rm_abstract.init(device="gpu:1")

# CPU
rm_abstract.init(device="cpu")

# Rebellions NPU
rm_abstract.init(device="rbln:0")
```

---

## 📁 예제 파일

```bash
# 예제 실행
python examples/gpu_vllm_usage.py
python examples/serving_engines_demo.py
```

| 파일 | 설명 |
|------|------|
| `gpu_vllm_usage.py` | GPU/vLLM 사용 및 디바이스 스위칭 |
| `serving_engines_demo.py` | vLLM, Triton, TorchServe 비교 |

---

## ❓ 자주 묻는 질문

### Q: 어떤 디바이스가 사용되나요?

```python
import rm_abstract

rm_abstract.init(device="auto")
info = rm_abstract.get_device_info()
print(f"디바이스: {info['device_type']}:{info['device_id']}")
```

### Q: GPU 메모리가 부족해요

```bash
# 다른 GPU 사용
CUDA_VISIBLE_DEVICES=1 python script.py

# 또는 CPU 사용
rm_abstract.init(device="cpu")
```

### Q: 기존 코드를 수정해야 하나요?

아니요! `rm_abstract.init()` 한 줄만 추가하면 됩니다:

```python
import rm_abstract
rm_abstract.init()  # 이 한 줄만 추가

# 기존 코드 그대로 사용
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

---

## 🔗 다음 단계

- [INSTALL.md](INSTALL.md) - 상세 설치 가이드
- [API.md](API.md) - REST API 레퍼런스
- [ARCHITECTURE.md](ARCHITECTURE.md) - 시스템 아키텍처
