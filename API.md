# REST API 레퍼런스

RM Abstract Layer REST API 문서입니다. OpenAI API와 호환됩니다.

---

## 🚀 서버 시작

```bash
# 기본 실행
python -m rm_abstract.api

# 포트 지정
python -m rm_abstract.api --port 8000

# Uvicorn 직접 사용
uvicorn rm_abstract.api.server:app --host 0.0.0.0 --port 8000 --reload
```

**서버 정보:**
- API 문서: http://localhost:8000/docs
- OpenAPI 스펙: http://localhost:8000/openapi.json

---

## 📋 엔드포인트 목록

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 서버 정보 |
| GET | `/health` | 헬스 체크 |
| GET | `/v1/models` | 모델 목록 |
| GET | `/v1/models/{model_id}` | 모델 정보 |
| POST | `/v1/completions` | 텍스트 생성 |
| POST | `/v1/chat/completions` | 채팅 생성 |
| GET | `/v1/devices/status` | 시스템 상태 |
| POST | `/v1/devices/switch` | 디바이스 전환 |

---

## 🔧 API 상세

### 헬스 체크

```http
GET /health
```

**응답:**
```json
{
  "status": "ok",
  "initialized": true
}
```

---

### 모델 목록

```http
GET /v1/models
```

**응답:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt2",
      "object": "model",
      "created": 1234567890,
      "owned_by": "rm-abstract"
    }
  ]
}
```

---

### 텍스트 생성 (Completions)

```http
POST /v1/completions
Content-Type: application/json
```

**요청:**
```json
{
  "model": "gpt2",
  "prompt": "Hello, I am",
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "n": 1,
  "stop": ["\n"]
}
```

**파라미터:**

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| model | string | 필수 | 모델 ID |
| prompt | string/array | 필수 | 입력 프롬프트 |
| max_tokens | integer | 100 | 최대 생성 토큰 수 |
| temperature | float | 1.0 | 샘플링 온도 (0-2) |
| top_p | float | 1.0 | Nucleus 샘플링 (0-1) |
| n | integer | 1 | 생성할 응답 수 |
| stop | string/array | null | 중지 시퀀스 |

**응답:**
```json
{
  "id": "cmpl-1234567890",
  "object": "text_completion",
  "created": 1234567890,
  "model": "gpt2",
  "choices": [
    {
      "text": " a language model trained by...",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 50,
    "total_tokens": 54
  }
}
```

---

### 채팅 생성 (Chat Completions)

```http
POST /v1/chat/completions
Content-Type: application/json
```

**요청:**
```json
{
  "model": "gpt2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"}
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

**메시지 역할:**

| 역할 | 설명 |
|------|------|
| system | 시스템 프롬프트 |
| user | 사용자 메시지 |
| assistant | 어시스턴트 응답 |

**응답:**
```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "AI, or Artificial Intelligence, is..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 100,
    "total_tokens": 120
  }
}
```

---

### 시스템 상태

```http
GET /v1/devices/status
```

**응답:**
```json
{
  "status": "ok",
  "current_device": "gpu:0",
  "current_backend": "VLLMBackend",
  "available_devices": [
    {
      "device_type": "gpu",
      "device_id": 0,
      "name": "NVIDIA GeForce RTX 3090",
      "vendor": "NVIDIA",
      "memory_total_gb": 24.0,
      "memory_free_gb": 20.0
    },
    {
      "device_type": "cpu",
      "device_id": 0,
      "name": "x86_64",
      "memory_total_gb": 128.0,
      "memory_free_gb": 100.0
    }
  ],
  "available_backends": [
    {
      "name": "gpu",
      "display_name": "vLLM GPU Backend",
      "available": true,
      "device_type": "GPU",
      "version": "0.12.0"
    },
    {
      "name": "cpu",
      "display_name": "PyTorch CPU Backend",
      "available": true,
      "device_type": "CPU"
    }
  ]
}
```

---

### 디바이스 전환

```http
POST /v1/devices/switch
Content-Type: application/json
```

**요청:**
```json
{
  "device": "cpu"
}
```

**디바이스 옵션:**

| 값 | 설명 |
|----|------|
| `gpu:0` | GPU 0번 |
| `gpu:1` | GPU 1번 |
| `cpu` | CPU |
| `rbln:0` | Rebellions NPU 0번 |
| `auto` | 자동 선택 |

**응답:**
```json
{
  "success": true,
  "previous_device": "gpu:0",
  "current_device": "cpu:0",
  "message": "Switched from gpu:0 to cpu:0"
}
```

---

## 💻 사용 예제

### cURL

```bash
# 텍스트 생성
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "Hello",
    "max_tokens": 50
  }'

# 채팅
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [{"role": "user", "content": "Hi!"}]
  }'

# 디바이스 전환
curl -X POST http://localhost:8000/v1/devices/switch \
  -H "Content-Type: application/json" \
  -d '{"device": "cpu"}'
```

### Python (requests)

```python
import requests

BASE_URL = "http://localhost:8000"

# 텍스트 생성
response = requests.post(
    f"{BASE_URL}/v1/completions",
    json={
        "model": "gpt2",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0.7,
    }
)
print(response.json()["choices"][0]["text"])

# 채팅
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "gpt2",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
        ],
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

# RM Abstract API 서버 사용
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # API 키 불필요
)

# 텍스트 생성
response = client.completions.create(
    model="gpt2",
    prompt="Hello, I am",
    max_tokens=50,
)
print(response.choices[0].text)

# 채팅
response = client.chat.completions.create(
    model="gpt2",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
)
print(response.choices[0].message.content)
```

---

## ❌ 에러 응답

```json
{
  "error": {
    "message": "Model not found: invalid-model",
    "type": "invalid_request_error",
    "param": "model",
    "code": "404"
  }
}
```

**에러 코드:**

| 코드 | 설명 |
|------|------|
| 400 | 잘못된 요청 |
| 404 | 모델을 찾을 수 없음 |
| 500 | 서버 내부 오류 |

---

## 🔗 관련 문서

- [QUICKSTART.md](QUICKSTART.md) - 빠른 시작
- [INSTALL.md](INSTALL.md) - 설치 가이드

