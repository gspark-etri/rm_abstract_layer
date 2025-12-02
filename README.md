# RM Abstract Layer

이종 AI 반도체 통합 호환 라이브러리 - GPU/NPU 추상화 레이어

## 개요

기존 GPU 추론 스크립트를 **코드 수정 없이** NPU/GPU 어디서든 실행 가능하도록 하는 추상화 레이어 라이브러리입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                  기존 사용자 추론 코드 (수정 없음)             │
├─────────────────────────────────────────────────────────────┤
│                 RM Abstract Layer (1줄 추가)                 │
│    import rm_abstract; rm_abstract.init(device="npu:0")     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┬─────────────┬─────────────┐                │
│  │  vLLM/CUDA  │  RBLN SDK   │ Furiosa SDK │                │
│  │    (GPU)    │ (Rebellions)│ (FuriosaAI) │                │
│  └─────────────┴─────────────┴─────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## 특징

- **Zero Code Change**: 기존 추론 코드 수정 없이 1줄 추가만으로 동작
- **투명한 컴파일**: NPU 컴파일이 필요한 경우 자동으로 처리 (캐싱 지원)
- **디바이스 자동 선택**: `device="auto"`로 최적의 디바이스 자동 선택
- **플러그인 아키텍처**: 새로운 NPU 벤더 쉽게 추가 가능

## 지원 환경

| 디바이스 | 추론 엔진 | NPU 모델 |
|----------|-----------|----------|
| GPU | vLLM | NVIDIA CUDA |
| NPU (Rebellions) | RBLN Runtime | ATOM |
| NPU (FuriosaAI) | Furiosa Runtime | RNGD |
| CPU | PyTorch | - |

## 설치

```bash
pip install rm-abstract
```

## 빠른 시작

### 기본 사용법

```python
# 1줄만 추가하면 됩니다!
import rm_abstract
rm_abstract.init(device="npu:0")  # 또는 "gpu:0", "auto"

# 이하 기존 코드 그대로 사용
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

### 환경 변수로 설정

```bash
export RM_DEVICE="npu:0"
export RM_CACHE_DIR="/data/npu_cache"
python your_inference_script.py
```

### 디바이스 옵션

```python
import rm_abstract

# GPU (vLLM)
rm_abstract.init(device="gpu:0")

# Rebellions ATOM NPU
rm_abstract.init(device="rbln:0")

# FuriosaAI RNGD NPU
rm_abstract.init(device="furiosa:0")

# 자동 선택
rm_abstract.init(device="auto")
```

### 컴파일 옵션 (NPU)

```python
import rm_abstract

rm_abstract.init(
    device="rbln:0",
    compile_options={
        "optimization_level": 3,
        "precision": "fp16",
    },
    cache_dir="~/.rm_abstract/cache",
    verbose=True  # 컴파일 진행상황 출력
)
```

## 동작 원리

```
rm_abstract.init(device="npu:0")
        │
        ▼
┌───────────────────────┐
│ DeviceFlowController  │
│ - 백엔드 선택          │
│ - 후킹 시스템 활성화    │
└───────────────────────┘
        │
        ▼
model = AutoModel.from_pretrained("llama")
        │
        ▼
┌───────────────────────┐
│ NPUBackend.prepare    │
│ - 캐시 확인            │
│ - ONNX 변환           │
│ - NPU 컴파일          │
│ - 캐시 저장            │
└───────────────────────┘
        │
        ▼
model.generate(inputs)
        │
        ▼
┌───────────────────────┐
│ NPUBackend.execute    │
│ - NPU 런타임 실행      │
└───────────────────────┘
```

## 프로젝트 구조

```
rm_abstract_layer/
├── src/rm_abstract/
│   ├── __init__.py           # 메인 진입점
│   ├── core/
│   │   ├── backend.py        # Backend ABC
│   │   ├── controller.py     # DeviceFlowController
│   │   └── config.py         # 설정 관리
│   ├── backends/
│   │   ├── gpu/              # vLLM 백엔드
│   │   ├── npu/              # NPU 백엔드
│   │   │   └── plugins/
│   │   │       ├── rebellions.py
│   │   │       └── furiosa.py
│   │   └── cpu/              # CPU 백엔드
│   └── hooks/                # 자동 후킹 시스템
├── tests/
├── examples/
└── docs/
```

## 개발 현황

- [x] Phase 1: 프로젝트 설계 및 계획
- [ ] Phase 2: 핵심 인프라 구축
- [ ] Phase 3: GPU 백엔드 (vLLM)
- [ ] Phase 4: NPU 백엔드 (Rebellions, FuriosaAI)
- [ ] Phase 5: 테스트 및 검증
- [ ] Phase 6: 문서화 및 배포

## 라이선스

MIT License

## 기여

기여를 환영합니다! [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md)를 참조하여 개발 계획을 확인해 주세요.
