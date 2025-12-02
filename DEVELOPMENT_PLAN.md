# 이종 AI 반도체 통합 호환 라이브러리 (RM Abstract Layer)

## 프로젝트 개요

**핵심 목표**: 기존 GPU 추론 스크립트를 **코드 수정 없이** NPU/GPU 어디서든 실행 가능하도록 하는 추상화 레이어

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     기존 사용자 추론 코드 (수정 없음)                      │
│   model = AutoModelForCausalLM.from_pretrained("llama")                 │
│   output = model.generate(input_ids)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                    RM Abstract Layer (1줄 추가)                          │
│   import rm_abstract; rm_abstract.init(device="npu:0")                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Device Flow Controller                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │  │
│  │  │  GPU Flow   │  │  NPU Flow   │  │  CPU Flow   │               │  │
│  │  │ (직접실행)   │  │ (컴파일→실행)│  │ (직접실행)   │               │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐              │
│  │  vLLM/CUDA  │ RBLN SDK    │ Furiosa SDK │  CPU Runtime│              │
│  │  (GPU)      │ (Rebellions)│ (FuriosaAI) │  (PyTorch)  │              │
│  └─────────────┴─────────────┴─────────────┴─────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 지원 환경

### 추론 엔진
| 디바이스 | 추론 엔진 | 비고 |
|----------|-----------|------|
| **GPU** | vLLM | Continuous Batching, Tensor Parallel 지원 |
| **NPU (Rebellions)** | RBLN Runtime / rbln-vllm | ATOM NPU용 추론 |
| **NPU (FuriosaAI)** | Furiosa Runtime | RNGD NPU용 추론 |
| **CPU** | PyTorch / ONNX Runtime | Fallback용 |

### NPU 벤더
| 벤더 | NPU 모델 | SDK | 특징 |
|------|----------|-----|------|
| **Rebellions** | ATOM | RBLN SDK | LLM 특화, vLLM 호환 |
| **FuriosaAI** | RNGD | Furiosa SDK | LLM 특화, 고성능 추론 |

---

## 핵심 설계 원칙

### 1. Zero Code Change (코드 무수정 원칙)
```python
# 기존 추론 코드 (이 코드는 절대 수정하지 않음)
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

```python
# RM Abstract Layer 적용 (1줄만 추가)
import rm_abstract
rm_abstract.init(device="npu:0")  # 이 한 줄만 추가!

# 이하 기존 코드 그대로
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
# ... 나머지 동일
```

### 2. Device-Aware Flow (디바이스별 자동 플로우)
```
Device 설정
    │
    ├─→ GPU 감지 → GPU Flow (직접 실행, PyTorch 그대로 사용)
    │
    ├─→ NPU 감지 → NPU Flow
    │       │
    │       ├─→ 컴파일된 모델 캐시 존재? → 캐시 로드 → 실행
    │       │
    │       └─→ 캐시 없음 → 모델 변환(ONNX) → NPU 컴파일 → 캐시 저장 → 실행
    │
    └─→ CPU 감지 → CPU Flow (PyTorch CPU 실행)
```

### 3. Transparent Compilation (투명한 컴파일)
- NPU는 컴파일이 필요하지만, 사용자는 이를 인지하지 않아도 됨
- 최초 실행 시 자동 컴파일, 이후 캐시 사용
- 컴파일 진행 상황은 로깅으로 표시

---

## Phase 1: 핵심 인프라 구축

### 1.1 프로젝트 초기 설정
- [ ] Python 패키지 구조 설계
- [ ] pyproject.toml 구성
- [ ] 의존성 관리 (requirements.txt)

### 1.2 핵심 추상화 인터페이스
- [ ] `Backend` 추상 베이스 클래스
  ```python
  class Backend(ABC):
      @abstractmethod
      def is_available(self) -> bool: ...

      @abstractmethod
      def prepare_model(self, model) -> Any:
          """GPU: 그대로 반환, NPU: 컴파일 후 반환"""
          ...

      @abstractmethod
      def execute(self, prepared_model, inputs) -> Any: ...

      @abstractmethod
      def get_device_info(self) -> DeviceInfo: ...
  ```

- [ ] `DeviceFlowController` 클래스
  ```python
  class DeviceFlowController:
      def __init__(self, device: str):
          self.backend = self._select_backend(device)

      def _select_backend(self, device: str) -> Backend:
          """device 문자열 파싱하여 적절한 백엔드 반환"""
          ...

      def intercept_model_call(self, model, method_name, *args, **kwargs):
          """모델 메서드 호출을 가로채서 적절한 백엔드로 라우팅"""
          ...
  ```

- [ ] `ModelInterceptor` 클래스 (핵심!)
  ```python
  class ModelInterceptor:
      """PyTorch 모델의 forward/generate 등을 투명하게 가로채기"""

      def wrap(self, model):
          """모델의 핵심 메서드를 백엔드 aware하게 래핑"""
          original_forward = model.forward
          model.forward = self._create_intercepted_forward(original_forward)
          return model
  ```

### 1.3 자동 Hooking 시스템
- [ ] `transformers` 라이브러리 monkey-patching
- [ ] `torch.nn.Module` 레벨 후킹
- [ ] 모델 로드 시점 자동 감지 및 래핑

---

## Phase 2: GPU 백엔드 (vLLM 통합)

### 2.1 GPU Backend 구현 (vLLM 기반)
```python
class GPUBackend(Backend):
    """GPU 백엔드 - vLLM 추론 엔진 활용"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.llm_engine = None

    def prepare_model(self, model_name_or_path: str, **kwargs):
        from vllm import LLM
        # vLLM 엔진 초기화
        self.llm_engine = LLM(
            model=model_name_or_path,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            dtype=kwargs.get("dtype", "auto"),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
        )
        return self.llm_engine

    def execute(self, prompts, sampling_params=None):
        from vllm import SamplingParams
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
        return self.llm_engine.generate(prompts, sampling_params)
```

### 2.2 vLLM 통합 옵션
```python
# vLLM 기반 GPU 추론 예시
import rm_abstract

rm_abstract.init(
    device="gpu:0",
    inference_engine="vllm",  # vLLM 사용
    engine_options={
        "tensor_parallel_size": 2,  # 멀티 GPU
        "dtype": "float16",
        "gpu_memory_utilization": 0.85,
    }
)
```

- [ ] vLLM 엔진 통합
- [ ] CUDA 가용성 체크
- [ ] 멀티 GPU 지원 (Tensor Parallel)
- [ ] 메모리 관리 유틸리티
- [ ] Continuous Batching 활용

---

## Phase 3: NPU 백엔드 (컴파일 플로우 포함)

### 3.1 NPU 공통 인터페이스
```python
class NPUBackend(Backend, ABC):
    """NPU 백엔드 공통 베이스 - 컴파일 플로우 포함"""

    def __init__(self, device_id: int, cache_dir: str):
        self.cache_dir = cache_dir
        self.compiled_models = {}

    def prepare_model(self, model):
        cache_key = self._get_cache_key(model)

        # 1. 캐시 확인
        if self._cache_exists(cache_key):
            return self._load_from_cache(cache_key)

        # 2. ONNX 변환
        onnx_model = self._convert_to_onnx(model)

        # 3. NPU 컴파일 (벤더별 구현)
        compiled = self._compile_for_npu(onnx_model)

        # 4. 캐시 저장
        self._save_to_cache(cache_key, compiled)

        return compiled

    @abstractmethod
    def _compile_for_npu(self, onnx_model) -> Any:
        """벤더별 NPU 컴파일러 호출"""
        ...

    @abstractmethod
    def _execute_on_npu(self, compiled_model, inputs) -> Any:
        """벤더별 NPU 런타임 실행"""
        ...
```

### 3.2 NPU 컴파일 파이프라인
```
PyTorch Model
     │
     ▼
┌─────────────────┐
│  ONNX Export    │  torch.onnx.export()
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  ONNX Optimize  │  onnxoptimizer, onnx-simplifier
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  NPU Compiler   │  벤더별 SDK (compile_model())
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Compiled Model │  .npu, .enf, .blob 등 벤더별 포맷
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Model Cache    │  ~/.rm_abstract/cache/
└─────────────────┘
```

### 3.3 NPU 벤더별 백엔드 구현

#### 지원 NPU 벤더
| 벤더 | SDK | 컴파일러 | 런타임 | 비고 |
|------|-----|---------|--------|------|
| **Rebellions** | RBLN SDK | rbln-compiler | RBLN Runtime | ATOM NPU |
| **FuriosaAI** | Furiosa SDK | furiosa-compiler | furiosa-runtime | RNGD NPU |

#### Rebellions (RBLN) 백엔드
```python
class RBLNBackend(NPUBackend):
    """Rebellions ATOM NPU 백엔드"""

    def _compile_for_npu(self, onnx_model):
        import rebel
        # RBLN 컴파일러로 모델 컴파일
        compiled = rebel.compile_from_onnx(
            onnx_model,
            target="atom",
            optimization_level=3
        )
        return compiled

    def _execute_on_npu(self, compiled_model, inputs):
        import rebel
        runtime = rebel.Runtime()
        return runtime.run(compiled_model, inputs)
```

#### FuriosaAI (RNGD) 백엔드
```python
class FuriosaBackend(NPUBackend):
    """FuriosaAI RNGD NPU 백엔드"""

    def _compile_for_npu(self, onnx_model):
        from furiosa import compiler
        # Furiosa 컴파일러로 모델 컴파일
        compiled = compiler.compile(
            onnx_model,
            target="rngd",  # RNGD NPU 타겟
            batch_size=1
        )
        return compiled

    def _execute_on_npu(self, compiled_model, inputs):
        from furiosa import runtime
        sess = runtime.create_runner(compiled_model)
        return sess.run(inputs)
```

- [ ] Rebellions RBLN SDK 연동
- [ ] FuriosaAI SDK 연동
- [ ] NPU 벤더 플러그인 인터페이스 정의
- [ ] 플러그인 자동 탐지 및 로드
- [ ] 벤더 SDK 없을 시 graceful fallback

### 3.4 컴파일 캐시 시스템
- [ ] 모델 해시 기반 캐시 키 생성
- [ ] 캐시 디렉토리 관리 (`~/.rm_abstract/cache/`)
- [ ] 캐시 무효화 정책 (모델 변경, SDK 버전 변경 등)
- [ ] 캐시 통계 및 관리 CLI

---

## Phase 4: 투명한 통합 API

### 4.1 메인 진입점
```python
# rm_abstract/__init__.py

_global_controller = None

def init(device: str = "auto",
         cache_dir: str = None,
         compile_options: dict = None,
         verbose: bool = True):
    """
    RM Abstract Layer 초기화

    Args:
        device: "auto", "gpu:0", "npu:0", "cpu" 등
        cache_dir: NPU 컴파일 캐시 디렉토리
        compile_options: NPU 컴파일 옵션
        verbose: 컴파일 진행상황 출력 여부
    """
    global _global_controller
    _global_controller = DeviceFlowController(
        device=device,
        cache_dir=cache_dir,
        compile_options=compile_options,
        verbose=verbose
    )

    # Transformers 자동 후킹
    _hook_transformers()

    # PyTorch 모델 자동 후킹
    _hook_pytorch_modules()
```

### 4.2 자동 후킹 시스템
```python
def _hook_transformers():
    """Hugging Face Transformers 라이브러리 자동 패칭"""
    try:
        import transformers

        original_from_pretrained = transformers.PreTrainedModel.from_pretrained

        @wraps(original_from_pretrained)
        def patched_from_pretrained(cls, *args, **kwargs):
            model = original_from_pretrained.__func__(cls, *args, **kwargs)
            return _global_controller.prepare_model(model)

        transformers.PreTrainedModel.from_pretrained = classmethod(patched_from_pretrained)
    except ImportError:
        pass  # transformers 미설치 시 무시

def _hook_pytorch_modules():
    """PyTorch nn.Module 후킹"""
    import torch.nn as nn

    original_call = nn.Module.__call__

    @wraps(original_call)
    def patched_call(self, *args, **kwargs):
        if _global_controller and _global_controller.should_intercept(self):
            return _global_controller.execute(self, *args, **kwargs)
        return original_call(self, *args, **kwargs)

    nn.Module.__call__ = patched_call
```

### 4.3 사용 예시

#### 예시 1: 기본 사용 (가장 간단)
```python
import rm_abstract
rm_abstract.init(device="npu:0")

# 이하 기존 코드 100% 동일
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
result = generator("Hello, I'm a language model")
```

#### 예시 2: 환경 변수로 설정
```bash
export RM_DEVICE="npu:0"
export RM_CACHE_DIR="/data/npu_cache"
python existing_inference_script.py  # 코드 수정 없이!
```

```python
# existing_inference_script.py (수정 없음, 단 rm_abstract import만 추가)
import rm_abstract  # 환경 변수 자동 인식
rm_abstract.init()  # 환경 변수에서 설정 로드

from transformers import AutoModelForCausalLM
# ... 기존 코드 그대로
```

#### 예시 3: 명시적 컴파일 제어
```python
import rm_abstract

# NPU 컴파일 옵션 상세 설정
rm_abstract.init(
    device="npu:0",
    compile_options={
        "optimization_level": 3,
        "precision": "fp16",
        "batch_size": [1, 4, 8],  # 동적 배치
    },
    verbose=True  # 컴파일 진행상황 출력
)

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("llama-7b")
# 최초 실행 시: "Compiling model for NPU... [====>    ] 45%"
# 이후 실행 시: "Loading compiled model from cache..."
```

#### 예시 4: 디바이스 전환
```python
import rm_abstract

# GPU로 시작
rm_abstract.init(device="gpu:0")
# ... 추론 실행 ...

# NPU로 전환
rm_abstract.switch_device("npu:0")
# ... 동일 코드로 NPU에서 추론 ...
```

---

## Phase 5: 테스트 및 검증

### 5.1 단위 테스트
- [ ] 각 백엔드 기본 기능
- [ ] 후킹 시스템 정상 동작
- [ ] 캐시 시스템

### 5.2 통합 테스트
- [ ] 실제 모델 (GPT-2, BERT, LLaMA) 테스트
- [ ] GPU ↔ NPU 결과 일치성 검증
- [ ] 다양한 추론 시나리오

### 5.3 성능 벤치마크
- [ ] 추상화 오버헤드 측정
- [ ] 컴파일 시간 측정
- [ ] 캐시 히트율 분석

---

## Phase 6: 문서화 및 배포

### 6.1 문서화
- [ ] Quick Start 가이드
- [ ] NPU 벤더 플러그인 개발 가이드
- [ ] API 레퍼런스
- [ ] FAQ / 트러블슈팅

### 6.2 배포
- [ ] PyPI 패키지
- [ ] Docker 이미지 (NPU SDK 포함)
- [ ] CI/CD 파이프라인

---

## 프로젝트 구조

```
rm_abstract_layer/
├── src/
│   └── rm_abstract/
│       ├── __init__.py              # 메인 진입점 (init, switch_device)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── backend.py           # Backend ABC
│       │   ├── controller.py        # DeviceFlowController
│       │   ├── interceptor.py       # ModelInterceptor
│       │   └── config.py            # 설정 관리
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── registry.py          # 백엔드 플러그인 레지스트리
│       │   ├── gpu/
│       │   │   ├── __init__.py
│       │   │   └── cuda_backend.py  # GPU Passthrough
│       │   ├── npu/
│       │   │   ├── __init__.py
│       │   │   ├── base.py          # NPU 공통 (컴파일 플로우)
│       │   │   ├── compiler.py      # 컴파일 파이프라인
│       │   │   ├── cache.py         # 컴파일 캐시 관리
│       │   │   └── plugins/         # 벤더별 플러그인
│       │   │       ├── __init__.py
│       │   │       ├── rebellions.py    # Rebellions ATOM
│       │   │       └── furiosa.py       # FuriosaAI RNGD
│       │   └── cpu/
│       │       ├── __init__.py
│       │       └── cpu_backend.py
│       ├── hooks/
│       │   ├── __init__.py
│       │   ├── transformers_hook.py # HF Transformers 후킹
│       │   └── pytorch_hook.py      # PyTorch 모듈 후킹
│       ├── conversion/
│       │   ├── __init__.py
│       │   └── onnx_utils.py        # ONNX 변환 유틸리티
│       └── utils/
│           ├── __init__.py
│           ├── logger.py
│           └── device_utils.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── examples/
│   ├── basic_usage.py
│   ├── npu_compilation.py
│   └── device_switching.py
├── docs/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 핵심 Flow 다이어그램

```
사용자 코드 실행
       │
       ▼
rm_abstract.init(device="npu:0")
       │
       ▼
┌─────────────────────────────┐
│   DeviceFlowController      │
│   - 백엔드 선택 (NPUBackend) │
│   - 후킹 시스템 활성화       │
└─────────────────────────────┘
       │
       ▼
model = AutoModel.from_pretrained("llama")
       │
       ▼
┌─────────────────────────────┐
│   Transformers Hook 발동    │
│   - from_pretrained 가로채기 │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   NPUBackend.prepare_model  │
│   ┌───────────────────────┐ │
│   │ 캐시 확인             │ │
│   │   ├─ Hit → 로드       │ │
│   │   └─ Miss → 컴파일    │ │
│   │       ├─ ONNX 변환    │ │
│   │       ├─ NPU 컴파일   │ │
│   │       └─ 캐시 저장    │ │
│   └───────────────────────┘ │
└─────────────────────────────┘
       │
       ▼
model.generate(inputs)
       │
       ▼
┌─────────────────────────────┐
│   PyTorch Hook 발동         │
│   - __call__ 가로채기       │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   NPUBackend.execute        │
│   - NPU 런타임 실행         │
│   - 결과 반환               │
└─────────────────────────────┘
       │
       ▼
output (사용자에게 반환)
```

---

## 다음 단계

1. **Phase 1 시작**: 프로젝트 구조 생성 및 핵심 인터페이스 구현
2. Backend ABC, DeviceFlowController, ModelInterceptor 구현
3. GPU 백엔드 (Passthrough) 구현으로 기본 동작 검증
4. NPU 컴파일 플로우 및 캐시 시스템 구현

---

*문서 작성일: 2025-12-02*
*버전: 0.2.0-draft*
