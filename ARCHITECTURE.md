# 시스템 아키텍처

RM Abstract Layer의 아키텍처와 설계 원칙을 설명합니다.

---

## 📐 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (HuggingFace Transformers, User Code, REST API Client)     │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  RM Abstract Layer                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Transformers│  │   Device     │  │   Serving Engine    │ │
│  │    Hook     │  │  Controller  │  │     Factory         │ │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘ │
│         │                │                      │            │
│         └────────────────┼──────────────────────┘            │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Backend Plugin Registry                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   GPU Backend    │  │   CPU Backend    │  │   NPU Backend    │
│     (vLLM)       │  │   (PyTorch)      │  │   (RBLN)         │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                      │
         ▼                     ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   NVIDIA GPU     │  │      CPU         │  │  Rebellions NPU  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 🧩 핵심 컴포넌트

### 1. DeviceFlowController

디바이스 관리 및 백엔드 라우팅을 담당합니다.

```python
# src/rm_abstract/core/controller.py

class DeviceFlowController:
    """디바이스 플로우 컨트롤러"""
    
    def __init__(self, config: Config):
        self.config = config
        self._backend = None
        self._select_backend()
    
    def _select_backend(self):
        """설정에 따라 적절한 백엔드 선택"""
        device = self.config.device
        
        if device.startswith("gpu"):
            self._backend = VLLMBackend(...)
        elif device.startswith("rbln"):
            self._backend = RBLNBackend(...)
        else:
            self._backend = CPUBackend(...)
    
    def switch_device(self, device: str):
        """런타임 디바이스 전환"""
        self.config.device = device
        self._select_backend()
    
    def prepare_model_with_proxy(self, model):
        """모델을 프록시로 래핑"""
        return ModelProxy(model, self._backend)
```

### 2. TransformersHook

HuggingFace Transformers의 `from_pretrained` 메서드를 후킹합니다.

```python
# src/rm_abstract/hooks/transformers_hook.py

def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    """from_pretrained 후킹"""
    # 원본 메서드 호출
    model = _original_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
    
    # 컨트롤러가 있으면 프록시로 래핑
    if _controller is not None:
        model = _controller.prepare_model_with_proxy(model)
    
    return model
```

### 3. Backend Interface

모든 백엔드가 구현해야 하는 인터페이스입니다.

```python
# src/rm_abstract/backends/base.py

class BackendBase(ABC):
    """백엔드 기본 클래스"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """백엔드 사용 가능 여부"""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """백엔드 초기화"""
        pass
    
    @abstractmethod
    def prepare_model(self, model: Any, model_config: Optional[Dict] = None) -> Any:
        """모델 준비 (컴파일, 최적화 등)"""
        pass
    
    @abstractmethod
    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """추론 실행"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """리소스 정리"""
        pass
```

### 4. ModelProxy

백엔드로 추론 요청을 라우팅하는 프록시입니다.

```python
# src/rm_abstract/core/proxy.py

class ModelProxy:
    """모델 프록시 - 백엔드로 요청 라우팅"""
    
    def __init__(self, model, backend):
        self._model = model
        self._backend = backend
    
    def generate(self, *args, **kwargs):
        """generate 메서드 프록시"""
        return self._backend.execute(
            self._model, 
            args[0] if args else kwargs.get('inputs'),
            _proxy_method="generate",
            **kwargs
        )
    
    def __call__(self, *args, **kwargs):
        """forward 메서드 프록시"""
        return self._backend.execute(
            self._model,
            args[0] if args else kwargs.get('inputs'),
            _proxy_method="forward",
            **kwargs
        )
```

---

## 🔌 백엔드 구현

### GPU Backend (vLLM)

```python
# src/rm_abstract/backends/gpu/vllm_backend.py

class VLLMBackend(BackendBase):
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._llm_engine = None
    
    def is_available(self) -> bool:
        try:
            import vllm
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def prepare_model(self, model, model_config=None):
        from vllm import LLM
        
        model_name = getattr(model.config, '_name_or_path', 'gpt2')
        self._llm_engine = LLM(model=model_name)
        return self._llm_engine
    
    def execute(self, model, inputs, **kwargs):
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=kwargs.get('max_new_tokens', 100),
            temperature=kwargs.get('temperature', 1.0),
        )
        
        outputs = self._llm_engine.generate(prompts, sampling_params)
        return self._convert_to_hf_format(outputs)
```

### CPU Backend (PyTorch)

```python
# src/rm_abstract/backends/cpu/cpu_backend.py

class CPUBackend(BackendBase):
    def is_available(self) -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def prepare_model(self, model, model_config=None):
        model.to('cpu')
        model.eval()
        return model
    
    def execute(self, model, inputs, **kwargs):
        import torch
        
        with torch.no_grad():
            if kwargs.get('_proxy_method') == 'generate':
                return model.generate(inputs, **kwargs)
            else:
                return model(inputs)
```

### NPU Backend (RBLN)

```python
# src/rm_abstract/backends/npu/plugins/rebellions.py

class RBLNBackend(NPUBackendBase):
    def __init__(self, device_id: int = 0, mode: str = "auto"):
        self.device_id = device_id
        self.mode = self._detect_mode(mode)
    
    def _detect_mode(self, mode: str) -> str:
        if mode == "auto":
            # vLLM-RBLN 우선
            try:
                import vllm
                return "vllm"
            except ImportError:
                return "optimum"
        return mode
    
    def prepare_model(self, model, model_config=None):
        if self.mode == "vllm":
            return self._prepare_vllm(model, model_config)
        else:
            return self._prepare_optimum(model, model_config)
```

---

## 🚀 서빙 엔진 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                  ServingEngineFactory                        │
└─────────────────────────────────┬───────────────────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   VLLMEngine     │  │  TritonEngine    │  │ TorchServeEngine │
│  - load_model()  │  │  - load_model()  │  │  - load_model()  │
│  - infer()       │  │  - infer()       │  │  - infer()       │
│  - start()       │  │  - start()       │  │  - start()       │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

```python
# src/rm_abstract/serving/base.py

class ServingEngine(ABC):
    """서빙 엔진 기본 클래스"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def infer(self, prompt: str, **kwargs) -> str:
        pass
```

---

## 📊 플러그인 시스템

### 플러그인 등록

```python
# src/rm_abstract/backends/auto_register.py

def auto_register_backends():
    """사용 가능한 백엔드 자동 등록"""
    
    # GPU 백엔드
    try:
        from .gpu.vllm_backend import VLLMBackend
        registry.register(create_backend_plugin(
            backend_class=VLLMBackend,
            name="gpu",
            priority=PluginPriority.HIGH,
        ))
    except ImportError:
        pass
    
    # CPU 백엔드
    try:
        from .cpu.cpu_backend import CPUBackend
        registry.register(create_backend_plugin(
            backend_class=CPUBackend,
            name="cpu",
            priority=PluginPriority.LOW,
        ))
    except ImportError:
        pass
```

### 플러그인 우선순위

```
NPU (HIGHEST) > GPU (HIGH) > PIM (MEDIUM) > CPU (LOW) > Remote (LOWEST)
```

---

## 🔄 데이터 흐름

### 추론 요청 흐름

```
1. User: model.generate(inputs)
         │
         ▼
2. ModelProxy.generate(inputs)
         │
         ▼
3. Backend.execute(model, inputs)
         │
         ▼
4. Hardware (GPU/CPU/NPU)
         │
         ▼
5. Backend._convert_to_hf_format(outputs)
         │
         ▼
6. Return to User
```

### 디바이스 전환 흐름

```
1. User: rm_abstract.switch_device("cpu")
         │
         ▼
2. Controller.switch_device("cpu")
         │
         ▼
3. Old Backend.cleanup()
         │
         ▼
4. New Backend = CPUBackend()
         │
         ▼
5. New Backend.initialize()
         │
         ▼
6. Update Controller._backend
```

---

## 📁 디렉토리 구조

```
src/rm_abstract/
├── __init__.py           # 메인 인터페이스
├── api/                  # REST API
│   ├── server.py         # FastAPI 서버
│   └── models.py         # Pydantic 모델
├── backends/             # 백엔드 구현
│   ├── base.py           # 백엔드 기본 클래스
│   ├── auto_register.py  # 자동 등록
│   ├── cpu/              # CPU 백엔드
│   ├── gpu/              # GPU 백엔드
│   └── npu/              # NPU 백엔드
├── serving/              # 서빙 엔진
│   ├── base.py           # 서빙 엔진 기본
│   ├── vllm_engine.py    # vLLM
│   ├── triton_engine.py  # Triton
│   └── torchserve_engine.py # TorchServe
├── core/                 # 코어 모듈
│   ├── controller.py     # 디바이스 컨트롤러
│   ├── config.py         # 설정
│   └── proxy.py          # 모델 프록시
├── hooks/                # 훅
│   └── transformers_hook.py
├── system_info.py        # 시스템 정보
├── system_validator.py   # 시스템 검증
└── installer.py          # 설치 헬퍼
```

---

## 🎯 설계 원칙

1. **최소 침습성**: 기존 코드 수정 최소화
2. **투명성**: 백엔드 세부사항 추상화
3. **확장성**: 새로운 백엔드 쉽게 추가
4. **유연성**: 런타임 디바이스 전환
5. **호환성**: OpenAI API 호환

---

## 🔗 관련 문서

- [INSTALL.md](INSTALL.md) - 설치 가이드
- [API.md](API.md) - REST API 레퍼런스
- [CONTRIBUTING.md](CONTRIBUTING.md) - 개발 가이드

