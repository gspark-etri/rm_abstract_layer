# 기여 가이드

RM Abstract Layer에 기여하는 방법을 설명합니다.

---

## 🚀 개발 환경 설정

### 1. 저장소 클론

```bash
git clone https://github.com/your-org/rm_abstract_layer.git
cd rm_abstract_layer
```

### 2. 가상환경 생성

```bash
# uv 사용 (권장)
uv venv .venv
source .venv/bin/activate

# 또는 venv 사용
python -m venv .venv
source .venv/bin/activate
```

### 3. 개발 의존성 설치

```bash
# 전체 의존성 설치
pip install -e ".[dev,all]"

# 또는 uv 사용
uv pip install -e ".[dev,all]"
```

### 4. 설치 확인

```bash
python -m rm_abstract.system_validator --quick
pytest tests/test_core.py -v
```

---

## 📁 프로젝트 구조

```
rm_abstract_layer/
├── src/rm_abstract/      # 소스 코드
│   ├── api/              # REST API
│   ├── backends/         # 백엔드 구현
│   ├── serving/          # 서빙 엔진
│   ├── core/             # 코어 모듈
│   └── hooks/            # 훅
├── tests/                # 테스트
├── examples/             # 예제
├── requirements/         # 의존성 파일
├── scripts/              # 스크립트
└── docker/               # Docker 설정
```

---

## 🧪 테스트

### 테스트 실행

```bash
# 전체 테스트
pytest tests/ -v

# 특정 파일
pytest tests/test_core.py -v
pytest tests/test_api.py -v

# 커버리지
pytest tests/ --cov=src/rm_abstract --cov-report=html
```

### 테스트 작성

```python
# tests/test_example.py

import pytest

class TestExample:
    def test_something(self):
        """테스트 설명"""
        result = some_function()
        assert result == expected
    
    @pytest.fixture
    def setup_data(self):
        """테스트 픽스처"""
        return {"key": "value"}
    
    def test_with_fixture(self, setup_data):
        """픽스처 사용 테스트"""
        assert setup_data["key"] == "value"
```

---

## 🔌 새로운 백엔드 추가

### 1. 백엔드 클래스 생성

```python
# src/rm_abstract/backends/new_backend/new_backend.py

from ..base import BackendBase

class NewBackend(BackendBase):
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
    
    @property
    def name(self) -> str:
        return "new_backend"
    
    def is_available(self) -> bool:
        try:
            import new_sdk
            return True
        except ImportError:
            return False
    
    def initialize(self) -> None:
        # 초기화 로직
        pass
    
    def prepare_model(self, model, model_config=None):
        # 모델 준비 로직
        return model
    
    def execute(self, model, inputs, **kwargs):
        # 추론 로직
        return outputs
    
    def cleanup(self) -> None:
        # 정리 로직
        pass
```

### 2. 플러그인 등록

```python
# src/rm_abstract/backends/auto_register.py

def auto_register_backends():
    # ... 기존 코드 ...
    
    # 새 백엔드 등록
    try:
        from .new_backend.new_backend import NewBackend
        registry.register(create_backend_plugin(
            backend_class=NewBackend,
            name="new",
            display_name="New Backend",
            priority=PluginPriority.MEDIUM,
            device_types=["new_device"],
        ))
    except ImportError:
        pass
```

### 3. 테스트 작성

```python
# tests/test_new_backend.py

class TestNewBackend:
    def test_is_available(self):
        from rm_abstract.backends.new_backend import NewBackend
        backend = NewBackend()
        # 테스트
```

---

## 📝 코드 스타일

### 포맷팅

```bash
# Black 포맷팅
black src/ tests/

# isort 임포트 정렬
isort src/ tests/
```

### 린팅

```bash
# flake8
flake8 src/ tests/

# mypy 타입 체크
mypy src/
```

### 설정 파일

```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py39']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.9"
strict = true
```

---

## 📋 Pull Request 가이드

### 브랜치 네이밍

```
feature/기능명
fix/버그명
docs/문서명
refactor/리팩토링명
```

### 커밋 메시지

```
feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서 수정
refactor: 리팩토링
test: 테스트 추가/수정
chore: 기타 작업
```

### PR 체크리스트

- [ ] 테스트 통과
- [ ] 문서 업데이트
- [ ] 코드 리뷰 요청
- [ ] 린트 통과

---

## 🔧 로컬 API 서버 개발

```bash
# 개발 모드 (자동 리로드)
python -m rm_abstract.api --reload

# 또는
uvicorn rm_abstract.api.server:app --reload --port 8000
```

---

## 📚 문서 작성

### 문서 파일

| 파일 | 역할 |
|------|------|
| README.md | 프로젝트 개요 |
| INSTALL.md | 설치 가이드 |
| QUICKSTART.md | 빠른 시작 |
| ARCHITECTURE.md | 아키텍처 |
| API.md | REST API |
| CONTRIBUTING.md | 기여 가이드 |

### docstring 형식

```python
def function_name(param1: str, param2: int = 0) -> bool:
    """
    함수 설명
    
    Args:
        param1: 파라미터1 설명
        param2: 파라미터2 설명 (기본값: 0)
    
    Returns:
        반환값 설명
    
    Raises:
        ValueError: 에러 조건 설명
    
    Example:
        >>> function_name("test", 1)
        True
    """
    pass
```

---

## ❓ 질문 및 지원

- **이슈**: GitHub Issues
- **토론**: GitHub Discussions
- **이메일**: maintainer@example.com

---

## 📄 라이선스

이 프로젝트에 기여하면 MIT 라이선스에 동의하는 것으로 간주합니다.

