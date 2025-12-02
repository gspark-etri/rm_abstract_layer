"""
설정 관리 모듈
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path
import os


@dataclass
class Config:
    """RM Abstract Layer 설정"""

    # 디바이스 설정
    device: str = "auto"

    # 캐시 설정
    cache_dir: Optional[str] = None

    # NPU 컴파일 옵션
    compile_options: Dict[str, Any] = field(default_factory=dict)

    # 로깅 설정
    verbose: bool = True

    # 추론 엔진 옵션 (vLLM 등)
    engine_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 기본 캐시 디렉토리 설정
        if self.cache_dir is None:
            self.cache_dir = os.path.join(Path.home(), ".rm_abstract", "cache")

        # 캐시 디렉토리 생성
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    @property
    def device_type(self) -> str:
        """디바이스 타입 파싱 (gpu, rbln, furiosa, cpu, auto)"""
        if self.device == "auto":
            return "auto"
        return self.device.split(":")[0].lower()

    @property
    def device_id(self) -> int:
        """디바이스 ID 파싱"""
        if self.device == "auto" or ":" not in self.device:
            return 0
        try:
            return int(self.device.split(":")[1])
        except (ValueError, IndexError):
            return 0

    def get_compile_option(self, key: str, default: Any = None) -> Any:
        """컴파일 옵션 조회"""
        return self.compile_options.get(key, default)

    def get_engine_option(self, key: str, default: Any = None) -> Any:
        """엔진 옵션 조회"""
        return self.engine_options.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "device": self.device,
            "cache_dir": self.cache_dir,
            "compile_options": self.compile_options,
            "verbose": self.verbose,
            "engine_options": self.engine_options,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """딕셔너리에서 설정 생성"""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """환경 변수에서 설정 로드"""
        return cls(
            device=os.environ.get("RM_DEVICE", "auto"),
            cache_dir=os.environ.get("RM_CACHE_DIR"),
            verbose=os.environ.get("RM_VERBOSE", "true").lower() == "true",
        )
