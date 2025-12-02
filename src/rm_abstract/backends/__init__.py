"""Backends module - 디바이스별 백엔드 구현"""

from .registry import register_all_backends

# 모든 백엔드 자동 등록
register_all_backends()
