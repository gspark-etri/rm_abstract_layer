"""로깅 유틸리티"""

import logging
import sys


def setup_logger(
    name: str = "rm_abstract",
    level: int = logging.INFO,
    format_string: str = "[%(name)s] %(levelname)s: %(message)s",
) -> logging.Logger:
    """
    로거 설정

    Args:
        name: 로거 이름
        level: 로깅 레벨
        format_string: 로그 포맷

    Returns:
        설정된 로거
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 핸들러가 없으면 추가
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
