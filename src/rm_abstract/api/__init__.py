"""
RM Abstract Layer - REST API

OpenAI-compatible REST API server for LLM inference.

Usage:
    # Start server
    python -m rm_abstract.api
    
    # Or with uvicorn
    uvicorn rm_abstract.api.server:app --host 0.0.0.0 --port 8000
"""

from .server import app, create_app
from .models import (
    CompletionRequest,
    CompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
)

__all__ = [
    "app",
    "create_app",
    "CompletionRequest",
    "CompletionResponse",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ModelInfo",
]

