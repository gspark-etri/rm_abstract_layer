"""
API Models - OpenAI-compatible request/response models
"""

from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import time


# ============================================================
# Completion API Models
# ============================================================

class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request"""
    model: str = Field(..., description="Model ID to use")
    prompt: Union[str, List[str]] = Field(..., description="Prompt(s) to generate completions for")
    max_tokens: int = Field(default=100, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=10, description="Number of completions")
    stream: bool = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None


class CompletionChoice(BaseModel):
    """Single completion choice"""
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = "stop"


class CompletionUsage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response"""
    id: str = Field(default_factory=lambda: f"cmpl-{int(time.time()*1000)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


# ============================================================
# Chat Completion API Models
# ============================================================

class ChatMessage(BaseModel):
    """Chat message"""
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(..., description="Model ID to use")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: int = Field(default=100, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=10)
    stream: bool = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """Single chat completion choice"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time()*1000)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


# ============================================================
# Model API Models
# ============================================================

class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "rm-abstract"
    permission: List[Dict[str, Any]] = Field(default_factory=list)
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelListResponse(BaseModel):
    """Model list response"""
    object: str = "list"
    data: List[ModelInfo]


# ============================================================
# Device/Backend API Models
# ============================================================

class DeviceInfo(BaseModel):
    """Device information"""
    device_type: str
    device_id: int
    name: str
    vendor: Optional[str] = None
    memory_total_gb: Optional[float] = None
    memory_free_gb: Optional[float] = None


class BackendInfo(BaseModel):
    """Backend information"""
    name: str
    display_name: str
    available: bool
    device_type: str
    version: Optional[str] = None


class SystemStatus(BaseModel):
    """System status"""
    status: str = "ok"
    current_device: Optional[str] = None
    current_backend: Optional[str] = None
    available_devices: List[DeviceInfo] = Field(default_factory=list)
    available_backends: List[BackendInfo] = Field(default_factory=list)


class SwitchDeviceRequest(BaseModel):
    """Switch device request"""
    device: str = Field(..., description="Device to switch to (e.g., 'gpu:0', 'cpu')")


class SwitchDeviceResponse(BaseModel):
    """Switch device response"""
    success: bool
    previous_device: str
    current_device: str
    message: str


# ============================================================
# Error Models
# ============================================================

class ErrorDetail(BaseModel):
    """Error detail"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: ErrorDetail

