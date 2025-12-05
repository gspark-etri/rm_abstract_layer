"""
FastAPI REST API Server - OpenAI-compatible API

Provides:
- /v1/completions - Text completion
- /v1/chat/completions - Chat completion
- /v1/models - List available models
- /v1/devices - Device management
"""

import logging
import time
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .models import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionUsage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    ModelInfo,
    ModelListResponse,
    DeviceInfo,
    BackendInfo,
    SystemStatus,
    SwitchDeviceRequest,
    SwitchDeviceResponse,
    ErrorResponse,
    ErrorDetail,
)

logger = logging.getLogger(__name__)

# Global state
_model_cache: Dict[str, Any] = {}
_tokenizer_cache: Dict[str, Any] = {}
_initialized = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global _initialized
    
    # Startup
    logger.info("Starting RM Abstract API server...")
    
    try:
        import rm_abstract
        rm_abstract.init(device="auto", verbose=True)
        _initialized = True
        logger.info("RM Abstract initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize RM Abstract: {e}")
        _initialized = False
    
    yield
    
    # Shutdown
    logger.info("Shutting down RM Abstract API server...")
    _model_cache.clear()
    _tokenizer_cache.clear()


def create_app() -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="RM Abstract Layer API",
        description="OpenAI-compatible REST API for heterogeneous LLM inference",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# ============================================================
# Error Handlers
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                message=exc.detail,
                type="invalid_request_error",
                code=str(exc.status_code),
            )
        ).model_dump(),
    )


# ============================================================
# Health & Info Endpoints
# ============================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "RM Abstract Layer API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "initialized": _initialized}


# ============================================================
# Model Endpoints
# ============================================================

@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models"""
    # Default models that can be loaded
    default_models = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "distilgpt2",
        "facebook/opt-125m",
        "facebook/opt-350m",
    ]
    
    models = []
    for model_id in default_models:
        models.append(ModelInfo(id=model_id))
    
    # Add loaded models
    for model_id in _model_cache.keys():
        if model_id not in default_models:
            models.append(ModelInfo(id=model_id))
    
    return ModelListResponse(data=models)


@app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get model info"""
    return ModelInfo(id=model_id)


# ============================================================
# Completion Endpoints
# ============================================================

def _get_model_and_tokenizer(model_id: str):
    """Get or load model and tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if model_id not in _model_cache:
        logger.info(f"Loading model: {model_id}")
        _tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(model_id)
        _model_cache[model_id] = AutoModelForCausalLM.from_pretrained(model_id)
    
    return _model_cache[model_id], _tokenizer_cache[model_id]


def _count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text"""
    return len(tokenizer.encode(text))


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Create text completion (OpenAI-compatible)"""
    try:
        model, tokenizer = _get_model_and_tokenizer(request.model)
        
        # Handle single or multiple prompts
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for i, prompt in enumerate(prompts):
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            total_prompt_tokens += inputs.input_ids.shape[1]
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature if request.temperature > 0 else 1.0,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                num_return_sequences=request.n,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            # Decode
            for j in range(request.n):
                output_idx = i * request.n + j
                generated_text = tokenizer.decode(
                    outputs[j][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                
                total_completion_tokens += _count_tokens(generated_text, tokenizer)
                
                choices.append(CompletionChoice(
                    text=generated_text,
                    index=output_idx,
                    finish_reason="stop",
                ))
        
        return CompletionResponse(
            model=request.model,
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )
        
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion (OpenAI-compatible)"""
    try:
        model, tokenizer = _get_model_and_tokenizer(request.model)
        
        # Build prompt from messages
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_tokens = inputs.input_ids.shape[1]
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature if request.temperature > 0 else 1.0,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            num_return_sequences=request.n,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        choices = []
        total_completion_tokens = 0
        
        for i in range(request.n):
            generated_text = tokenizer.decode(
                outputs[i][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()
            
            total_completion_tokens += _count_tokens(generated_text, tokenizer)
            
            choices.append(ChatCompletionChoice(
                index=i,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop",
            ))
        
        return ChatCompletionResponse(
            model=request.model,
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=prompt_tokens + total_completion_tokens,
            ),
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Device Management Endpoints
# ============================================================

@app.get("/v1/devices/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status including devices and backends"""
    try:
        import rm_abstract
        from rm_abstract.system_info import get_system_info
        
        info = get_system_info()
        
        devices = []
        for gpu in info.gpus:
            devices.append(DeviceInfo(
                device_type="gpu",
                device_id=gpu.id,
                name=gpu.name,
                vendor="NVIDIA",
                memory_total_gb=gpu.memory_total_gb,
                memory_free_gb=gpu.memory_free_gb,
            ))
        
        for npu in info.npus:
            devices.append(DeviceInfo(
                device_type="npu",
                device_id=npu.id,
                name=npu.name,
                vendor=npu.vendor,
            ))
        
        if info.cpu:
            devices.append(DeviceInfo(
                device_type="cpu",
                device_id=0,
                name=info.cpu.name,
                memory_total_gb=info.cpu.memory_total_gb,
                memory_free_gb=info.cpu.memory_free_gb,
            ))
        
        backends = []
        for backend in info.backends:
            backends.append(BackendInfo(
                name=backend.name,
                display_name=backend.display_name,
                available=backend.available,
                device_type=backend.device_type,
                version=backend.version,
            ))
        
        current_device = None
        current_backend = None
        
        controller = rm_abstract.get_controller()
        if controller:
            current_device = controller.device_name
            current_backend = type(controller._backend).__name__ if controller._backend else None
        
        return SystemStatus(
            status="ok",
            current_device=current_device,
            current_backend=current_backend,
            available_devices=devices,
            available_backends=backends,
        )
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/devices/switch", response_model=SwitchDeviceResponse)
async def switch_device(request: SwitchDeviceRequest):
    """Switch to a different device"""
    try:
        import rm_abstract
        
        controller = rm_abstract.get_controller()
        previous_device = controller.device_name if controller else "unknown"
        
        rm_abstract.switch_device(request.device)
        
        controller = rm_abstract.get_controller()
        current_device = controller.device_name if controller else request.device
        
        # Clear model cache after device switch
        _model_cache.clear()
        _tokenizer_cache.clear()
        
        return SwitchDeviceResponse(
            success=True,
            previous_device=previous_device,
            current_device=current_device,
            message=f"Switched from {previous_device} to {current_device}",
        )
        
    except Exception as e:
        logger.error(f"Switch device error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    """Run the API server"""
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="RM Abstract Layer API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--device", default="auto", help="Initial device")
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║          RM Abstract Layer - API Server                  ║
╠══════════════════════════════════════════════════════════╣
║  Host: {args.host:<15}  Port: {args.port:<10}              ║
║  Docs: http://{args.host}:{args.port}/docs                        ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "rm_abstract.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

