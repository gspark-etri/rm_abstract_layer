"""
REST API tests
"""

import pytest
from fastapi.testclient import TestClient


class TestAPIHealth:
    """Test API health endpoints"""
    
    def test_root(self, api_client):
        """Test root endpoint"""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["status"] == "running"
    
    def test_health(self, api_client):
        """Test health endpoint"""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestModelsAPI:
    """Test models API"""
    
    def test_list_models(self, api_client):
        """Test listing models"""
        response = api_client.get("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0
    
    def test_get_model(self, api_client):
        """Test getting model info"""
        response = api_client.get("/v1/models/gpt2")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "gpt2"
        assert data["object"] == "model"


class TestCompletionAPI:
    """Test completion API"""
    
    def test_completion_basic(self, api_client):
        """Test basic completion"""
        response = api_client.post(
            "/v1/completions",
            json={
                "model": "gpt2",
                "prompt": "Hello",
                "max_tokens": 10,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]
        assert "usage" in data
    
    def test_completion_with_params(self, api_client):
        """Test completion with parameters"""
        response = api_client.post(
            "/v1/completions",
            json={
                "model": "gpt2",
                "prompt": "The future of AI",
                "max_tokens": 20,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["choices"]) > 0
    
    def test_completion_multiple_prompts(self, api_client):
        """Test completion with multiple prompts"""
        response = api_client.post(
            "/v1/completions",
            json={
                "model": "gpt2",
                "prompt": ["Hello", "World"],
                "max_tokens": 10,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["choices"]) >= 2


class TestChatCompletionAPI:
    """Test chat completion API"""
    
    def test_chat_completion_basic(self, api_client):
        """Test basic chat completion"""
        response = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt2",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "max_tokens": 20,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert data["choices"][0]["message"]["role"] == "assistant"
    
    def test_chat_completion_with_system(self, api_client):
        """Test chat completion with system message"""
        response = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt2",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ],
                "max_tokens": 20,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["choices"]) > 0


class TestDeviceAPI:
    """Test device management API"""
    
    def test_get_status(self, api_client):
        """Test getting system status"""
        response = api_client.get("/v1/devices/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ok"
        assert "available_devices" in data
        assert "available_backends" in data
    
    def test_switch_device(self, api_client):
        """Test switching device"""
        import rm_abstract
        
        # Initialize first
        rm_abstract.init(device="auto", verbose=False)
        
        response = api_client.post(
            "/v1/devices/switch",
            json={"device": "cpu"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "cpu" in data["current_device"].lower()

