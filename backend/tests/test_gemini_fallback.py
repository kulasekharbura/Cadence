import pytest
from unittest.mock import patch, MagicMock
from app.services.llm.gemini_provider import GeminiLLMProvider

class MockAPIError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(message)

@pytest.fixture
def mock_settings_fallback(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.core.config.settings.GEMINI_LLM_MODEL", "gemini-3.5-flash")
    monkeypatch.setattr("app.core.config.settings.GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.7-flash")

@pytest.mark.asyncio
@patch("app.services.llm.gemini_provider.genai.Client")
async def test_gemini_fallback_primary_succeeds(mock_client_cls, mock_settings_fallback):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Success on first try"
    
    async def mock_generate_content(*args, **kwargs):
        return mock_response
        
    mock_client.aio.models.generate_content = mock_generate_content
    mock_client_cls.return_value = mock_client
    
    provider = GeminiLLMProvider()
    ans = await provider.generate_answer("sys", [{"role": "user", "content": "hello"}])
    assert ans == "Success on first try"

@pytest.mark.asyncio
@patch("app.services.llm.gemini_provider.genai.Client")
async def test_gemini_fallback_primary_429(mock_client_cls, mock_settings_fallback):
    mock_client = MagicMock()
    call_count = {"count": 0}
    
    async def mock_generate_content(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise MockAPIError(429, "Rate limit")
        mock_response = MagicMock()
        mock_response.text = "Success on fallback"
        return mock_response
        
    mock_client.aio.models.generate_content = mock_generate_content
    mock_client_cls.return_value = mock_client
    
    provider = GeminiLLMProvider()
    ans = await provider.generate_answer("sys", [{"role": "user", "content": "hello"}])
    assert ans == "Success on fallback"
    assert call_count["count"] == 2

@pytest.mark.asyncio
@patch("app.services.llm.gemini_provider.genai.Client")
async def test_gemini_fallback_primary_503(mock_client_cls, mock_settings_fallback):
    mock_client = MagicMock()
    call_count = {"count": 0}
    
    async def mock_generate_content(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise MockAPIError(503, "Unavailable")
        mock_response = MagicMock()
        mock_response.text = "Success on fallback"
        return mock_response
        
    mock_client.aio.models.generate_content = mock_generate_content
    mock_client_cls.return_value = mock_client
    
    provider = GeminiLLMProvider()
    ans = await provider.generate_answer("sys", [{"role": "user", "content": "hello"}])
    assert ans == "Success on fallback"
    assert call_count["count"] == 2

@pytest.mark.asyncio
@patch("app.services.llm.gemini_provider.genai.Client")
async def test_gemini_fallback_primary_400(mock_client_cls, mock_settings_fallback):
    mock_client = MagicMock()
    call_count = {"count": 0}
    
    async def mock_generate_content(*args, **kwargs):
        call_count["count"] += 1
        raise MockAPIError(400, "Bad Request")
        
    mock_client.aio.models.generate_content = mock_generate_content
    mock_client_cls.return_value = mock_client
    
    provider = GeminiLLMProvider()
    with pytest.raises(RuntimeError, match="Gemini LLM API error"):
        await provider.generate_answer("sys", [{"role": "user", "content": "hello"}])
    assert call_count["count"] == 1

@pytest.mark.asyncio
@patch("app.services.llm.gemini_provider.genai.Client")
async def test_gemini_fallback_all_fail(mock_client_cls, mock_settings_fallback):
    mock_client = MagicMock()
    call_count = {"count": 0}
    
    async def mock_generate_content(*args, **kwargs):
        call_count["count"] += 1
        raise MockAPIError(503, "Unavailable")
        
    mock_client.aio.models.generate_content = mock_generate_content
    mock_client_cls.return_value = mock_client
    
    provider = GeminiLLMProvider()
    with pytest.raises(RuntimeError, match="Gemini LLM API error \\(all models failed\\)"):
        await provider.generate_answer("sys", [{"role": "user", "content": "hello"}])
    assert call_count["count"] == 3

@pytest.mark.asyncio
@patch("app.services.llm.gemini_provider.genai.Client")
async def test_gemini_fallback_401_403(mock_client_cls, mock_settings_fallback):
    for code in [401, 403]:
        mock_client = MagicMock()
        call_count = {"count": 0}
        
        async def mock_generate_content(*args, **kwargs):
            call_count["count"] += 1
            raise MockAPIError(code, "Auth error")
            
        mock_client.aio.models.generate_content = mock_generate_content
        mock_client_cls.return_value = mock_client
        
        provider = GeminiLLMProvider()
        with pytest.raises(RuntimeError):
            await provider.generate_answer("sys", [{"role": "user", "content": "hello"}])
        assert call_count["count"] == 1

@pytest.mark.asyncio
@patch("app.services.llm.gemini_provider.genai.Client")
async def test_gemini_fallback_404(mock_client_cls, mock_settings_fallback):
    mock_client = MagicMock()
    call_count = {"count": 0}
    
    async def mock_generate_content(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise MockAPIError(404, "Not Found")
        mock_response = MagicMock()
        mock_response.text = "Success after 404"
        return mock_response
        
    mock_client.aio.models.generate_content = mock_generate_content
    mock_client_cls.return_value = mock_client
    
    provider = GeminiLLMProvider()
    ans = await provider.generate_answer("sys", [{"role": "user", "content": "hello"}])
    assert ans == "Success after 404"
    assert call_count["count"] == 2
