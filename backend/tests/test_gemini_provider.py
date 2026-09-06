import pytest
from unittest.mock import patch, MagicMock
from app.services.embedding.gemini_provider import GeminiEmbeddingProvider
from app.services.llm.gemini_provider import GeminiLLMProvider
from app.core.config import settings

@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr("app.core.config.settings.GEMINI_EMBEDDING_DIMENSION", 1536)
    
@pytest.fixture
def no_key_settings(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GEMINI_API_KEY", None)

@pytest.mark.asyncio
async def test_gemini_embedding_provider_missing_key(no_key_settings):
    provider = GeminiEmbeddingProvider()
    with pytest.raises(ValueError, match="Gemini API key is missing"):
        await provider.get_embeddings(["test"])

@pytest.mark.asyncio
@patch("app.services.embedding.gemini_provider.genai.Client")
async def test_gemini_embedding_success(mock_client_cls, mock_settings):
    # Setup mock
    mock_client = MagicMock()
    mock_aio = MagicMock()
    mock_models = MagicMock()
    
    # Setup the response
    mock_response = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [0.1] * 1536
    mock_response.embeddings = [mock_emb]
    
    # We must mock async function
    async def mock_embed_content(*args, **kwargs):
        return mock_response
        
    mock_models.embed_content = mock_embed_content
    mock_aio.models = mock_models
    mock_client.aio = mock_aio
    mock_client_cls.return_value = mock_client
    
    provider = GeminiEmbeddingProvider()
    assert provider.dimension == 1536
    
    embs = await provider.get_embeddings(["test text"])
    assert len(embs) == 1
    assert len(embs[0]) == 1536

@pytest.mark.asyncio
@patch("app.services.embedding.gemini_provider.genai.Client")
async def test_gemini_embedding_wrong_dimension(mock_client_cls, mock_settings):
    mock_client = MagicMock()
    mock_aio = MagicMock()
    mock_models = MagicMock()
    
    mock_response = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [0.1] * 768 # Wrong dimension!
    mock_response.embeddings = [mock_emb]
    
    async def mock_embed_content(*args, **kwargs):
        return mock_response
        
    mock_models.embed_content = mock_embed_content
    mock_aio.models = mock_models
    mock_client.aio = mock_aio
    mock_client_cls.return_value = mock_client
    
    provider = GeminiEmbeddingProvider()
    with pytest.raises(RuntimeError, match="Provider returned dimension 768, expected 1536"):
        await provider.get_embeddings(["test text"])

@pytest.mark.asyncio
@patch("app.services.llm.gemini_provider.genai.Client")
async def test_gemini_llm_success(mock_client_cls, mock_settings):
    mock_client = MagicMock()
    mock_aio = MagicMock()
    mock_models = MagicMock()
    
    mock_response = MagicMock()
    mock_response.text = "This is a gemini response"
    
    async def mock_generate_content(*args, **kwargs):
        return mock_response
        
    mock_models.generate_content = mock_generate_content
    mock_aio.models = mock_models
    mock_client.aio = mock_aio
    mock_client_cls.return_value = mock_client
    
    provider = GeminiLLMProvider()
    
    ans = await provider.generate_answer(
        system_prompt="system here",
        messages=[{"role": "user", "content": "hello"}]
    )
    
    assert ans == "This is a gemini response"

