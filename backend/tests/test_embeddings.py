import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.embedding_service import EmbeddingService
from app.schemas.chunk import DocumentChunk
from app.services.embedding.provider import EmbeddingProvider

from app.core.config import settings

class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension=1536, fail=False, return_wrong_size=False):
        self._dimension = dimension
        self.fail = fail
        self.return_wrong_size = return_wrong_size

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("Mock provider failure")
            
        if not texts:
            return []
            
        dim = self._dimension - 1 if self.return_wrong_size else self._dimension
        
        # Return a dummy vector of the specified dimension for each text
        return [[0.1] * dim for _ in texts]
        
    @property
    def dimension(self) -> int:
        return self._dimension

@pytest.mark.asyncio
async def test_embedding_service_success():
    provider = MockEmbeddingProvider(dimension=1536)
    service = EmbeddingService(provider=provider)
    
    chunks = [
        DocumentChunk(document_id="1", page_number=1, chunk_index=0, content="Test chunk 1"),
        DocumentChunk(document_id="1", page_number=1, chunk_index=1, content="Test chunk 2")
    ]
    
    results = await service.generate_embeddings_for_chunks(chunks)
    
    assert len(results) == 2
    assert len(results[0]["embedding"]) == 1536
    assert results[0]["chunk"].content == "Test chunk 1"

@pytest.mark.asyncio
async def test_embedding_service_empty():
    provider = MockEmbeddingProvider()
    service = EmbeddingService(provider=provider)
    
    results = await service.generate_embeddings_for_chunks([])
    assert len(results) == 0

@pytest.mark.asyncio
async def test_embedding_service_failure():
    provider = MockEmbeddingProvider(fail=True)
    service = EmbeddingService(provider=provider)
    
    chunks = [DocumentChunk(document_id="1", page_number=1, chunk_index=0, content="Fail me")]
    
    with pytest.raises(RuntimeError, match="Mock provider failure"):
        await service.generate_embeddings_for_chunks(chunks)

@pytest.mark.asyncio
async def test_embedding_service_mismatch_size():
    # Provider returns wrong number of embeddings compared to input texts
    provider = MockEmbeddingProvider()
    
    # Mock it to return fewer embeddings
    original_get = provider.get_embeddings
    
    async def bad_get(texts):
        return [[0.1]*1536] # Only return 1 embedding regardless of input
        
    provider.get_embeddings = bad_get
    
    service = EmbeddingService(provider=provider)
    chunks = [
        DocumentChunk(document_id="1", page_number=1, chunk_index=0, content="Chunk 1"),
        DocumentChunk(document_id="1", page_number=1, chunk_index=1, content="Chunk 2")
    ]
    
    with pytest.raises(RuntimeError, match="Embedding provider returned 1 embeddings but 2 chunks were provided"):
        await service.generate_embeddings_for_chunks(chunks)




@pytest.mark.asyncio
async def test_embedding_service_batching_100():
    provider = MockEmbeddingProvider(dimension=1536)
    
    # Spy on get_embeddings
    original_get = provider.get_embeddings
    call_counts = []
    
    async def spy_get(texts):
        call_counts.append(len(texts))
        return await original_get(texts)
        
    provider.get_embeddings = spy_get
    
    service = EmbeddingService(provider=provider)
    chunks = [DocumentChunk(document_id="1", page_number=1, chunk_index=i, content=f"Chunk {i}") for i in range(100)]
    
    results = await service.generate_embeddings_for_chunks(chunks)
    assert len(results) == 100
    assert call_counts == [100]
    
@pytest.mark.asyncio
async def test_embedding_service_batching_101():
    provider = MockEmbeddingProvider(dimension=1536)
    
    original_get = provider.get_embeddings
    call_counts = []
    
    async def spy_get(texts):
        call_counts.append(len(texts))
        return await original_get(texts)
        
    provider.get_embeddings = spy_get
    service = EmbeddingService(provider=provider)
    
    chunks = [DocumentChunk(document_id="1", page_number=1, chunk_index=i, content=f"Chunk {i}") for i in range(101)]
    results = await service.generate_embeddings_for_chunks(chunks)
    assert len(results) == 101
    assert call_counts == [100, 1]
    
@pytest.mark.asyncio
async def test_embedding_service_batching_200():
    provider = MockEmbeddingProvider(dimension=1536)
    original_get = provider.get_embeddings
    call_counts = []
    
    async def spy_get(texts):
        call_counts.append(len(texts))
        return await original_get(texts)
        
    provider.get_embeddings = spy_get
    service = EmbeddingService(provider=provider)
    
    chunks = [DocumentChunk(document_id="1", page_number=1, chunk_index=i, content=f"Chunk {i}") for i in range(200)]
    results = await service.generate_embeddings_for_chunks(chunks)
    assert len(results) == 200
    assert call_counts == [100, 100]

@pytest.mark.asyncio
async def test_embedding_service_batching_201():
    provider = MockEmbeddingProvider(dimension=1536)
    original_get = provider.get_embeddings
    call_counts = []
    
    async def spy_get(texts):
        call_counts.append(len(texts))
        return await original_get(texts)
        
    provider.get_embeddings = spy_get
    service = EmbeddingService(provider=provider)
    
    chunks = [DocumentChunk(document_id="1", page_number=1, chunk_index=i, content=f"Chunk {i}") for i in range(201)]
    results = await service.generate_embeddings_for_chunks(chunks)
    assert len(results) == 201
    assert call_counts == [100, 100, 1]

@pytest.mark.asyncio
async def test_embedding_service_order_preservation():
    provider = MockEmbeddingProvider(dimension=1536)
    original_get = provider.get_embeddings
    
    async def spy_get(texts):
        # Return unique embeddings based on the text to verify order
        return [[float(text.split()[-1])] * 1536 for text in texts]
        
    provider.get_embeddings = spy_get
    service = EmbeddingService(provider=provider)
    
    chunks = [DocumentChunk(document_id="1", page_number=1, chunk_index=i, content=f"Chunk {i}") for i in range(150)]
    results = await service.generate_embeddings_for_chunks(chunks)
    
    assert len(results) == 150
    for i in range(150):
        assert results[i]["embedding"][0] == float(i)
        assert results[i]["chunk"].content == f"Chunk {i}"
        
@pytest.mark.asyncio
async def test_embedding_service_batch_failure_propagation():
    provider = MockEmbeddingProvider(dimension=1536)
    original_get = provider.get_embeddings
    call_count = 0
    
    async def fail_second_batch(texts):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("API Rate Limit")
        return await original_get(texts)
        
    provider.get_embeddings = fail_second_batch
    service = EmbeddingService(provider=provider)
    
    chunks = [DocumentChunk(document_id="1", page_number=1, chunk_index=i, content=f"Chunk {i}") for i in range(150)]
    
    with pytest.raises(RuntimeError, match="API Rate Limit"):
        await service.generate_embeddings_for_chunks(chunks)
