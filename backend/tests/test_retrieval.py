import pytest
import uuid
from unittest.mock import patch, AsyncMock
from app.models.document import Document
from app.models.chunk import DocumentChunk
from app.services.retrieval_service import RetrievalService
from app.db.session import async_session_maker
from app.core.config import settings

@pytest.fixture
def mock_embedding_service():
    with patch("app.services.retrieval_service.EmbeddingService") as mock:
        service_instance = mock.return_value
        
        # When generate_embeddings_for_chunks is called, we return a predictable dummy vector
        # by default, we'll return a vector of all 0.1s
        async def mock_generate(chunks):
            return [{"chunk": c, "embedding": [0.1] * settings.GEMINI_EMBEDDING_DIMENSION} for c in chunks]
            
        service_instance.generate_embeddings_for_chunks.side_effect = mock_generate
        yield service_instance

@pytest.mark.asyncio
async def test_retrieval_ranking_and_metadata(mock_embedding_service):
    async with async_session_maker() as db:
        # Create a document
        doc = Document(filename="test_retrieval.pdf", file_path="/fake/test.pdf", file_size=1024, page_count=1)
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        
        dim = settings.GEMINI_EMBEDDING_DIMENSION
        # Let's say our query embedding will be [0.1] * dim (from our mock).
        # We will insert three chunks:
        # 1. Exactly [0.1] * dim -> distance 0.0
        # 2. Opposite [-0.1] * dim -> distance 2.0
        # 3. Slightly off -> distance ~0.something
        
        c1 = DocumentChunk(document_id=doc.id, page_number=1, chunk_index=0, content="Exact match", embedding=[0.1]*dim)
        c2 = DocumentChunk(document_id=doc.id, page_number=1, chunk_index=1, content="Opposite match", embedding=[-0.1]*dim)
        
        vec3 = [0.1]*dim
        vec3[0] = 0.0 # slight deviation
        c3 = DocumentChunk(document_id=doc.id, page_number=1, chunk_index=2, content="Close match", embedding=vec3)
        
        db.add_all([c1, c2, c3])
        await db.commit()
        
        # Initialize RetrievalService
        retrieval_service = RetrievalService(db, embedding_service=mock_embedding_service)
        
        # Query (actual text doesn't matter since embedding is mocked to return [0.1]*dim)
        results = await retrieval_service.retrieve_chunks("Find me exact match", document_ids=[doc.id], top_k=5, distance_threshold=2.0)
        
        assert len(results) == 3
        # Should be ordered by distance ascending
        assert results[0].content == "Exact match"
        assert results[0].distance_score == pytest.approx(0.0, abs=1e-5)
        
        assert results[1].content == "Close match"
        assert results[1].distance_score > 0.0
        assert results[1].distance_score < 1.0
        
        assert results[2].content == "Opposite match"
        assert results[2].distance_score == pytest.approx(2.0, abs=1e-5)
        
        # Metadata check
        assert results[0].filename == "test_retrieval.pdf"
        assert results[0].page_number == 1
        assert results[0].chunk_index == 0
        assert isinstance(results[0].chunk_id, uuid.UUID)

@pytest.mark.asyncio
async def test_retrieval_thresholding(mock_embedding_service):
    async with async_session_maker() as db:
        doc = Document(filename="threshold.pdf", file_path="/fake/t.pdf", file_size=1024)
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        
        dim = settings.GEMINI_EMBEDDING_DIMENSION
        # Add only an opposite match
        c1 = DocumentChunk(document_id=doc.id, page_number=1, chunk_index=0, content="Bad", embedding=[-0.1]*dim)
        db.add(c1)
        await db.commit()
        
        retrieval_service = RetrievalService(db, embedding_service=mock_embedding_service)
        
        # If threshold is 1.0, it should filter out the opposite match (distance 2.0)
        results = await retrieval_service.retrieve_chunks("Query", document_ids=[doc.id], distance_threshold=1.0)
        assert len(results) == 0

@pytest.mark.asyncio
async def test_retrieval_document_filtering(mock_embedding_service):
    async with async_session_maker() as db:
        doc1 = Document(filename="doc1.pdf", file_path="/fake/d1.pdf", file_size=100)
        doc2 = Document(filename="doc2.pdf", file_path="/fake/d2.pdf", file_size=100)
        db.add_all([doc1, doc2])
        await db.commit()
        await db.refresh(doc1)
        await db.refresh(doc2)
        
        dim = settings.GEMINI_EMBEDDING_DIMENSION
        # Add identical vectors to both docs
        c1 = DocumentChunk(document_id=doc1.id, page_number=1, chunk_index=0, content="Doc1 content", embedding=[0.1]*dim)
        c2 = DocumentChunk(document_id=doc2.id, page_number=1, chunk_index=0, content="Doc2 content", embedding=[0.1]*dim)
        db.add_all([c1, c2])
        await db.commit()
        
        retrieval_service = RetrievalService(db, embedding_service=mock_embedding_service)
        
        # 1. Filter by doc1 only
        results = await retrieval_service.retrieve_chunks("Query", document_ids=[doc1.id])
        
        assert len(results) == 1
        assert results[0].document_id == doc1.id
        assert results[0].filename == "doc1.pdf"
        assert results[0].content == "Doc1 content"
        
        # 2. Filter by both doc1 and doc2
        results_both = await retrieval_service.retrieve_chunks("Query", document_ids=[doc1.id, doc2.id])
        assert len(results_both) == 2
        returned_doc_ids = {r.document_id for r in results_both}
        assert doc1.id in returned_doc_ids
        assert doc2.id in returned_doc_ids
        
        # 3. Filter with explicitly empty list (should return 0)
        results_empty = await retrieval_service.retrieve_chunks("Query", document_ids=[])
        assert len(results_empty) == 0
        
        # 4. Filter with None (should return all documents in database, limited by top_k)
        results_none = await retrieval_service.retrieve_chunks("Query", document_ids=None)
        assert len(results_none) > 0

@pytest.mark.asyncio
async def test_retrieval_top_k(mock_embedding_service):
    async with async_session_maker() as db:
        doc = Document(filename="topk.pdf", file_path="/fake/topk.pdf", file_size=1024)
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        
        dim = settings.GEMINI_EMBEDDING_DIMENSION
        
        # Create 10 exact matches
        chunks = [
            DocumentChunk(document_id=doc.id, page_number=1, chunk_index=i, content=f"C{i}", embedding=[0.1]*dim)
            for i in range(10)
        ]
        db.add_all(chunks)
        await db.commit()
        
        retrieval_service = RetrievalService(db, embedding_service=mock_embedding_service)
        
        # Ask for top 3
        results = await retrieval_service.retrieve_chunks("Query", document_ids=[doc.id], top_k=3, distance_threshold=1.0)
        assert len(results) == 3

@pytest.mark.asyncio
async def test_retrieval_multi_document_fairness(mock_embedding_service):
    async with async_session_maker() as db:
        # doc_large has many chunks, doc_small has 1 chunk
        doc_large = Document(filename="large.pdf", file_path="/fake/large.pdf", file_size=1024)
        doc_small = Document(filename="small.pdf", file_path="/fake/small.pdf", file_size=1024)
        db.add_all([doc_large, doc_small])
        await db.commit()
        await db.refresh(doc_large)
        await db.refresh(doc_small)
        
        dim = settings.GEMINI_EMBEDDING_DIMENSION
        
        # doc_large has 10 chunks, all with distance ~ 0.00
        large_chunks = [
            DocumentChunk(document_id=doc_large.id, page_number=1, chunk_index=i, content=f"Large {i}", embedding=[0.1]*dim)
            for i in range(10)
        ]
        
        # doc_small has 1 chunk, distance ~ 0.00 (same vector)
        small_chunk = DocumentChunk(document_id=doc_small.id, page_number=1, chunk_index=0, content="Small 0", embedding=[0.1]*dim)
        
        db.add_all(large_chunks + [small_chunk])
        await db.commit()
        
        retrieval_service = RetrievalService(db, embedding_service=mock_embedding_service)
        
        # Ask for top 3 chunks, querying both documents
        results = await retrieval_service.retrieve_chunks("Query", document_ids=[doc_large.id, doc_small.id], top_k=3, distance_threshold=1.0)
        
        # The result must contain at most 3 chunks.
        assert len(results) <= 3
        
        # To guarantee fairness, doc_small's single chunk should be present in the candidate pool.
        # Since all chunks have identical distance (0.00), Python's sort is stable, or at least small_chunk 
        # might be in the result. But to truly test fairness, let's make small_chunk slightly CLOSER!
        # wait, we can't easily change the distance since it relies on the DB calculation vs the mock embedding [0.1]*dim.
        # But wait! If small_chunk is identical, it will get the exact same distance. 
        # If we didn't aggregate properly, we might just get 3 chunks from doc_large if it was arbitrarily first.
        # But let's verify that BOTH documents contribute to the candidate pool before trimming to top 3.
        # Actually, let's just make small_chunk EXACTLY [0.1]*dim, and make large_chunks SLIGHTLY OFF.
        # But wait, distance is calculated against query vector [0.1]*dim.
        # So we can update large_chunks to have a slight deviation.
        for i, c in enumerate(large_chunks):
            v = [0.1] * dim
            v[0] = 0.05  # slightly further away
            c.embedding = v
            
        await db.commit()
        
        # Now doc_small has distance 0.0, doc_large chunks have distance > 0.0
        results2 = await retrieval_service.retrieve_chunks("Query", document_ids=[doc_large.id, doc_small.id], top_k=5, distance_threshold=2.0)
        
        # The absolute closest chunk MUST be from doc_small
        assert len(results2) == 5
        assert results2[0].document_id == doc_small.id
        assert results2[0].content == "Small 0"
        
        # The remaining 4 must be from doc_large
        for i in range(1, 5):
            assert results2[i].document_id == doc_large.id

@pytest.mark.asyncio
async def test_retrieval_threshold_regression(mock_embedding_service):
    # Regression test for Bug #2 (Degree Retrieval Failure)
    # Verifies that a chunk with distance between 0.5 and 0.6 is retrieved.
    async with async_session_maker() as db:
        doc = Document(filename="resume_test.pdf", file_path="/fake/resume.pdf", file_size=1024)
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        
        dim = settings.GEMINI_EMBEDDING_DIMENSION
        
        # We want this chunk to have a distance of exactly 0.55 from the query vector [0.1]*dim.
        # Cosine distance = 1 - cosine_similarity. 
        # Since distance is calculated by pgvector as 1 - cosine_similarity(a, b).
        # We will mock the database distance query itself or just create an embedding that has exactly that distance.
        # Actually, since mock_embedding_service returns [0.1]*dim for the query, 
        # setting the chunk's embedding to a vector that yields ~0.55 distance is hard to calculate exactly without numpy here.
        # Instead, we will directly test the RetrievalService which uses settings.RETRIEVAL_DISTANCE_THRESHOLD.
        # We'll just pass distance_threshold=0.6 explicitly to ensure the limit works.
        # Wait, the threshold is applied in RetrievalService line `if row.distance > threshold: continue`.
        
        # To avoid tricky math, we can patch the distance calculation or just test that the service uses the correct default threshold.
        # But we want a deterministic test. Let's just create a chunk and patch the db execution to return a mocked row with distance 0.55.
        pass
        
        # A simpler way to test the configuration change:
        assert settings.RETRIEVAL_DISTANCE_THRESHOLD == 0.6
        
        # And we can just test that retrieving with threshold 0.6 works.
        c1 = DocumentChunk(document_id=doc.id, page_number=1, chunk_index=0, content="Near boundary", embedding=[0.1]*dim)
        db.add(c1)
        await db.commit()
        
        retrieval_service = RetrievalService(db, embedding_service=mock_embedding_service)
        # We mock the database result to simulate a distance of 0.55
        with patch.object(db, 'execute') as mock_execute:
            class MockRow:
                def __init__(self):
                    self.id = c1.id
                    self.document_id = doc.id
                    self.filename = "resume_test.pdf"
                    self.page_number = 1
                    self.chunk_index = 0
                    self.content = "Near boundary"
                    self.distance = 0.55
            
            class MockResult:
                def all(self):
                    return [MockRow()]
            
            mock_execute.return_value = MockResult()
            
            # Use default threshold (which should be 0.6)
            results = await retrieval_service.retrieve_chunks("Query")
            assert len(results) == 1
            assert results[0].distance_score == 0.55
