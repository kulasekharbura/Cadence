import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.core.config import settings
from app.db.session import async_session_maker
import uuid

@pytest.mark.asyncio
async def test_pgvector_extension_exists():
    async with async_session_maker() as db_session:
        # Verify extension is installed
        result = await db_session.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
        assert result.scalar() == "vector"

@pytest.mark.asyncio
async def test_persist_document_chunk_with_embedding():
    async with async_session_maker() as db_session:
        # Create a document first to satisfy FK constraint
        doc = Document(filename="test.pdf", file_path="/fake/test.pdf", file_size=1024, page_count=1)
        db_session.add(doc)
        await db_session.commit()
        await db_session.refresh(doc)
        
        # Create a vector
        dummy_vector = [0.01] * settings.GEMINI_EMBEDDING_DIMENSION
        
        chunk = DocumentChunk(
            document_id=doc.id,
            page_number=1,
            chunk_index=0,
            content="This is a test chunk.",
            embedding=dummy_vector
        )
        
        db_session.add(chunk)
        await db_session.commit()
        await db_session.refresh(chunk)
        
        assert chunk.id is not None
        assert len(chunk.embedding) == settings.GEMINI_EMBEDDING_DIMENSION
        assert chunk.embedding[0] == pytest.approx(0.01)

@pytest.mark.asyncio
async def test_cosine_distance_query():
    async with async_session_maker() as db_session:
        doc = Document(filename="test2.pdf", file_path="/fake/test2.pdf", file_size=1024, page_count=1)
        db_session.add(doc)
        await db_session.commit()
        await db_session.refresh(doc)
        
        dim = settings.GEMINI_EMBEDDING_DIMENSION
        vec1 = [0.1] * dim
        vec2 = [-0.1] * dim # Opposite direction, high cosine distance
        vec3 = [0.09] * dim # Very close direction, low cosine distance
        
        c1 = DocumentChunk(document_id=doc.id, page_number=1, chunk_index=0, content="C1", embedding=vec1)
        c2 = DocumentChunk(document_id=doc.id, page_number=1, chunk_index=1, content="C2", embedding=vec2)
        c3 = DocumentChunk(document_id=doc.id, page_number=1, chunk_index=2, content="C3", embedding=vec3)
        
        db_session.add_all([c1, c2, c3])
        await db_session.commit()
        
        # Search for vector closest to vec1 using cosine distance (<=>)
        from sqlalchemy import select
        # pgvector provides cosine distance using cosine_distance method on the Column
        stmt = select(DocumentChunk.content).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.embedding.cosine_distance(vec1)).limit(2)
        
        result = await db_session.execute(stmt)
        closest_contents = [row[0] for row in result.all()]
        
        assert closest_contents[0] == "C1" # Distance 0
        assert closest_contents[1] == "C3" # Small distance
    
@pytest.mark.asyncio
async def test_cascade_delete():
    async with async_session_maker() as db_session:
        doc = Document(filename="test3.pdf", file_path="/fake/test3.pdf", file_size=1024, page_count=1)
        db_session.add(doc)
        await db_session.commit()
        
        c1 = DocumentChunk(document_id=doc.id, page_number=1, chunk_index=0, content="C1", embedding=[0.1]*1536)
        db_session.add(c1)
        await db_session.commit()
        
        # Delete doc
        await db_session.delete(doc)
        await db_session.commit()
        
        # Verify chunk deleted
        from sqlalchemy import select
        result = await db_session.execute(select(DocumentChunk).where(DocumentChunk.document_id == doc.id))
        assert len(result.all()) == 0
