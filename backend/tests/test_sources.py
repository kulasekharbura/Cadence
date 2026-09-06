import pytest
from httpx import AsyncClient, ASGITransport
from app.models.document import Document
from app.models.chunk import DocumentChunk
import uuid
from app.main import app
from app.db.session import async_session_maker

@pytest.mark.asyncio
async def test_get_source_chunk_success():
    # Setup test data
    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, filename="test_doc.pdf", file_size=1024, file_path="foo.pdf")
    
    chunk_id = uuid.uuid4()
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=doc_id,
        page_number=1,
        chunk_index=0,
        content="This is a test chunk",
        embedding=[0.1] * 1536
    )
    
    async with async_session_maker() as db_session:
        db_session.add(doc)
        db_session.add(chunk)
        await db_session.commit()
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(f"/api/v1/sources/{chunk_id}")
        
    assert response.status_code == 200
    data = response.json()
    assert data["chunk_id"] == str(chunk_id)
    assert data["document_id"] == str(doc_id)
    assert data["filename"] == "test_doc.pdf"
    assert data["page_number"] == 1
    assert data["content"] == "This is a test chunk"

@pytest.mark.asyncio
async def test_get_source_chunk_not_found():
    random_id = uuid.uuid4()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(f"/api/v1/sources/{random_id}")
    assert response.status_code == 404
