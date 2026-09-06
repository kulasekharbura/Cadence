import pytest
import uuid
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.db.session import async_session_maker
from app.models.document import Document
from app.core.config import settings

@pytest.fixture
async def sample_docs():
    async with async_session_maker() as db:
        doc1 = Document(filename="doc1.pdf", file_path="/fake/d1.pdf", file_size=100)
        doc2 = Document(filename="doc2.pdf", file_path="/fake/d2.pdf", file_size=100)
        db.add_all([doc1, doc2])
        await db.commit()
        await db.refresh(doc1)
        await db.refresh(doc2)
        return doc1.id, doc2.id

@pytest.mark.asyncio
async def test_create_conversation_one_doc(sample_docs):
    doc1_id, doc2_id = sample_docs
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(f"{settings.API_V1_STR}/conversations", json={"document_ids": [str(doc1_id)]})
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "New Conversation"
        assert str(doc1_id) in data["document_ids"]
        assert len(data["document_ids"]) == 1

@pytest.mark.asyncio
async def test_create_conversation_multiple_docs(sample_docs):
    doc1_id, doc2_id = sample_docs
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(f"{settings.API_V1_STR}/conversations", json={"document_ids": [str(doc1_id), str(doc2_id)]})
        assert resp.status_code == 201
        data = resp.json()
        assert len(data["document_ids"]) == 2
        assert str(doc1_id) in data["document_ids"]
        assert str(doc2_id) in data["document_ids"]

@pytest.mark.asyncio
async def test_create_conversation_nonexistent_doc():
    bad_id = str(uuid.uuid4())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(f"{settings.API_V1_STR}/conversations", json={"document_ids": [bad_id]})
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

@pytest.mark.asyncio
async def test_create_conversation_no_docs():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(f"{settings.API_V1_STR}/conversations", json={"document_ids": []})
        assert resp.status_code == 400
        assert "at least one document" in resp.json()["detail"].lower()

@pytest.mark.asyncio
async def test_create_conversation_duplicate_docs(sample_docs):
    doc1_id, _ = sample_docs
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Send same doc twice
        resp = await ac.post(f"{settings.API_V1_STR}/conversations", json={"document_ids": [str(doc1_id), str(doc1_id)]})
        assert resp.status_code == 201
        data = resp.json()
        # Should deduplicate natively
        assert len(data["document_ids"]) == 1

@pytest.mark.asyncio
async def test_list_and_get_conversation(sample_docs):
    doc1_id, _ = sample_docs
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create
        resp = await ac.post(f"{settings.API_V1_STR}/conversations", json={"document_ids": [str(doc1_id)]})
        conv_id = resp.json()["id"]
        
        # List
        list_resp = await ac.get(f"{settings.API_V1_STR}/conversations")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert any(c["id"] == conv_id for c in list_data)
        
        # Get Detail
        det_resp = await ac.get(f"{settings.API_V1_STR}/conversations/{conv_id}")
        assert det_resp.status_code == 200
        det_data = det_resp.json()
        assert det_data["id"] == conv_id
        assert len(det_data["document_ids"]) == 1
        assert det_data["document_ids"][0] == str(doc1_id)
        assert det_data["messages"] == []

@pytest.mark.asyncio
async def test_delete_conversation(sample_docs):
    doc1_id, _ = sample_docs
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(f"{settings.API_V1_STR}/conversations", json={"document_ids": [str(doc1_id)]})
        conv_id = resp.json()["id"]
        
        del_resp = await ac.delete(f"{settings.API_V1_STR}/conversations/{conv_id}")
        assert del_resp.status_code == 204
        
        # Try to get conversation -> should be 404
        det_resp = await ac.get(f"{settings.API_V1_STR}/conversations/{conv_id}")
        assert det_resp.status_code == 404

        # Document should still exist
        async with async_session_maker() as db:
            doc = await db.get(Document, doc1_id)
            assert doc is not None
