import pytest
from httpx import AsyncClient, ASGITransport
import asyncio
import fitz
from unittest.mock import patch

from app.main import app

def create_valid_pdf(path: str):
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Page 1 Content")
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Page 2 Content")
    doc.save(path)
    doc.close()

@pytest.mark.asyncio
async def test_upload_invalid_file():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/v1/documents/upload", files={"file": ("test.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."

@pytest.mark.asyncio
@patch("app.services.embedding.gemini_provider.GeminiEmbeddingProvider.get_embeddings")
async def test_upload_valid_file_and_lifecycle(mock_get_embeddings, tmp_path):
    # Mock the embeddings to succeed
    async def mock_embeddings_side_effect(texts):
        return [[0.1]*1536 for _ in texts]
    mock_get_embeddings.side_effect = mock_embeddings_side_effect
    
    pdf_path = tmp_path / "valid.pdf"
    create_valid_pdf(str(pdf_path))
    
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/documents/upload", 
            files={"file": ("valid.pdf", pdf_content, "application/pdf")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "valid.pdf"
    assert data["processing_status"] == "UPLOADED"
    doc_id = data["id"]
    
    # Poll for READY status
    max_retries = 20
    for _ in range(max_retries):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            doc_res = await ac.get(f"/api/v1/documents/{doc_id}")
            doc_data = doc_res.json()
            if doc_data["processing_status"] == "READY":
                break
        await asyncio.sleep(0.5)
        
    assert doc_data["processing_status"] == "READY"
    assert doc_data["page_count"] == 2
    assert "file_path" not in doc_data # Assert we don't expose file_path in response

@pytest.mark.asyncio
async def test_upload_corrupt_file_lifecycle():
    # Corrupt PDF content
    corrupt_content = b"%PDF-1.4\n%EOF\n"
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/documents/upload", 
            files={"file": ("corrupt.pdf", corrupt_content, "application/pdf")}
        )
    
    assert response.status_code == 200
    doc_id = response.json()["id"]
    
    # Poll for FAILED status
    max_retries = 20
    for _ in range(max_retries):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            doc_res = await ac.get(f"/api/v1/documents/{doc_id}")
            doc_data = doc_res.json()
            if doc_data["processing_status"] == "FAILED":
                break
        await asyncio.sleep(0.5)
        
    assert doc_data["processing_status"] == "FAILED"
    assert doc_data["error_message"] is not None

@pytest.mark.asyncio
async def test_delete_document():
    # We can mock get_embeddings here if needed, but since we just insert directly or let it fail,
    # let's insert a fake document directly via DB or rely on the upload.
    # To keep it simple, we'll use an API upload and then delete it.
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        corrupt_content = b"%PDF-1.4\n%EOF\n"
        upload_resp = await ac.post(
            "/api/v1/documents/upload", 
            files={"file": ("to_delete.pdf", corrupt_content, "application/pdf")}
        )
        assert upload_resp.status_code == 200
        doc_id = upload_resp.json()["id"]
        
        # Delete it
        del_resp = await ac.delete(f"/api/v1/documents/{doc_id}")
        assert del_resp.status_code == 204
        
        # Verify it's gone
        get_resp = await ac.get(f"/api/v1/documents/{doc_id}")
        assert get_resp.status_code == 404

@pytest.mark.asyncio
async def test_delete_nonexistent_document():
    import uuid
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        del_resp = await ac.delete(f"/api/v1/documents/{uuid.uuid4()}")
        assert del_resp.status_code == 404
