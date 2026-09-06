import pytest
import fitz
import os
import uuid
import asyncio
from unittest.mock import patch, AsyncMock

from app.models.chunk import DocumentChunk
from app.db.session import async_session_maker
from app.core.config import settings

from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.fixture(autouse=True)
def setup_dummy_gemini_key():
    original_key = settings.GEMINI_API_KEY
    settings.GEMINI_API_KEY = "dummy-key-for-tests"
    yield
    settings.GEMINI_API_KEY = original_key

def create_e2e_pdf(path: str, pages: list[str]):
    doc = fitz.open()
    for text in pages:
        p = doc.new_page()
        p.insert_text((10, 10), text)
    doc.save(path)
    doc.close()

# We will intercept the LLM chat completion to verify the context it received.
# And also return deterministic answers.
class MockChatCompletionResponse:
    def __init__(self, content):
        self.text = content
        message = type("obj", (object,), {"content": content})
        choice = type("obj", (object,), {"message": message})
        self.choices = [choice]

def make_mock_chat(context_verify_callback=None):
    async def mock_chat_create(*args, **kwargs):
        contents = kwargs.get("contents", [])
        config = kwargs.get("config")
        
        system_prompt = config.system_instruction if config else ""
        
        user_message = ""
        if contents:
            last_content = contents[-1]
            if hasattr(last_content, "parts") and last_content.parts:
                user_message = last_content.parts[0].text
                
        if context_verify_callback:
            context_verify_callback(system_prompt, user_message)
            
        if "Apple" in user_message:
            if "Apple" in system_prompt:
                return MockChatCompletionResponse("The answer is Apple. [Source 1]")
                    
        if "Cherry" in user_message:
            if "Cherry" in system_prompt:
                return MockChatCompletionResponse("The answer is Cherry. [Source 1]")
                    
        if "Trigger Fake 999" in user_message:
            return MockChatCompletionResponse("The answer is Fake. [Source 999]")
            
        return MockChatCompletionResponse("I don't know.")
    return mock_chat_create

async def mock_embeddings_create(texts):
    if isinstance(texts, str):
        texts = [texts]
        
    dim = settings.GEMINI_EMBEDDING_DIMENSION
    data_list = []
    
    for text in texts:
        if "Apple" in text:
            vector = [0.1 if i % 2 == 0 else 0.0 for i in range(dim)]
        elif "Cherry" in text:
            vector = [0.0 if i % 2 == 0 else 0.1 for i in range(dim)]
        else:
            vector = [-0.1 for _ in range(dim)]
        data_list.append(vector)
        
    return data_list

@pytest.fixture
def mock_embeddings():
    with patch("app.services.embedding.gemini_provider.GeminiEmbeddingProvider.get_embeddings") as mock_get_embeddings:
        mock_get_embeddings.side_effect = mock_embeddings_create
        yield mock_get_embeddings

@pytest.mark.asyncio
async def test_e2e_rag_pipeline_with_grounding(tmp_path, mock_embeddings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a single document with two pages
        pdf_path = tmp_path / "e2e_doc.pdf"
        create_e2e_pdf(str(pdf_path), ["Fact 1: Apple Banana.", "Fact 2: Cherry Date."])
        
        with open(pdf_path, "rb") as f:
            response = await client.post("/api/v1/documents/upload", files={"file": ("e2e_doc.pdf", f, "application/pdf")})
            assert response.status_code == 200
            doc_data = response.json()
            doc_id = doc_data["id"]
            
        # Poll processing (bound deterministically)
        for _ in range(120): # 30s timeout max
            doc_res = await client.get(f"/api/v1/documents/{doc_id}")
            doc_status = doc_res.json()["processing_status"]
            if doc_status == "READY":
                break
            if doc_status == "FAILED":
                error_msg = doc_res.json().get("error_message", "Unknown error")
                pytest.fail(f"Document processing failed: {error_msg}")
            await asyncio.sleep(0.25)
        else:
            pytest.fail("Document processing timed out.")
            
        # Verify chunks created and stored in DB
        async with async_session_maker() as db:
            chunks_res = await db.execute(DocumentChunk.__table__.select().where(DocumentChunk.document_id == doc_id).order_by(DocumentChunk.page_number))
            chunks = chunks_res.fetchall()
            assert len(chunks) == 2
            assert chunks[0].page_number == 1
            assert "Apple" in chunks[0].content
            assert chunks[1].page_number == 2
            assert "Cherry" in chunks[1].content
            
        # Create conversation
        conv_res = await client.post("/api/v1/conversations", json={"document_ids": [doc_id]})
        assert conv_res.status_code == 201
        conv_id = conv_res.json()["id"]
        
        # Test A: Question whose answer exists (Page 1)
        with patch("app.services.llm.gemini_provider.genai.Client") as mock_client:
            def verify_apple_context(sys_prompt, usr_prompt):
                # Verify that the Apple chunk was actually retrieved and supplied
                assert "Apple Banana" in sys_prompt
                # Given pgvector retrieval, Cherry Date might also be there if it passed the threshold, 
                # but Apple MUST be there because it matches exactly
            
            mock_client.return_value.aio.models.generate_content = AsyncMock(side_effect=make_mock_chat(verify_apple_context))
            
            chat_req = {"conversation_id": conv_id, "message": "Tell me about Apple"}
            chat_res = await client.post("/api/v1/chat", json=chat_req)
            assert chat_res.status_code == 200
            chat_data = chat_res.json()
            
            # Verify citation mapped correctly to Page 1
            assert "Apple" in chat_data["answer"]
            assert len(chat_data["sources"]) == 1
            assert chat_data["sources"][0]["document_id"] == doc_id
            assert chat_data["sources"][0]["filename"] == "e2e_doc.pdf"
            assert chat_data["sources"][0]["page_number"] == 1
            assert chat_data["sources"][0]["chunk_index"] == 0
            
        # Test A2: Question whose answer exists (Page 2)
        with patch("app.services.llm.gemini_provider.genai.Client") as mock_client:
            def verify_cherry_context(sys_prompt, usr_prompt):
                assert "Cherry Date" in sys_prompt
            
            mock_client.return_value.aio.models.generate_content = AsyncMock(side_effect=make_mock_chat(verify_cherry_context))
            
            chat_req2 = {"conversation_id": conv_id, "message": "Tell me about Cherry"}
            chat_res2 = await client.post("/api/v1/chat", json=chat_req2)
            assert chat_res2.status_code == 200
            chat_data2 = chat_res2.json()
            
            # Verify citation mapped correctly to Page 2
            assert "Cherry" in chat_data2["answer"]
            assert len(chat_data2["sources"]) == 1
            assert chat_data2["sources"][0]["page_number"] == 2
            
        # Test C: Unrelated Question
        with patch("app.services.llm.gemini_provider.genai.Client") as mock_client:
            def verify_unrelated_context(sys_prompt, usr_prompt):
                # Since query embedding is [-0.9]*dim, distance to [0.1]*dim is > 1.5. 
                # Our distance threshold is 1.0. So it shouldn't be in context.
                assert "Apple Banana" not in sys_prompt
                assert "Cherry Date" not in sys_prompt
            
            mock_client.return_value.aio.models.generate_content = AsyncMock(side_effect=make_mock_chat(verify_unrelated_context))
            
            chat_req3 = {"conversation_id": conv_id, "message": "Unrelated"}
            chat_res3 = await client.post("/api/v1/chat", json=chat_req3)
            chat_data3 = chat_res3.json()
            assert "I don't know" in chat_data3["answer"]
            assert len(chat_data3["sources"]) == 0
            
        # Test E: Invalid Citation Rejection
        with patch("app.services.llm.gemini_provider.genai.Client") as mock_client:
            mock_client.return_value.aio.models.generate_content = AsyncMock(side_effect=make_mock_chat())
            
            chat_req4 = {"conversation_id": conv_id, "message": "Trigger Fake 999"}
            chat_res4 = await client.post("/api/v1/chat", json=chat_req4)
            chat_data4 = chat_res4.json()
            assert "Fake" in chat_data4["answer"]
            # The backend should drop Source 999
            assert len(chat_data4["sources"]) == 0

@pytest.mark.asyncio
async def test_e2e_multiple_documents_filtering(tmp_path, mock_embeddings):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Document A (Apple)
        pdf_path_a = tmp_path / "doc_a.pdf"
        create_e2e_pdf(str(pdf_path_a), ["Fact: Apple."])
        
        # Document B (Cherry)
        pdf_path_b = tmp_path / "doc_b.pdf"
        create_e2e_pdf(str(pdf_path_b), ["Fact: Cherry."])
        
        with open(pdf_path_a, "rb") as f:
            id_a = (await client.post("/api/v1/documents/upload", files={"file": ("doc_a.pdf", f, "application/pdf")})).json()["id"]
            
        with open(pdf_path_b, "rb") as f:
            id_b = (await client.post("/api/v1/documents/upload", files={"file": ("doc_b.pdf", f, "application/pdf")})).json()["id"]
            
        # Poll both
        for doc_id in [id_a, id_b]:
            for _ in range(120):
                res = await client.get(f"/api/v1/documents/{doc_id}")
                if res.json()["processing_status"] == "READY":
                    break
                await asyncio.sleep(0.25)
                
        # Scoped ONLY to Document A
        conv_res = await client.post("/api/v1/conversations", json={"document_ids": [id_a]})
        conv_id = conv_res.json()["id"]
        
        # Test D: Multiple Docs Filter - Question about Cherry (Document B)
        with patch("app.services.llm.gemini_provider.genai.Client") as mock_client:
            def verify_filtered_context(sys_prompt, usr_prompt):
                # Cherry chunk is in Document B, so it MUST NOT be retrieved, even if query matches Cherry.
                assert "Cherry" not in sys_prompt
                
            mock_client.return_value.aio.models.generate_content = AsyncMock(side_effect=make_mock_chat(verify_filtered_context))
            
            chat_req = {"conversation_id": conv_id, "message": "Tell me about Cherry"}
            chat_res = await client.post("/api/v1/chat", json=chat_req)
            chat_data = chat_res.json()
            assert "I don't know" in chat_data["answer"]
            assert len(chat_data["sources"]) == 0
            
        # Verify Apple still works
        with patch("app.services.llm.gemini_provider.genai.Client") as mock_client:
            def verify_apple_context(sys_prompt, usr_prompt):
                assert "Apple" in sys_prompt
                
            mock_client.return_value.aio.models.generate_content = AsyncMock(side_effect=make_mock_chat(verify_apple_context))
            
            chat_req2 = {"conversation_id": conv_id, "message": "Tell me about Apple"}
            chat_res2 = await client.post("/api/v1/chat", json=chat_req2)
            chat_data2 = chat_res2.json()
            assert "Apple" in chat_data2["answer"]
            assert len(chat_data2["sources"]) == 1
            assert chat_data2["sources"][0]["document_id"] == id_a
