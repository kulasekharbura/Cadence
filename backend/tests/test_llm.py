import pytest
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from app.schemas.chat import ChatRequest, ChatMessage, SourceChunk
from app.schemas.retrieval import RetrievedChunk
from app.schemas.conversation import ConversationDetailResponse, MessageResponse
from app.services.llm_service import LLMService
from app.services.llm.provider import LLMProvider
from app.services.conversation_service import ConversationService
from datetime import datetime

class MockLLMProvider(LLMProvider):
    def __init__(self):
        self.should_fail = False
        self.last_system_prompt = None
        self.last_messages = None
        
    async def generate_answer(self, system_prompt: str, messages: list) -> str:
        if self.should_fail:
            raise RuntimeError("Mock provider failure")
        self.last_system_prompt = system_prompt
        self.last_messages = messages
        return "This is a mocked answer."

class MockRetrievalService:
    def __init__(self):
        self.should_fail = False
        self.chunks_to_return = []
        
    async def retrieve_chunks(self, query, document_ids=None, top_k=None, distance_threshold=None):
        if self.should_fail:
            raise RuntimeError("Mock retrieval failure")
        return self.chunks_to_return

class MockConversationService:
    def __init__(self):
        self.doc_ids = []
        self.messages = []
        
    async def get_conversation_detail(self, conversation_id: uuid.UUID):
        from app.schemas.conversation import ConversationDetailResponse
        return ConversationDetailResponse(
            id=conversation_id,
            title="Test",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            document_ids=self.doc_ids,
            messages=self.messages
        )
        
    async def add_messages(self, conversation_id, user_content, assistant_content, sources=None):
        pass

@pytest.fixture
def llm_service():
    retrieval = MockRetrievalService()
    provider = MockLLMProvider()
    conversation = MockConversationService()
    conversation.doc_ids = [uuid.uuid4()]
    return LLMService(retrieval_service=retrieval, llm_provider=provider, conversation_service=conversation)

@pytest.mark.asyncio
async def test_llm_service_citation_parsing(llm_service):
    # Setup 3 chunks
    doc_id = uuid.uuid4()
    chunks = []
    for i in range(3):
        chunks.append(
            RetrievedChunk(
                chunk_id=uuid.uuid4(),
                document_id=doc_id,
                filename="test.pdf",
                page_number=1,
                chunk_index=i,
                content=f"Content {i+1}",
                distance_score=0.1
            )
        )
    llm_service.retrieval_service.chunks_to_return = chunks
    
    # 1. Valid individual sources: [Source 1]
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="According to [Source 1].")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert len(resp.sources) == 1
    assert resp.sources[0].chunk_id == chunks[0].chunk_id
    
    # 2. Valid adjacent sources: [Source 1][Source 2]
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="According to [Source 1][Source 2].")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert len(resp.sources) == 2
    assert resp.sources[0].chunk_id == chunks[0].chunk_id
    assert resp.sources[1].chunk_id == chunks[1].chunk_id

    # 3. Valid grouped sources: [Source 1, Source 2] (case-insensitive)
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="According to [source 1, Source 2].")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert len(resp.sources) == 2
    assert resp.sources[0].chunk_id == chunks[0].chunk_id
    assert resp.sources[1].chunk_id == chunks[1].chunk_id

    # 4. Valid multiple grouped sources: [Source 1, Source 2, Source 3]
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="According to [Source 1, source 2, SOURCE 3].")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert len(resp.sources) == 3
    assert resp.sources[0].chunk_id == chunks[0].chunk_id
    assert resp.sources[1].chunk_id == chunks[1].chunk_id
    assert resp.sources[2].chunk_id == chunks[2].chunk_id
    
    # 5. Invalid / fabricated source numbers (e.g. 99, 4)
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="[Source 1] and [Source 4] and [Source 99].")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert len(resp.sources) == 1
    assert resp.sources[0].chunk_id == chunks[0].chunk_id
    
    # 6. Duplicate sources
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="[Source 2] says this. Also [Source 2, Source 2].")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert len(resp.sources) == 1
    assert resp.sources[0].chunk_id == chunks[1].chunk_id
    
    # 7. Out of order citations (should be ordered ascending)
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="[Source 3], [Source 1], [Source 2]")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert len(resp.sources) == 3
    assert resp.sources[0].chunk_index == 0 # Source 1
    assert resp.sources[1].chunk_index == 1 # Source 2
    assert resp.sources[2].chunk_index == 2 # Source 3
    
    # 8. Malformed citation text
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="This is [Source 1, 2] and [Source A] and [1].")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert len(resp.sources) == 0 # None of these strictly match the expected bracket format
    
    # 9. No citations
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="I do not have enough information.")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert len(resp.sources) == 0

    # 10. Content is stripped from sources
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="[Source 1]")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert not hasattr(resp.sources[0], 'content')
    
@pytest.mark.asyncio
async def test_llm_service_cross_document_citations(llm_service):
    # Setup chunks from two different documents
    doc_a_id = uuid.uuid4()
    doc_b_id = uuid.uuid4()
    
    chunk_a = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=doc_a_id,
        filename="doc_A.pdf",
        page_number=1,
        chunk_index=0,
        content="Content A",
        distance_score=0.1
    )
    chunk_b = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=doc_b_id,
        filename="doc_B.pdf",
        page_number=5,
        chunk_index=1,
        content="Content B",
        distance_score=0.1
    )
    
    llm_service.retrieval_service.chunks_to_return = [chunk_a, chunk_b]
    
    # 1. Answer citing both documents
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="[Source 1] says X, while [Source 2] says Y.")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert len(resp.sources) == 2
    
    assert resp.sources[0].document_id == doc_a_id
    assert resp.sources[0].filename == "doc_A.pdf"
    
    assert resp.sources[1].document_id == doc_b_id
    assert resp.sources[1].filename == "doc_B.pdf"
    
    # 2. Answer citing only document B
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="It is in [Source 2].")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert len(resp.sources) == 1
    assert resp.sources[0].document_id == doc_b_id
    
    # 3. LLM tries to fabricate a cross-document citation [Source 3]
    llm_service.llm_provider.generate_answer = AsyncMock(return_value="[Source 3]")
    resp = await llm_service.process_chat(ChatRequest(conversation_id=uuid.uuid4(), message="Q"))
    assert len(resp.sources) == 0
    
    # Wait, the above replaces prompt construction, I should have added it.
    # I will recreate the prompt construction test here.
    
@pytest.mark.asyncio
async def test_llm_service_prompt_construction(llm_service):
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    llm_service.retrieval_service.chunks_to_return = [
        RetrievedChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            filename="test.pdf",
            page_number=1,
            chunk_index=0,
            content="Mock content.",
            distance_score=0.1
        )
    ]
    
    req = ChatRequest(conversation_id=uuid.uuid4(), message="What is the content?")
    response = await llm_service.process_chat(req)
    
    # The default mock returns "This is a mocked answer." which has no citations.
    # So sources will be empty.
    assert len(response.sources) == 0
    
    # Check prompt construction
    sys_prompt = llm_service.llm_provider.last_system_prompt
    assert "You are an AI assistant" in sys_prompt
    assert "test.pdf" in sys_prompt
    assert "Mock content." in sys_prompt
    assert str(chunk_id) in sys_prompt
    
    # Check messages array (just the user query)
    msgs = llm_service.llm_provider.last_messages
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What is the content?"

@pytest.mark.asyncio
async def test_llm_service_empty_context(llm_service):
    # No chunks returned
    llm_service.retrieval_service.chunks_to_return = []
    
    req = ChatRequest(conversation_id=uuid.uuid4(), message="No context query")
    response = await llm_service.process_chat(req)
    
    assert len(response.sources) == 0
    sys_prompt = llm_service.llm_provider.last_system_prompt
    assert "No document context available" in sys_prompt

@pytest.mark.asyncio
async def test_llm_service_history_bounding(llm_service):
    llm_service.retrieval_service.chunks_to_return = []
    
    # Create 10 history messages
    history = [MessageResponse(id=uuid.uuid4(), role="user", content=f"Msg {i}", created_at=datetime.utcnow()) for i in range(10)]
    llm_service.conversation_service.messages = history
    req = ChatRequest(conversation_id=uuid.uuid4(), message="Final query")
    
    await llm_service.process_chat(req)
    
    # With default limit of 5, we should see 5 history + 1 new query = 6 messages
    msgs = llm_service.llm_provider.last_messages
    assert len(msgs) == 6
    assert msgs[0]["content"] == "Msg 5" # The 6th message (index 5)
    assert msgs[-1]["content"] == "Final query"

@pytest.mark.asyncio
async def test_llm_service_provider_failure(llm_service):
    llm_service.llm_provider.should_fail = True
    
    req = ChatRequest(conversation_id=uuid.uuid4(), message="Query")
    with pytest.raises(RuntimeError, match="Mock provider failure"):
        await llm_service.process_chat(req)

@pytest.mark.asyncio
async def test_llm_service_retrieval_failure(llm_service):
    llm_service.retrieval_service.should_fail = True
    
    req = ChatRequest(conversation_id=uuid.uuid4(), message="Query")
    with pytest.raises(RuntimeError, match="Mock retrieval failure"):
        await llm_service.process_chat(req)

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_endpoint_success():
    # We must mock get_llm_service dependency
    from app.api.endpoints.chat import get_llm_service
    
    # Create a mock llm service
    retrieval = MockRetrievalService()
    provider = MockLLMProvider()
    conversation = MockConversationService()
    conversation.doc_ids = [uuid.uuid4()]
    mock_llm_svc = LLMService(retrieval, provider, conversation)
    
    app.dependency_overrides[get_llm_service] = lambda: mock_llm_svc
    
    from app.core.config import settings
    response = client.post(f"{settings.API_V1_STR}/chat", json={"conversation_id": str(uuid.uuid4()), "message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a mocked answer."
    assert data["sources"] == []
    
    app.dependency_overrides.clear()

def test_chat_endpoint_failure():
    from app.api.endpoints.chat import get_llm_service
    retrieval = MockRetrievalService()
    provider = MockLLMProvider()
    provider.should_fail = True # Trigger failure
    conversation = MockConversationService()
    conversation.doc_ids = [uuid.uuid4()]
    mock_llm_svc = LLMService(retrieval, provider, conversation)
    
    app.dependency_overrides[get_llm_service] = lambda: mock_llm_svc
    
    from app.core.config import settings
    response = client.post(f"{settings.API_V1_STR}/chat", json={"conversation_id": str(uuid.uuid4()), "message": "Hello"})
    assert response.status_code == 500
    assert "Mock provider failure" in response.json()["detail"]
    
    app.dependency_overrides.clear()
