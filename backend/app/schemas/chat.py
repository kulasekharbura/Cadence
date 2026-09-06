from pydantic import BaseModel, Field
from typing import List, Optional
import uuid

class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    conversation_id: uuid.UUID
    message: str

class SourceChunk(BaseModel):
    source_number: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_index: int

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
