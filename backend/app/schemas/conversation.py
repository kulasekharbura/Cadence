from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
import uuid
from datetime import datetime
from app.schemas.chat import SourceChunk

class ConversationCreate(BaseModel):
    document_ids: List[uuid.UUID]

class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    sources: List[SourceChunk] = Field(default_factory=list)

class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    document_ids: List[uuid.UUID]

class ConversationDetailResponse(ConversationResponse):
    messages: List[MessageResponse]
