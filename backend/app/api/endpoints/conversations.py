import uuid
from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.schemas.conversation import ConversationCreate, ConversationResponse, ConversationDetailResponse
from app.services.conversation_service import ConversationService

router = APIRouter()

def get_conversation_service(db: AsyncSession = Depends(get_db)) -> ConversationService:
    return ConversationService(db)

@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    req: ConversationCreate,
    service: ConversationService = Depends(get_conversation_service)
):
    return await service.create_conversation(req)

@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    service: ConversationService = Depends(get_conversation_service)
):
    return await service.get_conversations()

@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    service: ConversationService = Depends(get_conversation_service)
):
    return await service.get_conversation_detail(conversation_id)

@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID,
    service: ConversationService = Depends(get_conversation_service)
):
    await service.delete_conversation(conversation_id)
