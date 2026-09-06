from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
from app.services.conversation_service import ConversationService

router = APIRouter()

def get_llm_service(db: AsyncSession = Depends(get_db)) -> LLMService:
    retrieval_service = RetrievalService(db)
    conversation_service = ConversationService(db)
    return LLMService(retrieval_service=retrieval_service, conversation_service=conversation_service)

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, llm_service: LLMService = Depends(get_llm_service)):
    """
    Process a chat message, retrieve context, and return a grounded answer.
    """
    try:
        response = await llm_service.process_chat(request)
        return response
    except ValueError as e:
        # e.g., missing OpenAI API key
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        # e.g., OpenAI API failure or retrieval failure
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")
