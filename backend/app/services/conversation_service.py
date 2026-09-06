import uuid
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from app.models.conversation import Conversation, ConversationDocument, Message, MessageSource
from app.models.document import Document
from app.schemas.conversation import ConversationCreate, ConversationResponse, ConversationDetailResponse, MessageResponse
from app.schemas.chat import SourceChunk

class ConversationService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_conversation(self, req: ConversationCreate) -> ConversationResponse:
        # Validate documents exist
        if not req.document_ids:
            raise HTTPException(status_code=400, detail="Must provide at least one document_id")
            
        stmt = select(Document.id).where(Document.id.in_(req.document_ids))
        result = await self.db.execute(stmt)
        existing_doc_ids = result.scalars().all()
        
        missing_docs = set(req.document_ids) - set(existing_doc_ids)
        if missing_docs:
            raise HTTPException(status_code=404, detail=f"Documents not found: {missing_docs}")

        # Remove duplicate document IDs if provided
        unique_doc_ids = list(set(req.document_ids))

        # Create conversation
        conv = Conversation(title="New Conversation")
        self.db.add(conv)
        await self.db.flush()

        # Add associations
        for doc_id in unique_doc_ids:
            assoc = ConversationDocument(conversation_id=conv.id, document_id=doc_id)
            self.db.add(assoc)

        await self.db.commit()
        await self.db.refresh(conv)

        return ConversationResponse(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            document_ids=unique_doc_ids
        )

    async def get_conversations(self) -> List[ConversationResponse]:
        stmt = select(Conversation).options(selectinload(Conversation.documents)).order_by(Conversation.created_at.desc())
        result = await self.db.execute(stmt)
        convs = result.scalars().all()
        
        responses = []
        for c in convs:
            doc_ids = [doc.document_id for doc in c.documents]
            responses.append(ConversationResponse(
                id=c.id,
                title=c.title,
                created_at=c.created_at,
                updated_at=c.updated_at,
                document_ids=doc_ids
            ))
        return responses

    async def get_conversation_detail(self, conversation_id: uuid.UUID) -> ConversationDetailResponse:
        stmt = select(Conversation).options(
            selectinload(Conversation.documents),
            selectinload(Conversation.messages).selectinload(Message.sources)
        ).where(Conversation.id == conversation_id)
        
        result = await self.db.execute(stmt)
        conv = result.scalar_one_or_none()
        
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        doc_ids = [doc.document_id for doc in conv.documents]
        messages = [
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
                sources=[
                    SourceChunk(
                        source_number=src.source_number,
                        chunk_id=src.chunk_id,
                        document_id=src.document_id,
                        filename=src.filename,
                        page_number=src.page_number,
                        chunk_index=src.chunk_index
                    ) for src in m.sources
                ] if m.sources else []
            ) for m in conv.messages
        ]
        
        return ConversationDetailResponse(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            document_ids=doc_ids,
            messages=messages
        )

    async def delete_conversation(self, conversation_id: uuid.UUID) -> None:
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.db.execute(stmt)
        conv = result.scalar_one_or_none()
        
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        await self.db.delete(conv)
        await self.db.commit()

    async def add_messages(self, conversation_id: uuid.UUID, user_content: str, assistant_content: str, sources: Optional[List[SourceChunk]] = None) -> None:
        """
        Persists a user/assistant exchange transactionally.
        """
        user_msg = Message(conversation_id=conversation_id, role="user", content=user_content)
        asst_msg = Message(conversation_id=conversation_id, role="assistant", content=assistant_content)
        
        self.db.add_all([user_msg, asst_msg])
        await self.db.flush()
        
        if sources:
            for s in sources:
                msg_source = MessageSource(
                    message_id=asst_msg.id,
                    source_number=s.source_number,
                    chunk_id=s.chunk_id,
                    document_id=s.document_id,
                    filename=s.filename,
                    page_number=s.page_number,
                    chunk_index=s.chunk_index
                )
                self.db.add(msg_source)
                
        await self.db.commit()
