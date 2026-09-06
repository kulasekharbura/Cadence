import pytest
import asyncio
import os
import sys

from app.db.session import async_session_maker as SessionLocal
from app.models.document import Document
from app.models.conversation import Conversation, ConversationDocument, Message
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import pprint

@pytest.mark.asyncio
async def test_diagnose():
    async with SessionLocal() as db:
        print("=== 1. Find the exact conversation ===")
        # Get the latest message containing "What is CGPA"
        msg_stmt = select(Message).filter(Message.content.like("%What is CGPA of Sekahr%")).order_by(Message.created_at.desc()).limit(1)
        msg_result = await db.execute(msg_stmt)
        msg = msg_result.scalar_one_or_none()
        
        if not msg:
            print("Message not found. Trying to find the conversation with 2 documents.")
            conv_stmt = select(Conversation).options(selectinload(Conversation.documents).selectinload(ConversationDocument.document)).order_by(Conversation.created_at.desc()).limit(10)
            conv_result = await db.execute(conv_stmt)
            conversations = conv_result.scalars().all()
            for c in conversations:
                docs = c.documents
                if len(docs) == 2:
                    conversation_id = c.id
                    document_ids = [d.document_id for d in docs]
                    document_names = [d.document.filename for d in docs]
                    print(f"Found conversation with 2 docs: {conversation_id}")
                    print(f"Docs: {document_names}")
                    break
            else:
                print("No 2-document conversation found.")
                return
        else:
            conversation_id = msg.conversation_id
            print(f"Found message: {msg.content}")
            print(f"Conversation ID: {conversation_id}")
            
            conv_stmt = select(Conversation).options(selectinload(Conversation.documents).selectinload(ConversationDocument.document)).filter(Conversation.id == conversation_id)
            conv_result = await db.execute(conv_stmt)
            conv = conv_result.scalar_one()
            document_ids = [d.document_id for d in conv.documents]
            document_names = [d.document.filename for d in conv.documents]
            print(f"Docs in conversation: {document_names}")

        print("\n=== 2. Retrieve chunks for the CGPA question ===")
        query = "What is CGPA of Sekahr?"
        
        embedding_provider = EmbeddingService()
        retrieval_service = RetrievalService(db, embedding_provider)
        
        retrieval_result = await retrieval_service.retrieve_chunks(
            query=query,
            document_ids=document_ids,
            top_k=5
        )
        
        print("\n=== 3. Retrieved Chunks ===")
        for i, chunk in enumerate(retrieval_result):
            print(f"source_number: {i+1}")
            print(f"filename: {chunk.filename}")
            print(f"page_number: {chunk.page_number}")
            print(f"chunk_id: {chunk.chunk_id}")
            print(f"distance: {chunk.distance_score:.4f}")
            print(f"preview: {repr(chunk.content[:100])}...")
            print("-" * 40)
            
        print("\n=== 4. Source mapping passed to LLMService ===")
        source_mapping = { (i + 1): chunk for i, chunk in enumerate(retrieval_result) }
        for num, chunk in source_mapping.items():
            print(f"{num} -> {chunk.filename} (ID: {chunk.chunk_id})")
            
        print("\n=== 5. Exact Gemini Prompt (Simulated) ===")
        context_prompt = "Context information is below.\n---------------------\n"
        for i, chunk in enumerate(retrieval_result):
            context_prompt += f"\n[Source {i+1}]\n"
            context_prompt += f"Document: {chunk.filename}\n"
            context_prompt += f"Content:\n{chunk.content}\n"
            context_prompt += "-" * 20 + "\n"
            
        print(context_prompt[:500] + "\n... [truncated] ...\n" + context_prompt[-500:])

        print("\n=== 6. Gemini Generation Call ===")
        from app.services.conversation_service import ConversationService
        conversation_service = ConversationService(db)
        llm_service = LLMService(retrieval_service, conversation_service=conversation_service)
        
        from app.schemas.chat import ChatRequest
        request = ChatRequest(conversation_id=conversation_id, message=query)
        chat_response = await llm_service.process_chat(request)
        
        print(f"Raw Answer (with markers):\n{chat_response.answer}")
        
        print("\n=== 7. Backend's parsed citations ===")
        for src in chat_response.sources:
            print(f"Source {src.source_number}: {src.filename} p.{src.page_number}")
