import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import async_session_maker
from app.services.retrieval_service import RetrievalService
from sqlalchemy import select
from app.models.document import Document
import app.models.conversation

async def main():
    async with async_session_maker() as db:
        # Get documents
        result = await db.execute(select(Document))
        docs = result.scalars().all()
        for d in docs:
            print(f"Doc: {d.filename} - {d.id}")
        
        doc_ids = [d.id for d in docs]
        
        rs = RetrievalService(db)
        
        print("\n--- QUERY: what are projects of kulasekhar ---")
        chunks = await rs.retrieve_chunks("what are projects of kulasekhar", document_ids=doc_ids, top_k=10, distance_threshold=100.0)
        for i, c in enumerate(chunks):
            print(f"{i+1}. Doc: {c.filename}, Distance: {c.distance_score:.4f}, Content: {c.content[:50]}...")
            
        print("\n--- QUERY: What degree is he pursuing? ---")
        chunks = await rs.retrieve_chunks("What degree is he pursuing?", document_ids=doc_ids, top_k=10, distance_threshold=100.0)
        for i, c in enumerate(chunks):
            print(f"{i+1}. Doc: {c.filename}, Distance: {c.distance_score:.4f}, Content: {c.content[:50]}...")

if __name__ == "__main__":
    asyncio.run(main())
