import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import async_session_maker
from app.models.chunk import DocumentChunk
from app.models.document import Document
import app.models.conversation
from sqlalchemy import select
from app.services.embedding_service import EmbeddingService

async def main():
    async with async_session_maker() as db:
        # Search for a chunk that contains "B.Tech Information Technology"
        stmt = select(DocumentChunk).where(DocumentChunk.content.ilike("%B.Tech Information Technology%"))
        result = await db.execute(stmt)
        chunks = result.scalars().all()
        
        if not chunks:
            print("No chunks found with 'B.Tech Information Technology'")
            
            # Let's see if there are ANY chunks
            result = await db.execute(select(DocumentChunk).limit(5))
            all_chunks = result.scalars().all()
            print(f"Total random chunks found: {len(all_chunks)}")
            for c in all_chunks:
                print(f"Chunk id: {c.id}, content: {c.content[:50]}")
            return
            
        chunk = chunks[0]
        print(f"Found Chunk ID: {chunk.id}")
        print(f"Chunk Content:\n{chunk.content}\n")
        
        # Embed the query
        query = "What degree is he pursuing?"
        # EmbeddingService requires a DocumentChunk list, but we can bypass that if it's not strictly typed or just use the wrapper
        es = EmbeddingService()
        from app.schemas.chunk import DocumentChunk as ChunkSchema
        dummy = ChunkSchema(document_id="00000000-0000-0000-0000-000000000000", page_number=0, chunk_index=0, content=query)
        embedded_result = await es.generate_embeddings_for_chunks([dummy])
        query_vector = embedded_result[0]["embedding"]
        
        # Calculate distance to our target chunk
        distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
        stmt_dist = select(distance_col).where(DocumentChunk.id == chunk.id)
        dist_result = await db.execute(stmt_dist)
        distance = dist_result.scalar()
        
        print(f"Query Embedding Used (first 5 dims): {query_vector[:5]}")
        print(f"Cosine Distance: {distance}")
        from app.core.config import settings
        print(f"Current Threshold: {settings.RETRIEVAL_DISTANCE_THRESHOLD}")
        print(f"Is Rejected by Threshold? {distance > settings.RETRIEVAL_DISTANCE_THRESHOLD}")

if __name__ == "__main__":
    asyncio.run(main())
