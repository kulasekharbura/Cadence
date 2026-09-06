import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import async_session_maker
from app.models.document import Document
from app.models.chunk import DocumentChunk
import app.models.conversation
from sqlalchemy import select
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService

async def main():
    async with async_session_maker() as db:
        # 1. Find Resume_kulasekhar.pdf
        stmt = select(Document).where(Document.filename.ilike("%Resume%"))
        result = await db.execute(stmt)
        docs = result.scalars().all()
        
        if not docs:
            print("No document found matching %Resume%")
            return
            
        doc = docs[0]
        
        # 2. Show document_id, status, number of chunks
        stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index)
        result = await db.execute(stmt)
        chunks = result.scalars().all()
        
        print("=== 2. Document Info ===")
        print(f"document_id: {doc.id}")
        print(f"number of chunks: {len(chunks)}\n")
        
        degree_chunk = None
        cgpa_chunk = None
        
        print("=== 3. Chunks ===")
        for c in chunks:
            print(f"chunk_id: {c.id}, page_number: {c.page_number}, chunk_index: {c.chunk_index}")
            print(f"Content:\n{c.content}\n---")
            
            if "B.Tech in Information Technology" in c.content:
                degree_chunk = c
            if "8.92" in c.content:
                cgpa_chunk = c
                
        # 4 & 5. Identify which chunk contains the info
        print("=== 4 & 5. Information Location ===")
        print(f"'B.Tech in Information Technology' found in chunk_id: {degree_chunk.id if degree_chunk else 'NOT FOUND'}")
        print(f"'8.92' found in chunk_id: {cgpa_chunk.id if cgpa_chunk else 'NOT FOUND'}")
        
        # 6. Determine whether they are the same chunk
        print("=== 6. Same Chunk? ===")
        if degree_chunk and cgpa_chunk:
            print(f"Are they in the same chunk? {degree_chunk.id == cgpa_chunk.id}\n")
        else:
            print("Missing one or both pieces of information.\n")
            
        # 7. Trace RetrievalService behavior
        print("=== 7. Trace RetrievalService ===")
        retrieval_service = RetrievalService(db)
        query = "What degree is he pursuing?"
        print(f"Query: {query}")
        
        retrieved_chunks = await retrieval_service.retrieve_chunks(query, document_ids=[doc.id])
        
        print(f"Number of chunks retrieved by RetrievalService: {len(retrieved_chunks)}")
        
        degree_retrieved = False
        degree_distance = None
        for rc in retrieved_chunks:
            print(f"  Retrieved chunk: {rc.chunk_id}, distance: {rc.distance_score:.4f}")
            if degree_chunk and rc.chunk_id == degree_chunk.id:
                degree_retrieved = True
                degree_distance = rc.distance_score
                
        # Calculate EXACT distance just to be sure
        from app.services.embedding_service import EmbeddingService
        from app.schemas.chunk import DocumentChunk as ChunkSchema
        es = EmbeddingService()
        dummy_chunk = ChunkSchema(document_id=str(doc.id), page_number=1, chunk_index=0, content=query)
        embedded_result = await es.generate_embeddings_for_chunks([dummy_chunk])
        query_vector = embedded_result[0]["embedding"]
        
        print("\n=== 10. Actual Cosine Distance for ALL chunks ===")
        for c in chunks:
            distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
            stmt_dist = select(distance_col).where(DocumentChunk.id == c.id)
            dist_result = await db.execute(stmt_dist)
            actual_distance = dist_result.scalar()
            print(f"Chunk {c.chunk_index} distance: {actual_distance:.4f}")
            if degree_chunk and c.id == degree_chunk.id:
                degree_actual_dist = actual_distance
                
        print(f"\nActual distance for degree chunk specifically: {degree_actual_dist:.4f}")
        
        from app.core.config import settings
        threshold = settings.RETRIEVAL_DISTANCE_THRESHOLD
        print("\n=== 11. Compare with Threshold ===")
        print(f"Threshold: {threshold}")
        print(f"Is degree chunk rejected by distance? {degree_actual_dist > threshold}")
        
        print("\n=== 8. Conclusion on Retrieval State ===")
        if not degree_chunk:
            print("A. Not in document")
        elif degree_actual_dist > threshold:
            print("B. Retrieved but rejected by distance threshold")
        elif degree_retrieved:
            print("C. Retrieved and returned by RetrievalService (passed to context)")
        else:
            print("D. Returned by RetrievalService but missing from LLM context OR filtered by multi-doc limit")
                

        print("\n=== 11. Compare with Threshold ===")
        print(f"Threshold: {threshold}")
        print(f"Is chunk rejected by distance? {actual_distance > threshold}")
        
        print("\n=== 8. Conclusion on Retrieval State ===")
        if not degree_chunk:
            print("A. Not in document")
        elif actual_distance > threshold:
            print("B. Retrieved but rejected by distance threshold")
        elif degree_retrieved:
            print("C. Retrieved and returned by RetrievalService (passed to context)")
        else:
            print("D. Returned by RetrievalService but missing from LLM context OR filtered by multi-doc limit")
            
        # 12. Inspect exact context
        print("\n=== 12. LLM Context ===")
        llm_service = LLMService(db, retrieval_service)
        system_prompt = llm_service._construct_system_prompt(retrieved_chunks)
        print("System Prompt Context:")
        print(system_prompt)
        
        if degree_chunk and degree_chunk.content in system_prompt:
            print("\nCONCLUSION: The chunk IS in the LLM context. This is a GENERATION/PROMPT issue.")
        else:
            print("\nCONCLUSION: The chunk IS NOT in the LLM context. This is a RETRIEVAL issue.")

if __name__ == "__main__":
    asyncio.run(main())
