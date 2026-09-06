from typing import List, Optional
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.services.embedding_service import EmbeddingService
from app.schemas.retrieval import RetrievedChunk
from app.core.config import settings
from app.schemas.chunk import DocumentChunk as ChunkSchema

class RetrievalService:
    def __init__(self, db: AsyncSession, embedding_service: EmbeddingService = None):
        self.db = db
        self.embedding_service = embedding_service or EmbeddingService()
        
    async def retrieve_chunks(
        self, 
        query: str, 
        document_ids: Optional[List[uuid.UUID]] = None,
        top_k: Optional[int] = None,
        distance_threshold: Optional[float] = None
    ) -> List[RetrievedChunk]:
        """
        Embeds the query and searches the database for the most semantically similar chunks.
        Filters by document_ids if provided.
        Filters out chunks with distance > distance_threshold.
        """
        top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
        threshold = distance_threshold if distance_threshold is not None else settings.RETRIEVAL_DISTANCE_THRESHOLD
        
        # 1. Embed the natural language query
        # We reuse EmbeddingService. It expects DocumentChunk schemas. 
        # So we wrap the query in a dummy chunk.
        dummy_chunk = ChunkSchema(
            document_id="00000000-0000-0000-0000-000000000000", 
            page_number=0, 
            chunk_index=0, 
            content=query
        )
        
        # We only want the vector back
        try:
            embedded_result = await self.embedding_service.generate_embeddings_for_chunks([dummy_chunk])
            query_vector = embedded_result[0]["embedding"]
        except Exception as e:
            raise RuntimeError(f"Failed to embed retrieval query: {e}")
            
        # 2. Vector search using pgvector
        # Cosine distance: lower is better (more similar)
        distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
        
        base_stmt = select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            Document.filename,
            DocumentChunk.page_number,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            distance_col
        ).join(
            Document, DocumentChunk.document_id == Document.id
        )
        
        all_candidates = []
        
        # If document_ids is provided, query each document independently
        if document_ids is not None:
            if not document_ids:
                return []
            
            for doc_id in document_ids:
                stmt = base_stmt.where(DocumentChunk.document_id == doc_id)
                stmt = stmt.order_by(distance_col).limit(top_k)
                result = await self.db.execute(stmt)
                all_candidates.extend(result.all())
        else:
            # If no document_ids scope, search globally
            stmt = base_stmt.order_by(distance_col).limit(top_k)
            result = await self.db.execute(stmt)
            all_candidates.extend(result.all())
            
        # 3. Handle insufficient context (thresholding) and unique chunks
        retrieved_chunks = []
        seen_chunk_ids = set()
        
        for row in all_candidates:
            if row.distance > threshold:
                continue
            if row.id in seen_chunk_ids:
                continue
                
            seen_chunk_ids.add(row.id)
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    filename=row.filename,
                    page_number=row.page_number,
                    chunk_index=row.chunk_index,
                    content=row.content,
                    distance_score=row.distance
                )
            )
            
        # 4. Sort the combined candidates by cosine distance ascending
        retrieved_chunks.sort(key=lambda x: x.distance_score)
        
        # 5. Return the FINAL global top_k results
        return retrieved_chunks[:top_k]
