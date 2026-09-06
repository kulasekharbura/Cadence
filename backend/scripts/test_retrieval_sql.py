import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import async_session_maker
from sqlalchemy import select
from app.models.chunk import DocumentChunk
from app.models.document import Document
import uuid

async def main():
    async with async_session_maker() as db:
        # Mock vector
        query_vector = [0.1] * 1536
        distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
        
        stmt = select(
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
        
        doc_ids = [uuid.uuid4(), uuid.uuid4()]
        stmt = stmt.where(DocumentChunk.document_id.in_(doc_ids))
        stmt = stmt.order_by(distance_col).limit(5)
        
        from sqlalchemy.dialects import postgresql
        compiled = stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
        print("COMPILED QUERY:")
        print(compiled)

if __name__ == "__main__":
    asyncio.run(main())
