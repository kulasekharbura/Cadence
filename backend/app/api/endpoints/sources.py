from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import uuid

from app.db.session import get_db
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.schemas.chunk import ChunkResponse

router = APIRouter()

@router.get("/{chunk_id}", response_model=ChunkResponse)
async def get_source_chunk(
    chunk_id: uuid.UUID = Path(...),
    db: AsyncSession = Depends(get_db)
):
    stmt = select(DocumentChunk).where(DocumentChunk.id == chunk_id)
    result = await db.execute(stmt)
    chunk = result.scalar_one_or_none()
    
    if not chunk:
        raise HTTPException(status_code=404, detail="Source chunk not found")
        
    stmt_doc = select(Document).where(Document.id == chunk.document_id)
    result_doc = await db.execute(stmt_doc)
    doc = result_doc.scalar_one_or_none()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document for chunk not found")
        
    return ChunkResponse(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        filename=doc.filename,
        page_number=chunk.page_number,
        chunk_index=chunk.chunk_index,
        content=chunk.content
    )
