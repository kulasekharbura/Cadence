import os
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.document import Document, ProcessingStatus
from app.models.chunk import DocumentChunk
from app.services.pdf_service import PDFService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from sqlalchemy import select
from typing import List
from fastapi import HTTPException

class DocumentService:
    
    @staticmethod
    async def get_all_documents(db: AsyncSession) -> List[Document]:
        result = await db.execute(select(Document).order_by(Document.created_at.desc()))
        return result.scalars().all()

    @staticmethod
    async def delete_document(db: AsyncSession, document_id: str) -> None:
        doc = await db.get(Document, document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        file_path = doc.file_path
        
        await db.delete(doc)
        await db.commit()
        
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete physical file {file_path}: {e}")

    @staticmethod
    async def process_document_background(db_session_factory, document_id: str, filepath: str):
        """
        Background task to process the PDF:
        1. Read page count
        2. Extract text page by page
        3. Update status to READY (or FAILED)
        """
        # We need a new session for the background task
        async with db_session_factory() as db:
            doc = await db.get(Document, document_id)
            if not doc:
                return

            try:
                # Update status to PROCESSING
                doc.processing_status = ProcessingStatus.PROCESSING
                await db.commit()

                # Extract data using PyMuPDF
                # Run CPU-bound extraction in a thread to avoid blocking the event loop
                page_count = await asyncio.to_thread(PDFService.get_page_count, filepath)
                doc.page_count = page_count
                
                # Extract text using PyMuPDF
                extracted_pages = await asyncio.to_thread(PDFService.extract_text, filepath)
                
                # Phase 3: Chunking
                _chunks = await asyncio.to_thread(
                    ChunkingService.chunk_document,
                    str(document_id),
                    extracted_pages,
                    chunk_size=1000,
                    chunk_overlap=200
                )
                
                # Phase 4: Embeddings
                embedding_service = EmbeddingService()
                embedded_chunks = await embedding_service.generate_embeddings_for_chunks(_chunks)
                
                # Phase 5: Persist Chunks
                for data in embedded_chunks:
                    chunk = data["chunk"]
                    embedding_vector = data["embedding"]
                    
                    db_chunk = DocumentChunk(
                        document_id=doc.id,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        embedding=embedding_vector
                    )
                    db.add(db_chunk)

                # Update status to READY
                doc.processing_status = ProcessingStatus.READY
                await db.commit()

            except Exception as e:
                doc.processing_status = ProcessingStatus.FAILED
                doc.error_message = str(e)
                await db.commit()
            # We do NOT delete the PDF file here. It remains the durable source for Phase 3.
