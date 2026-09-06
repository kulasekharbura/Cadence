import os
import shutil
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db, async_session_maker
from app.models.document import Document
from app.schemas.document import DocumentResponse
from app.services.document_service import DocumentService

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024 # 10MB

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    # Read file to check size
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the 10MB limit.")
        
    # Create unique filepath
    file_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    with open(filepath, "wb") as f:
        f.write(contents)
        
    # Create DB Record
    new_doc = Document(
        id=file_id,
        filename=file.filename,
        file_path=filepath,
        file_size=file_size
    )
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)
        
    # Add background task for PDF processing
    background_tasks.add_task(
        DocumentService.process_document_background,
        async_session_maker,
        new_doc.id,
        filepath
    )
    
    return new_doc

@router.get("", response_model=List[DocumentResponse])
async def list_documents(db: AsyncSession = Depends(get_db)):
    docs = await DocumentService.get_all_documents(db)
    return docs

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    await DocumentService.delete_document(db, document_id)
