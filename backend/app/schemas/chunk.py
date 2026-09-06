from pydantic import BaseModel

class DocumentChunk(BaseModel):
    document_id: str
    page_number: int
    chunk_index: int
    content: str

import uuid
class ChunkResponse(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_index: int
    content: str
