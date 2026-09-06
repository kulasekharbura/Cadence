from pydantic import BaseModel
from typing import List, Optional
import uuid

class RetrievedChunk(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_index: int
    content: str
    # Cosine distance (0.0 means identical, 2.0 means opposite direction)
    # We use distance (lower is better) rather than similarity score here,
    # as pgvector natively uses `<=>` distance operator.
    distance_score: float

class RetrievalResult(BaseModel):
    query: str
    chunks: List[RetrievedChunk]
