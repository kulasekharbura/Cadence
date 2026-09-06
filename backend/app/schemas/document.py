from pydantic import BaseModel, ConfigDict
import uuid
from datetime import datetime
from typing import Optional
from app.models.document import ProcessingStatus

class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    file_size: int
    page_count: Optional[int] = None
    processing_status: ProcessingStatus
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
