from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Document Q&A"
    API_V1_STR: str = "/api/v1"
    
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@db:5432/document_qa"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_DISTANCE_THRESHOLD: float = 0.6
    MAX_HISTORY_MESSAGES: int = 5

    # AI Provider Configuration
    GEMINI_API_KEY: str | None = None
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    GEMINI_EMBEDDING_DIMENSION: int = 1536
    GEMINI_LLM_MODEL: str = "gemini-3.5-flash"
    GEMINI_FALLBACK_MODELS: str = "gemini-3.5-flash-lite,gemini-3.7-flash"
    
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

settings = Settings()
