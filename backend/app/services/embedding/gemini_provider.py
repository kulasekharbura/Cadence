import os
from typing import List
from google import genai
from google.genai import types
from app.services.embedding.provider import EmbeddingProvider
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY is not configured.")
            self.client = None
        else:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            
        self.model = "gemini-embedding-001"
        # Always output exactly 1536 to match the pgvector column and OpenAI
        self._dimension = settings.GEMINI_EMBEDDING_DIMENSION

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            raise ValueError("Gemini API key is missing. Cannot generate embeddings.")
            
        if not texts:
            return []
            
        try:
            # We explicitly configure the output dimensionality to match 1536
            # Use async method if available, or just use sync if not natively supported in aio yet.
            # Currently genai sdk has async client: genai.Client().aio.models.embed_content
            # Let's use the async interface.
            response = await self.client.aio.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=self._dimension
                )
            )
            
            # Extract embeddings depending on SDK batching response format
            if isinstance(response.embeddings, list):
                embeddings = [emb.values for emb in response.embeddings]
            else:
                embeddings = [response.embeddings.values]
            
            if not embeddings:
                raise ValueError("Provider returned no embeddings")
                
            # Strict validation
            if len(embeddings[0]) != self._dimension:
                raise ValueError(f"Provider returned dimension {len(embeddings[0])}, expected {self._dimension}")
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to fetch embeddings from Gemini: {e}")
            raise RuntimeError(f"Gemini Embedding API error: {e}")

    @property
    def dimension(self) -> int:
        return self._dimension
