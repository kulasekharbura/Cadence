from typing import List, Dict, Any
from app.schemas.chunk import DocumentChunk
from app.services.embedding.provider import EmbeddingProvider
from app.services.embedding.gemini_provider import GeminiEmbeddingProvider

class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider = None):
        self.provider = provider or GeminiEmbeddingProvider()
        
    async def generate_embeddings_for_chunks(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Takes a list of DocumentChunks and generates embeddings for their content.
        Returns a list of dictionaries containing the original chunk and its embedding vector.
        """
        if not chunks:
            return []
            
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings via provider abstraction
        embeddings = await self.provider.get_embeddings(texts)
        
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding provider returned {len(embeddings)} embeddings "
                f"but {len(chunks)} chunks were provided."
            )
            
        result = []
        for chunk, emb in zip(chunks, embeddings):
            result.append({
                "chunk": chunk,
                "embedding": emb
            })
            
        return result
