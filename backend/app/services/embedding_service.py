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
        
        # Generate embeddings via provider abstraction in batches to respect API limits (e.g. Gemini 100 max)
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = await self.provider.get_embeddings(batch_texts)
            embeddings.extend(batch_embeddings)
        
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
