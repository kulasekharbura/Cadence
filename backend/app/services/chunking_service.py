from typing import List, Dict, Any
from app.schemas.chunk import DocumentChunk

class ChunkingService:
    @staticmethod
    def chunk_document(
        document_id: str, 
        pages: List[Dict[str, Any]], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[DocumentChunk]:
        """
        Takes extracted pages and chunks them strictly within page boundaries.
        No chunk will ever span multiple pages.
        """
        document_chunks = []
        global_chunk_index = 0
        
        for page in pages:
            page_num = page.get("page_number", 1)
            text = page.get("text", "")
            
            chunks = ChunkingService.chunk_text(text, chunk_size, chunk_overlap)
            
            for content in chunks:
                document_chunks.append(DocumentChunk(
                    document_id=str(document_id),
                    page_number=page_num,
                    chunk_index=global_chunk_index,
                    content=content
                ))
                global_chunk_index += 1
                
        return document_chunks
        
    @staticmethod
    def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Greedy word-based chunking sliding window algorithm.
        Splits text into chunks of at most `chunk_size` characters.
        Overlap is at most `chunk_overlap` characters.
        Preserves word boundaries.
        """
        if not text.strip():
            return []
            
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            # +1 for the space separator if the chunk is not empty
            word_len = len(word) + (1 if current_chunk else 0) 
            
            # If a single word is larger than the chunk_size (edge case),
            # we must put it in its own chunk or split it. 
            # For simplicity, we just allow the chunk to exceed the size for that single massive word.
            if current_length + word_len > chunk_size and current_chunk:
                # Save the current chunk
                chunks.append(" ".join(current_chunk))
                
                # Calculate the overlap from the end of the current_chunk
                overlap_chunk = []
                overlap_length = 0
                for w in reversed(current_chunk):
                    w_len = len(w) + (1 if overlap_chunk else 0)
                    if overlap_length + w_len <= chunk_overlap:
                        overlap_chunk.insert(0, w)
                        overlap_length += w_len
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_chunk
                current_length = overlap_length
            
            current_chunk.append(word)
            current_length += len(word) + (1 if len(current_chunk) > 1 else 0)
            i += 1
            
        # Add the final chunk if anything remains
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
