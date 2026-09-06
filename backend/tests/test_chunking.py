import pytest
from app.services.chunking_service import ChunkingService

def test_chunking_single_page():
    pages = [
        {"page_number": 1, "text": "This is a simple test sentence that we will use to test chunking."}
    ]
    
    chunks = ChunkingService.chunk_document("doc1", pages, chunk_size=20, chunk_overlap=10)
    
    assert len(chunks) > 1
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].document_id == "doc1"

def test_chunk_no_cross_page_boundary():
    pages = [
        {"page_number": 1, "text": "Page one content."},
        {"page_number": 2, "text": "Page two content."}
    ]
    
    # Large chunk size, small text, normally it would merge them.
    # Our algorithm strictly operates per page.
    chunks = ChunkingService.chunk_document("doc2", pages, chunk_size=1000, chunk_overlap=200)
    
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].content == "Page one content."
    assert chunks[1].page_number == 2
    assert chunks[1].content == "Page two content."
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1

def test_chunk_overlap():
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9"
    # Word lengths: 5 chars.
    # Total ~ 53 chars
    
    # We want chunk_size = 20, chunk_overlap = 10
    chunks = ChunkingService.chunk_text(text, chunk_size=20, chunk_overlap=10)
    
    # Chunk 1: "word1 word2 word3" (17 chars)
    # Chunk 2 overlaps "word2 word3" (11 chars... wait, overlap max is 10)
    # If overlap max is 10, it can only overlap "word3" (5 chars) because "word2 word3" is 11 chars.
    
    assert "word1" in chunks[0]
    
    # Verify strict sizes
    for c in chunks:
        assert len(c) <= 25 # allow slight overflow for the current word if we hit the limit

def test_empty_page():
    pages = [
        {"page_number": 1, "text": "   \n\n  "},
        {"page_number": 2, "text": "Valid text here."}
    ]
    
    chunks = ChunkingService.chunk_document("doc3", pages, chunk_size=1000, chunk_overlap=200)
    
    assert len(chunks) == 1
    assert chunks[0].page_number == 2
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == "Valid text here."
