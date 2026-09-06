import pytest
import fitz
import os
from app.services.pdf_service import PDFService

def create_multi_page_pdf(path: str):
    doc = fitz.open()
    
    # Page 1
    p1 = doc.new_page()
    p1.insert_text((10, 10), "Apple Banana")
    
    # Page 2
    p2 = doc.new_page()
    p2.insert_text((10, 10), "Cherry Date")
    
    # Page 3
    p3 = doc.new_page()
    p3.insert_text((10, 10), "Elderberry Fig")
    
    doc.save(path)
    doc.close()

def test_page_aware_extraction(tmp_path):
    pdf_path = tmp_path / "multi.pdf"
    create_multi_page_pdf(str(pdf_path))
    
    pages = PDFService.extract_text(str(pdf_path))
    
    assert len(pages) == 3
    
    # Verify page numbers (1-indexed) and content
    assert pages[0]["page_number"] == 1
    assert "Apple Banana" in pages[0]["text"]
    
    assert pages[1]["page_number"] == 2
    assert "Cherry Date" in pages[1]["text"]
    
    assert pages[2]["page_number"] == 3
    assert "Elderberry Fig" in pages[2]["text"]
