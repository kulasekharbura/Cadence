try:
    import pymupdf as fitz
except ImportError:
    import fitz  # PyMuPDF fallback
import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Minimum characters required to consider a page's text as successfully extracted normally
MIN_MEANINGFUL_TEXT_LENGTH = 50

class PDFService:
    @staticmethod
    def get_page_count(filepath: str) -> int:
        """Get the total number of pages in the PDF."""
        try:
            with fitz.open(filepath) as doc:
                return doc.page_count
        except Exception as e:
            raise ValueError(f"Failed to open PDF: {str(e)}")

    @staticmethod
    def extract_text(filepath: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF page by page.
        Uses normal extraction first, and falls back to OCR if the extracted text is insufficient.
        Returns a list of dictionaries containing page_number and text.
        """
        extracted_pages = []
        try:
            with fitz.open(filepath) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    
                    # Normal extraction
                    text = page.get_text("text").strip()
                    
                    # OCR Fallback for scanned/image-only pages
                    if len(text) < MIN_MEANINGFUL_TEXT_LENGTH:
                        logger.info(f"Page {page_num + 1} has insufficient text ({len(text)} chars). Falling back to OCR.")
                        try:
                            # Use PyMuPDF's Tesseract-backed OCR
                            textpage = page.get_textpage_ocr(language="eng", dpi=300)
                            text = page.get_text("text", textpage=textpage).strip()
                        except Exception as ocr_e:
                            logger.error(f"OCR failed for page {page_num + 1}: {ocr_e}")
                            # Keep whatever text we had (even if empty) or re-raise if strict failure is needed.
                            # We proceed with existing text to avoid failing the entire document just for one page.
                    
                    # We store 1-indexed page numbers for user readability
                    extracted_pages.append({
                        "page_number": page_num + 1,
                        "text": text
                    })
        except Exception as e:
            raise ValueError(f"Failed to process PDF: {str(e)}")
            
        return extracted_pages
