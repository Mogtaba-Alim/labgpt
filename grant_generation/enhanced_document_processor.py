#!/usr/bin/env python3
"""
enhanced_document_processor.py

Advanced document processing pipeline for grant generation that handles:
- Large documents with intelligent chunking
- OCR for scanned PDFs
- Section-aware text extraction
- Document summarization
- Context-aware information extraction
"""

import os
import io
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import re
from dataclasses import dataclass
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

@dataclass
class ProcessedDocument:
    """Container for processed document with metadata"""
    filename: str
    text: str
    sections: Dict[str, str]  # section_name -> content
    summary: str
    document_type: str  # 'grant', 'paper', 'report', etc.
    word_count: int
    is_scanned: bool

class EnhancedDocumentProcessor:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name, device='cuda')
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
        
        # Grant section keywords for intelligent extraction
        self.section_keywords = {
            "Background": ["background", "literature", "previous work", "state of art", "context"],
            "Objectives": ["objective", "goal", "aim", "purpose", "target"],
            "Specific Aims": ["specific aim", "research question", "hypothesis", "investigation"],
            "Methods": ["method", "methodology", "approach", "technique", "protocol", "procedure"],
            "Preliminary work": ["preliminary", "pilot", "previous result", "prior work", "baseline"],
            "Impact/Relevance": ["impact", "relevance", "significance", "importance", "benefit"],
            "Feasibility, Risks and Mitigation Strategies": ["feasibility", "risk", "challenge", "mitigation", "limitation"],
            "Project Outcomes and Future Directions": ["outcome", "deliverable", "future", "next step", "continuation"],
            "Research Data Management and Open Science": ["data management", "open science", "sharing", "repository"],
            "Expertise, Experience, and Resources": ["experience", "expertise", "qualification", "resource", "facility"],
            "Summary of Progress of Principal Investigator": ["progress", "achievement", "accomplishment", "track record"],
            "Lay Abstract": ["abstract", "summary", "overview"],
            "Lay Summary": ["lay summary", "public summary", "general audience"]
        }

    def is_document_scanned(self, pdf_path: str) -> bool:
        """Detect if PDF contains scanned images vs searchable text"""
        try:
            doc = fitz.open(pdf_path)
            total_chars = 0
            total_pages = len(doc)
            
            for page in doc:
                text = page.get_text()
                total_chars += len(text.strip())
            
            # If average characters per page is very low, likely scanned
            avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0
            return avg_chars_per_page < 100
        except Exception as e:
            logging.error(f"Error checking if document is scanned: {e}")
            return False

    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR for scanned documents"""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Try regular text extraction first
                text = page.get_text()
                if len(text.strip()) > 50:  # Sufficient text found
                    text_parts.append(text)
                else:
                    # Use OCR for this page
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    ocr_text = pytesseract.image_to_string(img)
                    text_parts.append(ocr_text)
            
            return "\n".join(text_parts)
        except Exception as e:
            logging.error(f"Error extracting text with OCR: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, bool]:
        """Extract text from PDF, handling both regular and scanned documents"""
        is_scanned = self.is_document_scanned(pdf_path)
        
        if is_scanned:
            text = self.extract_text_with_ocr(pdf_path)
        else:
            try:
                doc = fitz.open(pdf_path)
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                text = "\n".join(text_parts)
            except Exception as e:
                logging.error(f"Error extracting text from PDF: {e}")
                text = ""
        
        return text, is_scanned

    def classify_document_type(self, text: str) -> str:
        """Classify document type based on content"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ["grant", "proposal", "funding", "nih", "nsf"]):
            return "grant"
        elif any(keyword in text_lower for keyword in ["abstract", "introduction", "methodology", "results", "conclusion"]):
            return "paper"
        elif any(keyword in text_lower for keyword in ["report", "progress", "quarterly", "annual"]):
            return "report"
        else:
            return "document"

    def intelligent_chunking(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Create intelligent chunks preserving sentence and paragraph boundaries"""
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                if overlap > 0:
                    sentences = current_chunk.split('. ')
                    overlap_text = '. '.join(sentences[-2:]) if len(sentences) > 1 else ""
                    current_chunk = overlap_text + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def extract_section_relevant_content(self, text: str, section_name: str) -> str:
        """Extract content relevant to a specific grant section"""
        if section_name not in self.section_keywords:
            return ""
        
        keywords = self.section_keywords[section_name]
        text_lower = text.lower()
        relevant_chunks = []
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            # Check if paragraph contains relevant keywords
            if any(keyword in paragraph_lower for keyword in keywords):
                relevant_chunks.append(paragraph)
        
        return "\n\n".join(relevant_chunks)

    def summarize_document(self, text: str, max_length: int = 500) -> str:
        """Generate a summary of the document"""
        try:
            # Split into chunks if too long for summarizer
            max_input_length = 1000
            if len(text) <= max_input_length:
                summary = self.summarizer(text, max_length=max_length, min_length=50)[0]['summary_text']
            else:
                # Chunk and summarize each part, then combine
                chunks = self.intelligent_chunking(text, max_input_length, 100)
                chunk_summaries = []
                for chunk in chunks[:5]:  # Limit to first 5 chunks
                    try:
                        summary = self.summarizer(chunk, max_length=100, min_length=20)[0]['summary_text']
                        chunk_summaries.append(summary)
                    except:
                        continue
                
                # Combine and summarize again
                combined_summary = " ".join(chunk_summaries)
                if len(combined_summary) > max_input_length:
                    combined_summary = combined_summary[:max_input_length]
                
                summary = self.summarizer(combined_summary, max_length=max_length, min_length=50)[0]['summary_text']
            
            return summary
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            # Fallback: return first few sentences
            sentences = text.split('. ')
            return '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else text[:500]

    def process_document(self, file_path: str) -> ProcessedDocument:
        """Process a single document comprehensively"""
        filename = Path(file_path).name
        
        # Extract text
        if file_path.lower().endswith('.pdf'):
            text, is_scanned = self.extract_text_from_pdf(file_path)
        else:
            # Handle text files
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                is_scanned = False
            except Exception as e:
                logging.error(f"Error reading text file {file_path}: {e}")
                text = ""
                is_scanned = False
        
        if not text.strip():
            logging.warning(f"No text extracted from {filename}")
            return None
        
        # Classify document type
        doc_type = self.classify_document_type(text)
        
        # Extract section-relevant content
        sections = {}
        for section_name in self.section_keywords.keys():
            sections[section_name] = self.extract_section_relevant_content(text, section_name)
        
        # Generate summary
        summary = self.summarize_document(text)
        
        return ProcessedDocument(
            filename=filename,
            text=text,
            sections=sections,
            summary=summary,
            document_type=doc_type,
            word_count=len(text.split()),
            is_scanned=is_scanned
        )

    def process_multiple_documents(self, file_paths: List[str]) -> List[ProcessedDocument]:
        """Process multiple documents"""
        processed_docs = []
        
        for file_path in file_paths:
            try:
                doc = self.process_document(file_path)
                if doc:
                    processed_docs.append(doc)
                    logging.info(f"Processed {doc.filename}: {doc.word_count} words, type: {doc.document_type}")
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                continue
        
        return processed_docs

    def create_section_context(self, processed_docs: List[ProcessedDocument], 
                              section_name: str, grant_overview: str, 
                              max_context_length: int = 4000) -> str:
        """Create optimized context for a specific section"""
        context_parts = []
        
        # Add grant overview
        context_parts.append(f"Grant Overview:\n{grant_overview}\n")
        
        # Add relevant sections from documents
        for doc in processed_docs:
            if section_name in doc.sections and doc.sections[section_name].strip():
                context_parts.append(f"From {doc.filename} ({doc.document_type}):\n{doc.sections[section_name]}\n")
        
        # Add document summaries if space allows
        for doc in processed_docs:
            summary_text = f"Summary of {doc.filename}: {doc.summary}\n"
            if sum(len(part) for part in context_parts) + len(summary_text) < max_context_length:
                context_parts.append(summary_text)
        
        # Truncate if necessary
        full_context = "\n".join(context_parts)
        if len(full_context) > max_context_length:
            full_context = full_context[:max_context_length] + "...[truncated]"
        
        return full_context 