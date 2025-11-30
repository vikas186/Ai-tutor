"""
Document chunking utility for very large documents
Splits documents into manageable chunks for processing
"""
from typing import List, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Split large documents into chunks for processing"""
    
    def __init__(self, chunk_size: int = 30000, overlap: int = 1000):
        """
        Initialize chunker
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks (to avoid splitting questions)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks (memory-efficient version)
        Processes incrementally to avoid MemoryError
        
        Returns:
            List of tuples: (chunk_text, start_index, end_index)
        """
        text_len = len(text)
        
        if text_len <= self.chunk_size:
            return [(text, 0, text_len)]
        
        chunks = []
        start = 0
        
        # Limit total chunks to prevent memory issues
        MAX_CHUNKS = 20
        chunk_count = 0
        
        # For very large documents, use smaller chunks
        effective_chunk_size = self.chunk_size
        if text_len > 100000:  # For documents > 100k chars
            effective_chunk_size = min(self.chunk_size, 10000)
            logger.info(f"Large document detected ({text_len} chars), using smaller chunks ({effective_chunk_size})")
        
        while start < text_len and chunk_count < MAX_CHUNKS:
            end = min(start + effective_chunk_size, text_len)
            
            # If not at the end, try to break at a natural boundary
            # PRIORITIZE breaking at question boundaries to avoid splitting questions
            if end < text_len:
                # Search in a larger window to find question boundaries
                search_start = max(start, end - 3000)  # Increased search window
                
                # Look for question number patterns (higher priority)
                question_patterns = [
                    (r'\n\s*\d+\s+[A-Z]', 'numbered question start'),  # "1 Fill", "2 Write"
                    (r'\n\s*\d+[\.\)]\s+', 'numbered question'),  # "1. ", "2) "
                    (r'\n\s*\d+[\(]', 'numbered with letter'),  # "1(a)", "2(b)"
                ]
                
                question_breaks = []
                for pattern, desc in question_patterns:
                    matches = list(re.finditer(pattern, text[search_start:end], re.IGNORECASE))
                    if matches:
                        # Get the last match position (closest to end)
                        last_match = matches[-1]
                        pos = search_start + last_match.start()
                        if pos > start:
                            question_breaks.append((pos, desc))
                
                # Also look for standard boundaries
                break_points = [
                    (text.rfind('\n\n', search_start, end), 'paragraph'),
                    (text.rfind('\nTest ', search_start, end), 'test boundary'),
                    (text.rfind('\nSection ', search_start, end), 'section boundary'),
                    (text.rfind('\nQuestion ', search_start, end), 'question marker'),
                    (text.rfind('\n', search_start, end), 'newline'),
                ]
                
                # Prioritize question breaks, then other boundaries
                all_breaks = [(bp, desc) for bp, desc in break_points if bp > start]
                all_breaks.extend(question_breaks)
                
                if all_breaks:
                    # Sort by position (closest to end first)
                    all_breaks.sort(key=lambda x: x[0], reverse=True)
                    # Prefer question breaks
                    question_break = next((bp for bp, desc in all_breaks if 'question' in desc.lower()), None)
                    if question_break:
                        end = min(question_break, end)
                    else:
                        # Use best available break
                        best_break = all_breaks[0][0]
                        end = min(best_break + 1, end)  # Include the newline
            
            # Extract chunk (only the slice we need)
            chunk_text = text[start:end]
            
            # Safety check: ensure chunk is reasonable size
            if len(chunk_text) > effective_chunk_size * 2:
                # Chunk too large, force split
                end = start + effective_chunk_size
                chunk_text = text[start:end]
            
            # Add chunk
            chunks.append((chunk_text, start, end))
            chunk_count += 1
            
            # Move start forward with overlap
            if end < text_len:
                start = max(start + 1, end - self.overlap)
            else:
                break
        
        if chunk_count >= MAX_CHUNKS:
            logger.warning(f"⚠️  Reached maximum chunk limit ({MAX_CHUNKS}), some text may be skipped")
        
        logger.info(f"Split document into {len(chunks)} chunks (memory-efficient, {text_len} chars total)")
        return chunks
    
    def find_question_boundaries(self, text: str) -> List[int]:
        """Find natural question boundaries in text"""
        boundaries = [0]  # Start
        
        # Look for question patterns
        import re
        patterns = [
            r'\n\s*\d+[\.\)]\s+',  # Numbered questions: "1. " or "1) "
            r'\n\s*[A-Z][a-z]+:\s*\n',  # Section headers: "Section A:"
            r'\n\s*Test\s+\d+',  # Test boundaries
            r'\n\s*Question\s+\d+',  # Question markers
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                boundaries.append(match.start())
        
        boundaries.append(len(text))  # End
        boundaries = sorted(set(boundaries))
        
        return boundaries


