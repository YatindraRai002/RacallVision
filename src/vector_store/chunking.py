from typing import List
from src.utils.logger import get_logger
from src.config import config

logger = get_logger(__name__)


class TextChunker:
    """Text chunking with configurable size and overlap"""
    
    def __init__(self, chunk_size=None, chunk_overlap=None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        logger.info(f"Initialized chunker: size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Strategy:
        - Try to split on paragraph boundaries first
        - Fall back to sentence boundaries
        - Finally split on character boundaries if needed
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            # if we're at the end, just take the rest
            if end >= text_length:
                chunks.append(text[start:].strip())
                break
            
            chunk_text = text[start:end]
            
            # try to break on paragraph boundary
            last_para = chunk_text.rfind('\n\n')
            if last_para > self.chunk_size // 2:
                end = start + last_para
            else:
                # try to break on sentence boundary
                last_period = max(
                    chunk_text.rfind('. '),
                    chunk_text.rfind('! '),
                    chunk_text.rfind('? ')
                )
                if last_period > self.chunk_size // 2:
                    end = start + last_period + 1
                else:
                    # fall back to word boundary
                    last_space = chunk_text.rfind(' ')
                    if last_space > self.chunk_size // 2:
                        end = start + last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # move forward with overlap
            start = end - self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks from {text_length:,} characters")
        
        return chunks
    
    def chunk_with_metadata(self, text: str, source="") -> List[dict]:
        """Chunk text and include metadata for each chunk"""
        chunks = self.chunk_text(text)
        
        return [
            {
                'text': chunk,
                'chunk_id': i,
                'source': source,
                'char_count': len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]
