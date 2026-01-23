"""
RecallVision - Text Cleaning Utilities
Text processing and cleaning functions
"""

import re
from typing import List
from src.utils.logger import get_logger

logger = get_logger(__name__)


def remove_citations(text: str) -> str:
    """
    Remove citation brackets from text.
    
    Args:
        text: Input text with citations
        
    Returns:
        Text with citations removed
    """
    text = re.sub(r'\[\d+\]', '', text)
    
    text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    text = re.sub(r' +', ' ', text)
    
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    text = text.strip()
    
    return text


def remove_special_characters(text: str, keep_chars: str = '') -> str:
    """
    Remove special characters from text.
    
    Args:
        text: Input text
        keep_chars: Additional characters to keep
        
    Returns:
        Text with special characters removed
    """
    pattern = f'[^\\w\\s{re.escape(keep_chars)}]'
    return re.sub(pattern, '', text)


def clean_wikipedia_text(text: str) -> str:
    """
    Clean Wikipedia article text.
    
    Args:
        text: Raw Wikipedia text
        
    Returns:
        Cleaned text
    """
    text = remove_citations(text)
    
    text = normalize_whitespace(text)
    
    logger.debug(f"Cleaned text: {len(text)} characters")
    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def truncate_text(text: str, max_length: int, suffix: str = '...') -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Word count
    """
    return len(text.split())


def extract_paragraphs(text: str, min_length: int = 20) -> List[str]:
    """
    Extract paragraphs from text.
    
    Args:
        text: Input text
        min_length: Minimum paragraph length
        
    Returns:
        List of paragraphs
    """
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]
    return paragraphs
