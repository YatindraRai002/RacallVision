"""
RecallVision - File Utilities
Common file operations with error handling
"""

import os
import hashlib
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {dir_path}")
    return dir_path


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Create a safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Safe filename string
    """
    import re
    safe_name = re.sub(r'[^\w\s-]', '', filename)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    safe_name = safe_name.strip('_').lower()
    
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    
    return safe_name


def file_exists(path: str | Path) -> bool:
    """
    Check if a file exists.
    
    Args:
        path: File path
        
    Returns:
        True if file exists, False otherwise
    """
    return Path(path).is_file()


def get_file_hash(path: str | Path, algorithm: str = "md5") -> str:
    """
    Calculate hash of a file.
    
    Args:
        path: File path
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest of file hash
    """
    hash_func = hashlib.new(algorithm)
    
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def read_text_file(path: str | Path, encoding: str = 'utf-8') -> Optional[str]:
    """
    Read text file with error handling.
    
    Args:
        path: File path
        encoding: Text encoding
        
    Returns:
        File contents or None if error
    """
    try:
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
        logger.debug(f"Read {len(content)} characters from {path}")
        return content
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        return None


def write_text_file(
    path: str | Path,
    content: str,
    encoding: str = 'utf-8',
    create_dirs: bool = True
) -> bool:
    """
    Write text file with error handling.
    
    Args:
        path: File path
        content: Text content to write
        encoding: Text encoding
        create_dirs: Create parent directories if they don't exist
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(path)
        
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        logger.info(f"Wrote {len(content)} characters to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing file {path}: {e}")
        return False


def get_file_size(path: str | Path) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: File path
        
    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
