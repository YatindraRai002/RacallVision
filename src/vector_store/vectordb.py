"""
RecallVision - Vector Database Module
FAISS-based vector database for similarity search
"""

import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.file_utils import ensure_directory
from src.config import VECTOR_DB_DIR

logger = get_logger(__name__)


class VectorDatabase:
    """FAISS-based vector database"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize vector database.
        
        Args:
            db_path: Path to database directory (uses config default if None)
        """
        self.db_path = Path(db_path) if db_path else VECTOR_DB_DIR
        self.index_path = self.db_path / "index.faiss"
        self.chunks_path = self.db_path / "chunks.pkl"
        self.metadata_path = self.db_path / "metadata.pkl"
        
        self.index: Optional[faiss.Index] = None
        self.chunks: List[str] = []
        self.metadata: List[dict] = []
    
    def create(self, embeddings: np.ndarray, chunks: List[str], metadata: List[dict] = None):
        """
        Create new vector database from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings (shape: [num_chunks, embedding_dim])
            chunks: List of text chunks
            metadata: Optional list of metadata dicts for each chunk
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        logger.info(f"Creating vector database with {len(chunks)} chunks")
        
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        self.index.add(embeddings)
        
        self.chunks = chunks
        self.metadata = metadata or [{}] * len(chunks)
        
        logger.info(f"Vector database created with {self.index.ntotal} vectors")
    
    def save(self):
        """Save vector database to disk"""
        if self.index is None:
            raise ValueError("No index to save. Create or load an index first.")
        
        logger.info(f"Saving vector database to {self.db_path}")
        
        ensure_directory(self.db_path)
        
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info("Vector database saved successfully")
    
    def load(self):
        """Load vector database from disk"""
        logger.info(f"Loading vector database from {self.db_path}")
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")
        
        self.index = faiss.read_index(str(self.index_path))
        
        with open(self.chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        if self.metadata_path.exists():
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = [{}] * len(self.chunks)
        
        logger.info(f"Loaded vector database with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 2) -> Tuple[List[str], List[float], List[dict]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (chunks, distances, metadata)
        """
        if self.index is None:
            raise ValueError("No index loaded. Load or create an index first.")
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        result_chunks = [self.chunks[idx] for idx in indices[0]]
        result_distances = distances[0].tolist()
        result_metadata = [self.metadata[idx] for idx in indices[0]]
        
        logger.debug(f"Retrieved {len(result_chunks)} chunks")
        
        return result_chunks, result_distances, result_metadata
    
    @property
    def size(self) -> int:
        """Get number of vectors in database"""
        return self.index.ntotal if self.index else 0
