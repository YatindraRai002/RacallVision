"""
RecallVision - Retriever Module
High-level interface for vector similarity search
"""

from typing import List, Tuple
import numpy as np

from src.vector_store.vectordb import VectorDatabase
from src.vector_store.embeddings import EmbeddingModel
from src.utils.logger import get_logger
from src.config import config

logger = get_logger(__name__)


class Retriever:
    """High-level retriever for semantic search"""
    
    def __init__(self, db_path: str = None, embedding_model: str = None):
        """
        Initialize retriever.
        
        Args:
            db_path: Path to vector database
            embedding_model: Name of embedding model (uses config default if None)
        """
        self.vector_db = VectorDatabase(db_path)
        self.embedding_model = EmbeddingModel(embedding_model)
        
        try:
            self.vector_db.load()
            logger.info(f"Retriever initialized with {self.vector_db.size} documents")
        except FileNotFoundError as e:
            logger.error(f"Vector database not found: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        """
        Retrieve top-k most similar chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return (uses config default if None)
            
        Returns:
            List of retrieved text chunks
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        logger.info(f"Retrieving top-{top_k} chunks for query: {query[:100]}...")
        
        query_embedding = self.embedding_model.encode_single(query)
        
        chunks, distances, metadata = self.vector_db.search(query_embedding, top_k)
        
        for i, (chunk, dist) in enumerate(zip(chunks, distances), 1):
            logger.debug(f"Result {i} (distance={dist:.4f}): {chunk[:100]}...")
        
        return chunks
    
    def retrieve_with_scores(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Retrieve top-k chunks with similarity scores.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk, distance) tuples
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        query_embedding = self.embedding_model.encode_single(query)
        
        chunks, distances, _ = self.vector_db.search(query_embedding, top_k)
        
        return list(zip(chunks, distances))
    
    def retrieve_with_metadata(self, query: str, top_k: int = None) -> List[dict]:
        """
        Retrieve top-k chunks with full metadata.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'text', 'distance', and metadata
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        query_embedding = self.embedding_model.encode_single(query)
        
        chunks, distances, metadata = self.vector_db.search(query_embedding, top_k)
        
        results = []
        for chunk, dist, meta in zip(chunks, distances, metadata):
            result = {
                'text': chunk,
                'distance': float(dist),
                **meta
            }
            results.append(result)
        
        return results
