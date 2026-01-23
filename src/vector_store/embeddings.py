"""
RecallVision - Embeddings Module
Wrapper for embedding model operations
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from src.utils.logger import get_logger
from src.config import config

logger = get_logger(__name__)


class EmbeddingModel:
    """Wrapper for sentence transformer embedding model"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model (uses config default if None)
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} texts...")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings.astype('float32')
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding.
        
        Args:
            text: Text string to encode
            
        Returns:
            Numpy array of embedding (shape: [embedding_dim])
        """
        return self.encode([text])[0]
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        return self.model.get_sentence_embedding_dimension()
