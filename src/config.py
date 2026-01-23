import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    """Centralized configuration for RecallVision"""
    
    # API keys
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    
    # ASR settings
    ASR_ENDPOINT = os.getenv("ASR_ENDPOINT", "http://localhost:8001/transcribe")
    ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "base")
    ASR_MODEL_TYPE = os.getenv("ASR_MODEL_TYPE", "nemo")  # "whisper" or "nemo"
    ASR_MAX_FILE_SIZE_MB = int(os.getenv("ASR_MAX_FILE_SIZE_MB", "25"))
    
    # NeMo model path
    NEMO_MODEL_PATH = os.getenv("NEMO_MODEL_PATH", 
        r"C:\Users\Asus\Downloads\RecallVision\indicconformer_stt_multi_hybrid_rnnt_600m.nemo")
    
    # LLM settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia")
    LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
    
    # embedding and retrieval
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "2"))
    
    # translation settings
    TRANSLATION_SOURCE_LANG = os.getenv("TRANSLATION_SOURCE_LANG", "auto")
    TRANSLATION_TARGET_LANG = os.getenv("TRANSLATION_TARGET_LANG", "en-IN")
    TRANSLATION_TIMEOUT = int(os.getenv("TRANSLATION_TIMEOUT", "30"))
    TRANSLATION_MAX_RETRIES = int(os.getenv("TRANSLATION_MAX_RETRIES", "3"))
    
    # wikipedia scraping
    WIKI_REQUEST_TIMEOUT = int(os.getenv("WIKI_REQUEST_TIMEOUT", "15"))
    WIKI_MIN_PARAGRAPH_LENGTH = int(os.getenv("WIKI_MIN_PARAGRAPH_LENGTH", "20"))
    
    # logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", None)
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.SARVAM_API_KEY:
            errors.append("SARVAM_API_KEY is required")
        
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is required")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    @classmethod
    def display(cls):
        """Display current configuration (hiding sensitive data)"""
        config_str = "Current Configuration:\n"
        config_str += f"  ASR Model: {cls.ASR_MODEL_NAME}\n"
        config_str += f"  LLM Provider: {cls.LLM_PROVIDER}\n"
        config_str += f"  LLM Model: {cls.LLM_MODEL}\n"
        config_str += f"  Embedding Model: {cls.EMBEDDING_MODEL}\n"
        config_str += f"  Chunk Size: {cls.CHUNK_SIZE}\n"
        config_str += f"  Chunk Overlap: {cls.CHUNK_OVERLAP}\n"
        config_str += f"  Top-K Retrieval: {cls.TOP_K_RETRIEVAL}\n"
        config_str += f"  Log Level: {cls.LOG_LEVEL}\n"
        return config_str


config = Config()
