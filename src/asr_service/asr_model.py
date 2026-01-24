import whisper
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional

from src.utils.logger import get_logger
from src.config import config

logger = get_logger(__name__)


class ASRModel:
    """Whisper ASR model wrapper"""
    
    def __init__(self, model_name=None):
        # Explicitly add FFmpeg to PATH
        ffmpeg_path = r"C:\ffmpeg_tool"
        if ffmpeg_path not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + ffmpeg_path
            logger.info(f"Added {ffmpeg_path} to PATH")
            
        self.model_name = model_name or config.ASR_MODEL_NAME
        self.model = None
        logger.info(f"ASR model initialized: {self.model_name}")
    
    def load(self):
        """Load the Whisper model"""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            try:
                self.model = whisper.load_model(self.model_name)
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    def transcribe(self, audio_path: str) -> Tuple[str, str]:
        """Transcribe audio file to text"""
        if self.model is None:
            self.load()
        
        logger.info(f"Transcribing audio: {audio_path}")
        
        try:
            result = self.model.transcribe(audio_path)
            
            transcribed_text = result["text"].strip()
            detected_language = result.get("language", "unknown")
            
            logger.info(f"Transcription complete. Language: {detected_language}")
            logger.debug(f"Transcribed text: {transcribed_text[:100]}...")
            
            return transcribed_text, detected_language
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_bytes(self, audio_bytes: bytes, file_ext=".wav") -> Tuple[str, str]:
        """Transcribe audio from bytes"""
        # create temp file for whisper to process
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        try:
            text, language = self.transcribe(temp_file_path)
            return text, language
        finally:
            # cleanup temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
