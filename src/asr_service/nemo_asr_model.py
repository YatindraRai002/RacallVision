"""
NeMo-based ASR Model for Multilingual Indian Languages
Supports Hindi, English, and other Indic languages
"""

import nemo.collections.asr as nemo_asr
import tempfile
import os
from pathlib import Path
from typing import Tuple

from src.utils.logger import get_logger
from src.config import config

logger = get_logger(__name__)


class NeMoASRModel:
    """NeMo ASR model wrapper for multilingual Indic speech recognition"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or config.NEMO_MODEL_PATH
        self.model = None
        logger.info(f"NeMo ASR model initialized with path: {self.model_path}")
    
    def load(self):
        """Load the NeMo ASR model"""
        if self.model is None:
            logger.info(f"Loading NeMo model from: {self.model_path}")
            try:
                # load the .nemo model file
                self.model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(self.model_path)
                self.model.eval()  # set to evaluation mode
                logger.info("NeMo model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load NeMo model: {e}")
                raise
    
    def transcribe(self, audio_path: str) -> Tuple[str, str]:
        """Transcribe audio file to text"""
        if self.model is None:
            self.load()
        
        logger.info(f"Transcribing audio: {audio_path}")
        
        try:
            # transcribe using NeMo model
            transcriptions = self.model.transcribe([audio_path])
            
            if transcriptions and len(transcriptions) > 0:
                transcribed_text = transcriptions[0].strip()
                
                # NeMo Indic models support multiple languages
                # detect language based on script/content
                detected_language = self._detect_language(transcribed_text)
                
                logger.info(f"Transcription complete. Language: {detected_language}")
                logger.debug(f"Transcribed text: {transcribed_text[:100]}...")
                
                return transcribed_text, detected_language
            else:
                logger.warning("No transcription returned")
                return "", "unknown"
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_bytes(self, audio_bytes: bytes, file_ext=".wav") -> Tuple[str, str]:
        """Transcribe audio from bytes"""
        # create temp file for NeMo to process
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
    
    def _detect_language(self, text: str) -> str:
        """Detect language from transcribed text"""
        # simple heuristic: check for Devanagari script (Hindi)
        has_devanagari = any('\u0900' <= char <= '\u097F' for char in text)
        
        if has_devanagari:
            return "hi"  # Hindi
        elif text.strip():
            return "en"  # English (default for Latin script)
        else:
            return "unknown"
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
