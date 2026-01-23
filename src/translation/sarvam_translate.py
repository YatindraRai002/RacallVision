import requests
import time
from typing import Optional

from src.utils.logger import get_logger
from src.config import config

logger = get_logger(__name__)


class SarvamTranslator:
    """Sarvam API translator with retry logic"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or config.SARVAM_API_KEY
        
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is required")
        
        self.base_url = "https://api.sarvam.ai/translate"
        self.timeout = config.TRANSLATION_TIMEOUT
        self.max_retries = config.TRANSLATION_MAX_RETRIES
        
        logger.info("Sarvam translator initialized")
    
    def translate(self, text: str, source_language=None, 
                  target_language=None) -> str:
        """Translate text using Sarvam API with retry logic"""
        if not text:
            return text
        
        source_lang = source_language or config.TRANSLATION_SOURCE_LANG
        target_lang = target_language or config.TRANSLATION_TARGET_LANG
        
        logger.info(f"Translating text ({source_lang} -> {target_lang})")
        logger.debug(f"Input text: {text[:100]}...")
        
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": text,
            "source_language_code": source_lang,
            "target_language_code": target_lang,
            "speaker_gender": "Male",
            "mode": "formal",
            "model": "mayura:v1",
            "enable_preprocessing": True
        }
        
        # retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                
                result = response.json()
                translated_text = result.get("translated_text", text)
                
                logger.info("Translation successful")
                logger.debug(f"Translated text: {translated_text[:100]}...")
                
                return translated_text
                
            except requests.Timeout:
                logger.warning(f"Translation timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Translation failed after all retries (timeout)")
                    return text
                    
            except requests.RequestException as e:
                logger.warning(f"Translation error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Translation failed after all retries: {e}")
                    return text
                    
            except Exception as e:
                logger.error(f"Unexpected translation error: {e}")
                return text
        
        return text


def translate_to_english(text: str, api_key=None) -> str:
    """Helper function to translate text to English"""
    translator = SarvamTranslator(api_key)
    return translator.translate(text, target_language="en-IN")
