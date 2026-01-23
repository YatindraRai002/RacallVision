import requests
from typing import Optional

from src.utils.logger import get_logger
from src.config import config

logger = get_logger(__name__)


class LLMClient:
    """LLM API client supporting multiple providers"""
    
    def __init__(self, provider=None, api_key=None, model=None, 
                 temperature=None, max_tokens=None):
        self.provider = provider or config.LLM_PROVIDER
        self.api_key = api_key or config.LLM_API_KEY
        self.model = model or config.LLM_MODEL
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self.timeout = config.LLM_TIMEOUT
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY is required")
        
        logger.info(f"LLM client initialized: provider={self.provider}, model={self.model}")
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        logger.info(f"Generating response with {self.provider}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        if self.provider == "groq":
            return self._call_groq(prompt)
        elif self.provider == "cohere":
            return self._call_cohere(prompt)
        elif self.provider == "nvidia":
            return self._call_nvidia(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API"""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model or "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()["choices"][0]["message"]["content"]
            logger.info("Response generated successfully")
            return result
            
        except requests.Timeout:
            logger.error("LLM request timed out")
            raise
        except requests.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    def _call_cohere(self, prompt: str) -> str:
        """Call Cohere API"""
        url = "https://api.cohere.ai/v1/generate"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model or "command",
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()["generations"][0]["text"]
            logger.info("Response generated successfully")
            return result
            
        except requests.Timeout:
            logger.error("LLM request timed out")
            raise
        except requests.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    def _call_nvidia(self, prompt: str) -> str:
        """Call NVIDIA API"""
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model or "meta/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()["choices"][0]["message"]["content"]
            logger.info("Response generated successfully")
            return result
            
        except requests.Timeout:
            logger.error("LLM request timed out")
            raise
        except requests.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            raise
