import requests
from typing import Dict, Any, Optional
from pathlib import Path

from src.vector_store import Retriever
from src.translation import translate_to_english
from src.rag.llm_client import LLMClient
from src.rag.prompt import create_rag_prompt
from src.utils.logger import get_logger
from src.config import config

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for voice Q&A"""
    
    def __init__(self, asr_endpoint=None, vector_db_path=None, 
                 llm_provider=None, llm_api_key=None):
        self.asr_endpoint = asr_endpoint or config.ASR_ENDPOINT
        
        logger.info("Initializing retriever...")
        self.retriever = Retriever(db_path=vector_db_path)
        
        logger.info("Initializing LLM client...")
        self.llm_client = LLMClient(provider=llm_provider, api_key=llm_api_key)
        
        logger.info("RAG pipeline initialized successfully")
    
    def transcribe_audio(self, audio_file_path: str) -> Dict[str, str]:
        """Send audio to ASR service and get transcription"""
        logger.info(f"Transcribing audio: {audio_file_path}")
        
        try:
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                response = requests.post(self.asr_endpoint, files=files, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"Transcription complete. Language: {result.get('language', 'unknown')}")
                
                return result
                
        except requests.RequestException as e:
            logger.error(f"ASR request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    def translate_query(self, text: str) -> str:
        """Translate text to English if needed"""
        logger.info("Translating to English...")
        
        try:
            translated = translate_to_english(text)
            logger.info("Translation complete")
            return translated
        except Exception as e:
            logger.warning(f"Translation failed, using original text: {e}")
            return text
    
    def retrieve_context(self, query: str, top_k=None) -> list:
        """Get relevant context chunks from vector DB"""
        logger.info(f"Retrieving context for query: {query[:100]}...")
        
        chunks = self.retriever.retrieve(query, top_k=top_k)
        
        logger.info(f"Retrieved {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Chunk {i}: {chunk[:100]}...")
        
        return chunks
    
    def generate_answer(self, query: str, context_chunks: list) -> str:
        """Generate answer using LLM with retrieved context"""
        logger.info("Generating answer...")
        
        prompt = create_rag_prompt(query, context_chunks)
        answer = self.llm_client.generate(prompt)
        
        logger.info("Answer generated successfully")
        return answer
    
    def process_audio_query(self, audio_file_path: str) -> Dict[str, Any]:
        """Process audio query through the full RAG pipeline"""
        logger.info("=" * 70)
        logger.info("PROCESSING AUDIO QUERY")
        logger.info("=" * 70)
        
        # step 1: transcribe
        logger.info("[1/5] Transcribing audio...")
        transcription_result = self.transcribe_audio(audio_file_path)
        transcribed_text = transcription_result.get('text', '')
        detected_language = transcription_result.get('language', 'unknown')
        
        # step 2: translate
        logger.info("[2/5] Translating to English...")
        english_text = self.translate_query(transcribed_text)
        
        # step 3: retrieve context
        logger.info("[3/5] Retrieving relevant context...")
        context_chunks = self.retrieve_context(english_text)
        
        # step 4: generate answer
        logger.info("[4/5] Generating answer...")
        answer = self.generate_answer(english_text, context_chunks)
        
        logger.info("[5/5] Complete!")
        
        results = {
            "transcription": transcribed_text,
            "detected_language": detected_language,
            "translation": english_text,
            "context_chunks": context_chunks,
            "answer": answer
        }
        
        return results
    
    def process_text_query(self, query: str) -> Dict[str, Any]:
        """Process text query (skip ASR and translation)"""
        logger.info("Processing text query...")
        
        context_chunks = self.retrieve_context(query)
        answer = self.generate_answer(query, context_chunks)
        
        return {
            "query": query,
            "context_chunks": context_chunks,
            "answer": answer
        }
