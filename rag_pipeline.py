# Task 5: Complete RAG pipeline
# Integrates ASR, translation, vector search, and LLM

import os
import pickle
import requests
import numpy as np
import faiss
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from translation import translate_text


class RAGPipeline:
    """End-to-end RAG pipeline for voice-enabled Q&A"""
    
    def __init__(
        self,
        asr_endpoint: str = "http://localhost:8000/transcribe",
        vector_db_path: str = "data/faiss_db",
        llm_api_key: Optional[str] = None,
        llm_provider: str = "nvidia",  # Options: groq, cohere, nvidia
        sarvam_api_key: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            asr_endpoint (str): URL of the ASR service endpoint
            vector_db_path (str): Path to FAISS database
            llm_api_key (str): API key for LLM provider
            llm_provider (str): LLM provider (groq, cohere, nvidia)
            sarvam_api_key (str): Sarvam API key for translation
        """
        self.asr_endpoint = asr_endpoint
        self.vector_db_path = vector_db_path
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY")
        self.llm_provider = llm_provider
        self.sarvam_api_key = sarvam_api_key or os.getenv("SARVAM_API_KEY")
        
        # Initialize embedding model and vector store
        print("Loading vector database...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load FAISS index
        index_path = os.path.join(vector_db_path, 'index.faiss')
        chunks_path = os.path.join(vector_db_path, 'chunks.pkl')
        
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Vector database not found at {vector_db_path}. Please run create_vector_db.py first.")
        
        self.index = faiss.read_index(index_path)
        
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"Vector database loaded with {len(self.chunks)} documents")
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Step 1: Transcribe audio using ASR service.
        
        Args:
            audio_file_path (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            print(f"Transcribing audio: {audio_file_path}")
            
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                response = requests.post(self.asr_endpoint, files=files, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                transcribed_text = result.get('text', '')
                language = result.get('language', 'unknown')
                
                print(f"Transcription complete. Language: {language}")
                print(f"Text: {transcribed_text}")
                
                return transcribed_text
        
        except Exception as e:
            print(f"Error during transcription: {e}")
            raise
    
    def translate_to_english(self, text: str) -> str:
        """
        Step 2: Translate text to English using Sarvam API.
        
        Args:
            text (str): Text to translate
            
        Returns:
            str: Translated text
        """
        try:
            translated = translate_text(
                text=text,
                source_language="auto",
                target_language="en-IN",
                api_key=self.sarvam_api_key
            )
            
            return translated
        
        except Exception as e:
            print(f"Translation error (using original text): {e}")
            return text  # Fallback to original text
    
    def retrieve_context(self, query: str, top_k: int = 2) -> list:
        """
        Step 3: Retrieve top-k relevant chunks from vector database.
        
        Args:
            query (str): Search query
            top_k (int): Number of chunks to retrieve
            
        Returns:
            list: List of relevant text chunks
        """
        try:
            print(f"Retrieving top-{top_k} chunks for query: {query}")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            
            # Search in FAISS index
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Get corresponding chunks
            retrieved_chunks = [self.chunks[idx] for idx in indices[0]]
            
            print(f"Retrieved {len(retrieved_chunks)} chunks")
            for i, chunk in enumerate(retrieved_chunks, 1):
                print(f"Chunk {i}: {chunk[:100]}...")
            
            return retrieved_chunks
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def generate_answer(self, query: str, context_chunks: list) -> str:
        """
        Step 4: Generate answer using LLM API.
        
        Args:
            query (str): User question
            context_chunks (list): Retrieved context chunks
            
        Returns:
            str: Generated answer
        """
        try:
            # Combine context chunks
            context = "\n\n".join(context_chunks)
            
            # Create prompt
            prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer: Provide a clear, concise answer based only on the information in the context. If the context doesn't contain enough information, say so."""
            
            # Call LLM based on provider
            if self.llm_provider == "groq":
                answer = self._call_groq(prompt)
            elif self.llm_provider == "cohere":
                answer = self._call_cohere(prompt)
            elif self.llm_provider == "nvidia":
                answer = self._call_nvidia(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
            
            return answer
        
        except Exception as e:
            print(f"Error during answer generation: {e}")
            raise
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API"""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_cohere(self, prompt: str) -> str:
        """Call Cohere API"""
        url = "https://api.cohere.ai/v1/generate"
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "command",
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        return response.json()["generations"][0]["text"]
    
    def _call_nvidia(self, prompt: str) -> str:
        """Call NVIDIA API"""
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def process_audio_query(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Process an audio query through the entire RAG pipeline.
        
        Args:
            audio_file_path (str): Path to audio file
            
        Returns:
            dict: Results containing transcription, translation, context, and answer
        """
        print("\n" + "=" * 80)
        print("PROCESSING AUDIO QUERY")
        print("=" * 80)
        
        # Step 1: Transcribe audio
        print("\n[1/5] Transcribing audio...")
        transcribed_text = self.transcribe_audio(audio_file_path)
        
        # Step 2: Translate to English
        print("\n[2/5] Translating to English...")
        english_text = self.translate_to_english(transcribed_text)
        
        # Step 3: Retrieve context
        print("\n[3/5] Retrieving relevant context...")
        context_chunks = self.retrieve_context(english_text, top_k=2)
        
        # Step 4: Generate answer
        print("\n[4/5] Generating answer...")
        answer = self.generate_answer(english_text, context_chunks)
        
        print("\n[5/5] Complete!")
        
        # Return results
        results = {
            "transcription": transcribed_text,
            "translation": english_text,
            "context": context_chunks,
            "answer": answer
        }
        
        return results


def main():
    """Test the RAG pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Pipeline for voice-enabled Q&A')
    parser.add_argument('audio_file', type=str, help='Path to audio file')
    parser.add_argument(
        '--asr-endpoint',
        type=str,
        default='http://localhost:8000/transcribe',
        help='ASR service endpoint'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/faiss_db',
        help='Vector database path'
    )
    parser.add_argument(
        '--llm-provider',
        type=str,
        default='nvidia',
        choices=['groq', 'cohere', 'nvidia'],
        help='LLM provider'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        asr_endpoint=args.asr_endpoint,
        vector_db_path=args.db_path,
        llm_provider=args.llm_provider
    )
    
    # Process audio query
    results = pipeline.process_audio_query(args.audio_file)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nTranscription: {results['transcription']}")
    print(f"\nTranslation: {results['translation']}")
    print(f"\nAnswer: {results['answer']}")


if __name__ == "__main__":
    main()
