"""
RecallVision - Voice Conversational RAG Assistant
Main CLI for processing audio queries through the RAG pipeline
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import RAGPipeline
from src.utils import get_logger
from src.config import config

logger = get_logger(__name__)


def print_results(results: dict):
    """Print pipeline results in a formatted way"""
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    
    if "transcription" in results:
        print(f"\n🎤 Transcription ({results.get('detected_language', 'unknown')}):")
        print(f"   {results['transcription']}")
    
    if "translation" in results and results['translation'] != results.get('transcription'):
        print(f"\n🌐 Translation (English):")
        print(f"   {results['translation']}")
    
    if "query" in results:
        print(f"\n❓ Query:")
        print(f"   {results['query']}")
    
    print(f"\n📚 Retrieved Context ({len(results['context_chunks'])} chunks):")
    for i, chunk in enumerate(results['context_chunks'], 1):
        print(f"\n   [{i}] {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
    
    print(f"\n💡 Answer:")
    print(f"   {results['answer']}")
    
    print("\n" + "=" * 70)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='RecallVision - Voice Conversational RAG Assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process audio query
  python scripts/voice_chatbot.py --audio question.wav
  
  # Process text query
  python scripts/voice_chatbot.py --text "What is artificial intelligence?"
  
  # Specify custom vector DB
  python scripts/voice_chatbot.py --audio query.mp3 --db_path custom_db
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--audio',
        type=str,
        help='Path to audio file'
    )
    input_group.add_argument(
        '--text',
        type=str,
        help='Text query (skip ASR)'
    )
    
    parser.add_argument(
        '--db_path',
        type=str,
        default=None,
        help='Path to vector database (default: from config)'
    )
    
    parser.add_argument(
        '--asr_endpoint',
        type=str,
        default=config.ASR_ENDPOINT,
        help=f'ASR service endpoint (default: {config.ASR_ENDPOINT})'
    )
    
    parser.add_argument(
        '--llm_provider',
        type=str,
        default=config.LLM_PROVIDER,
        choices=['groq', 'cohere', 'nvidia'],
        help=f'LLM provider (default: {config.LLM_PROVIDER})'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=config.TOP_K_RETRIEVAL,
        help=f'Number of context chunks to retrieve (default: {config.TOP_K_RETRIEVAL})'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎙️  RECALLVISION - VOICE CONVERSATIONAL RAG ASSISTANT")
    print("=" * 70)
    print(f"LLM Provider: {args.llm_provider}")
    print(f"Top-K Retrieval: {args.top_k}")
    if args.audio:
        print(f"Audio File: {args.audio}")
        print(f"ASR Endpoint: {args.asr_endpoint}")
    else:
        print(f"Text Query: {args.text}")
    print("=" * 70)
    print()
    
    try:
        logger.info("Initializing RAG pipeline...")
        print("⏳ Initializing RAG pipeline...")
        
        pipeline = RAGPipeline(
            asr_endpoint=args.asr_endpoint,
            vector_db_path=args.db_path,
            llm_provider=args.llm_provider
        )
        
        print("✅ Pipeline initialized\n")
        
        if args.audio:
            print(f"🎤 Processing audio query...")
            results = pipeline.process_audio_query(args.audio)
        else:
            print(f"💬 Processing text query...")
            results = pipeline.process_text_query(args.text)
        
        print_results(results)
        
        print("\n✅ Query processed successfully!")
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
