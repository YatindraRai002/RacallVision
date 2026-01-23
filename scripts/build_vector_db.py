"""
RecallVision - Vector Database Builder Script
Builds FAISS vector database from text files
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import TextChunker, EmbeddingModel, VectorDatabase
from src.utils import get_logger, read_text_file
from src.config import config, VECTOR_DB_DIR

logger = get_logger(__name__)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Build FAISS vector database from text files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_vector_db.py --input data/raw/artificial_intelligence.txt
  python scripts/build_vector_db.py --input data/raw/ai.txt --chunk_size 1000 --chunk_overlap 100
  python scripts/build_vector_db.py --input data/raw/ai.txt --output custom_db
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input text file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=str(VECTOR_DB_DIR),
        help=f'Output directory for vector database (default: {VECTOR_DB_DIR})'
    )
    
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=config.CHUNK_SIZE,
        help=f'Chunk size in characters (default: {config.CHUNK_SIZE})'
    )
    
    parser.add_argument(
        '--chunk_overlap',
        type=int,
        default=config.CHUNK_OVERLAP,
        help=f'Chunk overlap in characters (default: {config.CHUNK_OVERLAP})'
    )
    
    parser.add_argument(
        '--embedding_model',
        type=str,
        default=config.EMBEDDING_MODEL,
        help=f'Embedding model name (default: {config.EMBEDDING_MODEL})'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔨 VECTOR DATABASE BUILDER")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"Embedding model: {args.embedding_model}")
    print("=" * 70)
    print()
    
    logger.info(f"Reading input file: {args.input}")
    text = read_text_file(args.input)
    
    if not text:
        print("❌ Failed to read input file")
        return 1
    
    print(f"✅ Read {len(text):,} characters from input file")
    
    logger.info("Chunking text...")
    chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks_with_meta = chunker.chunk_with_metadata(text, source=args.input)
    chunks = [c['text'] for c in chunks_with_meta]
    
    print(f"✅ Created {len(chunks)} chunks")
    
    logger.info("Generating embeddings...")
    print("⏳ Generating embeddings (this may take a moment)...")
    
    embedding_model = EmbeddingModel(args.embedding_model)
    embeddings = embedding_model.encode(chunks, show_progress=True)
    
    print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    
    logger.info("Creating vector database...")
    vector_db = VectorDatabase(args.output)
    vector_db.create(embeddings, chunks, chunks_with_meta)
    
    print(f"✅ Created vector database with {vector_db.size} vectors")
    
    logger.info("Saving vector database...")
    vector_db.save()
    
    print(f"✅ Saved vector database to: {args.output}")
    
    print()
    print("=" * 70)
    print("✅ VECTOR DATABASE BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    print("\nDatabase Summary:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} characters")
    print(f"  Database location: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
