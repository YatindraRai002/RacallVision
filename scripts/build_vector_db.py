

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
  # Process single file
  python scripts/build_vector_db.py --input data/raw/artificial_intelligence.txt
  
  # Process all .txt files in directory
  python scripts/build_vector_db.py --input data/raw
  
  # Custom settings
  python scripts/build_vector_db.py --input data/raw --chunk_size 1000 --output custom_db
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input text file or directory containing .txt files'
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
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"Embedding model: {args.embedding_model}")
    print("=" * 70)
    print()
    
    input_path = Path(args.input)
    files_to_process = []
    
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        files_to_process.extend(list(input_path.glob("*.txt")))
    else:
        print(f" Input path not found: {args.input}")
        return 1
        
    if not files_to_process:
        print(f" No files to process. If excluding directory, make sure it contains .txt files.")
        return 1
        
    print(f"Found {len(files_to_process)} files to process.")
    
    # Read and chunk files
    all_chunks = []
    all_chunk_metadata = []
    
    chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    
    for file_path in files_to_process:
        logger.info(f"Processing: {file_path.name}")
        print(f"\n Processing: {file_path.name}")
        
        text = read_text_file(str(file_path))
        if not text:
            print(f"   Failed to read or empty")
            continue
            
        chunks_with_meta = chunker.chunk_with_metadata(text, source=str(file_path))
        
        if chunks_with_meta:
            file_chunks = [c['text'] for c in chunks_with_meta]
            all_chunks.extend(file_chunks)
            all_chunk_metadata.extend(chunks_with_meta)
            print(f"   Generated {len(file_chunks)} chunks")
        else:
            print(f"   No chunks generated")

    if not all_chunks:
        print("\n No chunks generated from any files.")
        return 1
        
    print(f"\n Total chunks to embed: {len(all_chunks)}")
    
    logger.info("Generating embeddings...")
    print("\n Generating embeddings (this may take a moment)...")
    
    embedding_model = EmbeddingModel(args.embedding_model)
    embeddings = embedding_model.encode(all_chunks, show_progress=True)
    
    print(f" Generated embeddings with shape: {embeddings.shape}")
    
    logger.info("Creating vector database...")
    vector_db = VectorDatabase(args.output)
    vector_db.create(embeddings, all_chunks, all_chunk_metadata)
    
    print(f" Created vector database with {vector_db.size} vectors")
    
    logger.info("Saving vector database...")
    vector_db.save()
    
    print(f" Saved vector database to: {args.output}")
    
    print()
    print("=" * 70)
    print(" BUILD COMPLETE!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
