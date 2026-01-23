"""
Combine multiple text files and build a single vector database
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import TextChunker, EmbeddingModel, VectorDatabase
from src.utils import get_logger, read_text_file
from src.config import config, VECTOR_DB_DIR, RAW_DATA_DIR

logger = get_logger(__name__)

def main():
    print("=" * 70)
    print("🔨 BUILDING COMBINED VECTOR DATABASE")
    print("=" * 70)
    
    # find all txt files in raw data directory
    txt_files = list(RAW_DATA_DIR.glob("*.txt"))
    
    if not txt_files:
        print("❌ No .txt files found in data/raw/")
        return 1
    
    print(f"\nFound {len(txt_files)} text files:")
    for f in txt_files:
        print(f"  - {f.name}")
    
    # read and combine all files
    all_text = []
    all_metadata = []
    
    for txt_file in txt_files:
        print(f"\n📖 Reading: {txt_file.name}")
        text = read_text_file(str(txt_file))
        if text:
            print(f"  ✅ {len(text):,} characters")
            all_text.append((text, str(txt_file)))
        else:
            print(f"  ❌ Failed to read")
    
    if not all_text:
        print("\n❌ No text content found")
        return 1
    
    # chunk all texts
    print(f"\n⏳ Chunking {len(all_text)} documents...")
    chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    
    all_chunks = []
    all_chunk_metadata = []
    
    for text, source in all_text:
        chunks_with_meta = chunker.chunk_with_metadata(text, source=source)
        chunks = [c['text'] for c in chunks_with_meta]
        all_chunks.extend(chunks)
        all_chunk_metadata.extend(chunks_with_meta)
        print(f"  ✅ {Path(source).name}: {len(chunks)} chunks")
    
    print(f"\n✅ Total chunks: {len(all_chunks)}")
    
    # generate embeddings
    print("\n⏳ Generating embeddings...")
    embedding_model = EmbeddingModel(config.EMBEDDING_MODEL)
    embeddings = embedding_model.encode(all_chunks, show_progress=True)
    
    print(f"✅ Generated embeddings: {embeddings.shape}")
    
    # create vector database
    print("\n⏳ Creating vector database...")
    vector_db = VectorDatabase(str(VECTOR_DB_DIR))
    vector_db.create(embeddings, all_chunks, all_chunk_metadata)
    
    print(f"✅ Created database with {vector_db.size} vectors")
    
    # save
    print("\n⏳ Saving...")
    vector_db.save()
    
    print(f"✅ Saved to: {VECTOR_DB_DIR}")
    
    print("\n" + "=" * 70)
    print("✅ VECTOR DATABASE BUILD COMPLETE!")
    print("=" * 70)
    
    print("\nDatabase Summary:")
    print(f"  Documents: {len(all_text)}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Average chunk size: {sum(len(c) for c in all_chunks) // len(all_chunks)} characters")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
