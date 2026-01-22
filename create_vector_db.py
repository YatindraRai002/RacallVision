# Task 2: Vector database with FAISS
# Chunk size: 500 chars, overlap: 50 chars
# This balances context vs precision for retrieval

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import argparse


def load_text(filepath='data/scraped_text.txt'):
    """
    Load the scraped text from file.
    
    Args:
        filepath (str): Path to the text file
        
    Returns:
        str: Text content
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded {len(text)} characters from {filepath}")
        return text
    except Exception as e:
        print(f"Error loading file: {e}")
        return ""


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    Split text into chunks with overlap.
    
    Args:
        text (str): Input text to chunk
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    try:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
            
            start += (chunk_size - chunk_overlap)
        
        print(f"Created {len(chunks)} chunks")
        print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        
        return chunks
    
    except Exception as e:
        print(f"Error chunking text: {e}")
        return []


def create_vector_database(chunks, persist_directory='data/faiss_db'):
    """
    Create embeddings and store in FAISS.
    
    Args:
        chunks (list): List of text chunks
        persist_directory (str): Directory to persist the database
        
    Returns:
        tuple: (faiss_index, chunks, embeddings)
    """
    try:
        print("Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
        
        # Initialize embedding model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print("Generating embeddings...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        
        # Convert to numpy array
        embeddings = np.array(embeddings).astype('float32')
        
        print("Creating FAISS index...")
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, os.path.join(persist_directory, 'index.faiss'))
        
        # Save chunks and metadata
        with open(os.path.join(persist_directory, 'chunks.pkl'), 'wb') as f:
            pickle.dump(chunks, f)
        
        with open(os.path.join(persist_directory, 'embeddings.pkl'), 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"Vector database created and persisted to {persist_directory}")
        print(f"Total documents in database: {len(chunks)}")
        
        return index, chunks, embeddings
    
    except Exception as e:
        print(f"Error creating vector database: {e}")
        return None, None, None


def main():
    """Main function to orchestrate vector database creation."""
    parser = argparse.ArgumentParser(
        description='Create a vector database from scraped Wikipedia text'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/scraped_text.txt',
        help='Input text file path (default: data/scraped_text.txt)'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/faiss_db',
        help='FAISS DB persist directory (default: data/faiss_db)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Chunk size in characters (default: 500)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='Chunk overlap in characters (default: 50)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VECTOR DATABASE CREATION (FAISS)")
    print("=" * 60)
    
    # Step 1: Load text
    text = load_text(args.input)
    if not text:
        print("No text loaded. Exiting.")
        return
    
    # Step 2: Chunk text
    chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
    if not chunks:
        print("No chunks created. Exiting.")
        return
    
    # Step 3: Create vector database
    index, chunks, embeddings = create_vector_database(chunks, args.db_path)
    
    if index:
        print("\n✓ Vector database creation completed successfully!")
        print(f"\nWhy FAISS?")
        print("  Benefits:")
        print("    - Extremely fast similarity search")
        print("    - Lightweight and efficient")
        print("    - Works great with Python 3.14")
        print("    - Industry-standard for vector search")
        print("  Drawbacks:")
        print("    - Requires manual persistence management")
        print("    - Less feature-rich than ChromaDB")
    else:
        print("\n✗ Failed to create vector database")


if __name__ == "__main__":
    main()
