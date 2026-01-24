import sys
import requests
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_header(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def check_config():
    print_header("Checking Configuration")
    try:
        from src.config import config
        config.validate()
        print("✅ Configuration validation passed.")
        print(config.display())
        return True
    except ImportError:
        print("❌ Failed to import src.config")
        return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_service_health():
    print_header("Checking Service Health")
    url = "http://127.0.0.1:8001/health"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ ASR Service is HEALTHY: {response.json()}")
            return True
        else:
            print(f"❌ ASR Service returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {url}. Is the service running?")
        return False
    except Exception as e:
        print(f"❌ Error checking service: {e}")
        return False

def check_rag_pipeline():
    print_header("Testing RAG Pipeline")
    try:
        from src.rag import RAGPipeline
        pipeline = RAGPipeline()
        
        query = "What is Python?"
        print(f"Query: '{query}'")
        
        start_time = time.time()
        result = pipeline.process_text_query(query)
        duration = time.time() - start_time
        
        print(f"⏱️ Duration: {duration:.2f}s")
        print(f"✅ Answer: {result['answer']}")
        print(f"✅ Context Chunks: {len(result['context_chunks'])}")
        
        return True
    except Exception as e:
        print(f"❌ RAG Pipeline failed: {e}")
        return False

def main():
    print("Starting Sanity Checks...")
    
    if not check_config():
        sys.exit(1)
        
    if not check_service_health():
        print("⚠️ Warning: Service health check failed. Continuing...")
        
    if not check_rag_pipeline():
        sys.exit(1)
        
    print_header("Sanity Check Complete: ALL SYSTEMS GO 🚀")

if __name__ == "__main__":
    main()
