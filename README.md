# RecallVision - Voice Conversational RAG Assistant

A production-grade voice-enabled RAG (Retrieval-Augmented Generation) chatbot that processes audio queries, retrieves relevant context from a vector database, and generates intelligent responses using LLMs.

## Features

- **Audio Transcription**: Whisper-based ASR service for accurate speech-to-text
- **Multi-language Support**: Automatic translation using Sarvam AI
- **Vector Search**: Efficient semantic search using FAISS
- **Multiple LLM Providers**: Support for Groq, Cohere, and NVIDIA APIs
- **Wikipedia Integration**: Automated data collection from Wikipedia
- **Modular Architecture**: Clean separation of concerns for easy maintenance

## Project Structure

```
RecallVision/
├── src/
│   ├── asr_service/        # Whisper ASR service
│   ├── data_collection/    # Wikipedia scraping
│   ├── vector_store/       # Embeddings and retrieval
│   ├── translation/        # Sarvam translation
│   ├── rag/               # RAG pipeline and LLM client
│   └── utils/             # Logging and utilities
├── scripts/
│   ├── voice_chatbot.py   # Main CLI interface
│   └── build_vector_db.py # Vector DB builder
├── data/
│   ├── raw/               # Raw text data
│   └── vector_db/         # FAISS index
└── .env                   # Configuration
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env`:
   ```
   SARVAM_API_KEY=your_sarvam_key
   LLM_API_KEY=your_llm_key
   LLM_PROVIDER=nvidia
   ASR_MODEL_NAME=base
   ```

## Usage

### 1. Collect Data
```bash
python data_collection.py --query "Python programming language"
```

### 2. Build Vector Database
```bash
python scripts/build_vector_db.py
```

### 3. Start ASR Service
```bash
python -m src.asr_service.main
```

### 4. Run Voice Chatbot
```bash
# Process audio query
python scripts/voice_chatbot.py --audio question.wav

# Process text query
python scripts/voice_chatbot.py --text "What is Python?"
```

## Configuration

Key configuration options in `.env`:

- `ASR_MODEL_NAME`: Whisper model size (tiny, base, small, medium, large)
- `LLM_PROVIDER`: LLM provider (groq, cohere, nvidia)
- `LLM_MODEL`: Model name for the selected provider
- `CHUNK_SIZE`: Text chunk size for vector DB
- `TOP_K_RETRIEVAL`: Number of context chunks to retrieve

## Architecture

The system follows a modular pipeline architecture:

1. **ASR Service**: Converts audio to text using Whisper
2. **Translation**: Translates non-English text to English
3. **Retrieval**: Searches vector DB for relevant context
4. **Generation**: LLM generates answer using retrieved context

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a PR.
