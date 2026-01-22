# RecallVision - Voice-Enabled RAG Chatbot

A comprehensive end-to-end voice-enabled RAG (Retrieval-Augmented Generation) chatbot that scrapes Wikipedia articles, stores knowledge in a vector database, transcribes audio queries, translates them to English, and generates intelligent answers using LLMs.

## 🎯 Project Overview

This project implements a complete RAG pipeline with the following components:

1. **Data Collection**: Wikipedia article search and scraping
2. **Vector Database**: Text chunking and embedding storage using ChromaDB
3. **ASR Service**: Audio transcription using Whisper model via FastAPI
4. **Translation**: Text translation using Sarvam AI API
5. **RAG Pipeline**: End-to-end integration of all components
6. **Streamlit UI**: Interactive web interface with audio recording/upload

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Setup Instructions](#setup-instructions)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Technical Decisions](#technical-decisions)
- [Observations and Challenges](#observations-and-challenges)
- [Future Improvements](#future-improvements)

## ✨ Features

- 🎤 **Voice Input**: Record or upload audio questions
- 🌐 **Multi-language Support**: Automatic transcription and translation
- 📚 **Wikipedia Knowledge**: Scrape and index any Wikipedia article
- 🔍 **Semantic Search**: Vector-based retrieval for relevant context
- 🤖 **LLM Integration**: Support for Groq, Cohere, and NVIDIA APIs
- 💬 **Chat Interface**: Beautiful Streamlit UI with chat history
- ⚡ **Fast API**: RESTful ASR endpoint for scalability

## 🏗️ Architecture

```
┌─────────────┐
│ Audio Input │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  ASR Service    │ (Whisper Model)
│  (FastAPI)      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Translation    │ (Sarvam AI API)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Vector DB      │ (ChromaDB)
│  Retrieval      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  LLM API        │ (Groq/Cohere/NVIDIA)
│  Generation     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Final Answer   │
└─────────────────┘
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection for API calls

### Step 1: Clone or Download the Project

```bash
cd RecallVision
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The installation may take 5-10 minutes as it downloads the Whisper model and sentence-transformers.

### Step 4: Configure API Keys

1. Copy the example environment file:
   ```bash
   copy .env.example .env  # Windows
   cp .env.example .env    # Linux/Mac
   ```

2. Edit `.env` and add your API keys:
   ```
   SARVAM_API_KEY=your_sarvam_api_key_here
   LLM_API_KEY=your_llm_api_key_here
   ```

**Getting API Keys:**
- **Sarvam AI**: Sign up at https://docs.sarvam.ai/ (1000 free credits)
- **Groq**: Get free API key at https://console.groq.com/
- **Cohere**: Get free API key at https://dashboard.cohere.com/
- **NVIDIA**: Get free API key at https://build.nvidia.com/

### Step 5: Load Environment Variables

```bash
# Windows PowerShell
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.+)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

# Windows CMD
for /f "tokens=*" %i in (.env) do set %i

# Linux/Mac
export $(cat .env | xargs)
```

## 📖 Usage Guide

### Task 1: Data Collection

Scrape a Wikipedia article on any topic:

```bash
python data_collection.py "Artificial Intelligence"
```

**Options:**
```bash
python data_collection.py "Machine Learning" --output data/ml_article.txt
```

**Output**: Creates `data/scraped_text.txt` with the article content.

### Task 2: Create Vector Database

Generate embeddings and store in ChromaDB:

```bash
python create_vector_db.py
```

**Options:**
```bash
python create_vector_db.py --input data/scraped_text.txt --db-path data/chroma_db --chunk-size 500 --chunk-overlap 50
```

**Output**: Creates `data/chroma_db/` directory with the vector database.

### Task 3: Start ASR Service

Run the FastAPI ASR service in a separate terminal:

```bash
python asr_service.py
```

The service will start on `http://localhost:8000`

**API Documentation**: Visit `http://localhost:8000/docs` for interactive API docs.

### Task 4: Test Translation

Test the translation function:

```bash
python translation.py "नमस्ते, आप कैसे हैं?"
```

### Task 5: Run RAG Pipeline

Process an audio query through the complete pipeline:

```bash
python rag_pipeline.py path/to/audio.wav --llm-provider groq
```

**Options:**
```bash
python rag_pipeline.py audio.wav --asr-endpoint http://localhost:8000/transcribe --db-path data/chroma_db --llm-provider cohere
```

### Bonus: Launch Streamlit UI

Start the interactive web interface:

```bash
streamlit run app.py
```

The UI will open in your browser at `http://localhost:8501`

**Features:**
- 🎙️ Record audio directly in the browser
- 📤 Upload audio files (WAV, MP3, M4A, OGG)
- 💬 View chat history with processing details
- ⚙️ Configure endpoints and LLM provider

## 🔌 API Documentation

### ASR Service Endpoints

#### POST `/transcribe`

Transcribe audio to text.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Audio file

**Example:**
```bash
curl -X POST "http://localhost:8000/transcribe" -F "audio=@question.wav"
```

**Response:**
```json
{
  "text": "What is artificial intelligence?",
  "language": "en"
}
```

#### GET `/health`

Check service health.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🧠 Technical Decisions

### 1. Search API Choice: DuckDuckGo Search

**Why?**
- ✅ Free and unlimited
- ✅ No API key required
- ✅ Good Wikipedia search results
- ✅ Easy to use with `duckduckgo-search` library

**Alternatives Considered:**
- Google Search API (limited free tier)
- Bing API (requires API key)

### 2. Chunking Strategy

**Configuration:**
- Chunk Size: 500 characters (~100-125 tokens)
- Overlap: 50 characters (10%)

**Justification:**
- **500 chars**: Balances context preservation with retrieval precision
  - Smaller chunks: More precise but lose context
  - Larger chunks: Better context but less precise matching
- **50 char overlap**: Ensures continuity and prevents information loss at boundaries
- **RecursiveCharacterTextSplitter**: Respects natural text boundaries (paragraphs, sentences)

### 3. Embedding Model: sentence-transformers/all-MiniLM-L6-v2

**Why?**
- ✅ Runs locally (no API costs)
- ✅ Fast inference on CPU
- ✅ Good quality embeddings (384 dimensions)
- ✅ Well-supported by LangChain

**Alternatives:**
- OpenAI embeddings (costs money)
- Larger models (slower, require GPU)

### 4. Vector Database: ChromaDB

**Benefits:**
- ✅ Lightweight and easy to set up
- ✅ Persistent local storage (no external server)
- ✅ Fast similarity search
- ✅ Excellent LangChain integration
- ✅ Perfect for development and small-scale production

**Drawbacks:**
- ❌ Not suitable for very large-scale deployments
- ❌ Limited distributed computing capabilities
- ❌ Single-machine performance limits

**Alternatives:**
- Pinecone (cloud-based, costs money)
- Weaviate (more complex setup)
- FAISS (no persistence by default)

### 5. ASR Model: Whisper Base

**Why?**
- ✅ Good balance between accuracy and speed
- ✅ Works well on CPU
- ✅ Supports 99+ languages
- ✅ Robust to accents and background noise
- ✅ Open-source and free

**Model Sizes:**
- `tiny`: Fastest, lowest accuracy
- `base`: **Selected** - Good balance
- `small`: Better accuracy, slower
- `medium/large`: Best accuracy, requires GPU

### 6. LLM Providers: Groq/Cohere/NVIDIA

**Why Multiple Providers?**
- ✅ Flexibility for users
- ✅ Free tiers available
- ✅ Different strengths (speed vs quality)

**Default: Groq**
- Very fast inference
- Free tier with good limits
- Llama 3.3 70B model

## 🔍 Observations and Challenges

### Challenges Faced

#### 1. **Whisper Model Download Size**
- **Issue**: First-time setup downloads ~150MB model
- **Solution**: Used `base` model (good balance), documented in README
- **Learning**: Consider model caching for production

#### 2. **API Rate Limits**
- **Issue**: Free API tiers have rate limits
- **Solution**: Implemented proper error handling and fallbacks
- **Learning**: Need to implement retry logic with exponential backoff

#### 3. **Audio Format Compatibility**
- **Issue**: Different audio formats (WAV, MP3, M4A)
- **Solution**: Whisper handles multiple formats natively
- **Learning**: Always test with various input formats

#### 4. **Chunking Optimization**
- **Issue**: Finding optimal chunk size for retrieval
- **Solution**: Tested multiple sizes (300, 500, 1000 chars)
- **Result**: 500 chars with 50 overlap worked best for Wikipedia articles
- **Learning**: Chunk size should match typical query scope

#### 5. **Environment Variable Management**
- **Issue**: API keys need to be secure and easy to configure
- **Solution**: Used `.env` file with `.env.example` template
- **Learning**: Never commit actual API keys to version control

#### 6. **Streamlit Audio Recording**
- **Issue**: Browser audio recording requires special library
- **Solution**: Used `audio-recorder-streamlit` package
- **Learning**: Web audio APIs have browser compatibility issues

#### 7. **Translation API Integration**
- **Issue**: Sarvam API documentation had limited examples
- **Solution**: Tested with curl first, then implemented in Python
- **Learning**: Always test APIs independently before integration

### Performance Observations

1. **Data Collection**: ~2-5 seconds for typical Wikipedia article
2. **Vector DB Creation**: ~10-30 seconds depending on article size
3. **ASR Transcription**: ~1-3 seconds for 10-second audio (base model)
4. **Translation**: ~0.5-1 second per request
5. **LLM Generation**: ~2-5 seconds 

**Total Pipeline Latency**: ~6-15 seconds for end-to-end query

### Key Learnings

1. **Modular Design**: Separating components made testing easier
2. **Error Handling**: Critical for production-ready systems
3. **Documentation**: Clear setup instructions reduce user friction
4. **Free Tiers**: Sufficient for development and demos
5. **Local Models**: Reduce costs but require more compute

## 🔮 Future Improvements

### Short-term
- [ ] Add retry logic with exponential backoff for API calls
- [ ] Implement caching for repeated queries
- [ ] Add support for multiple Wikipedia articles
- [ ] Improve error messages in UI
- [ ] Add audio playback for answers (TTS)

### Medium-term
- [ ] Add conversation history/context
- [ ] Implement streaming responses in UI
- [ ] Add support for custom documents (PDFs, etc.)
- [ ] Deploy ASR service to cloud (Docker + Cloud Run)
- [ ] Add authentication for API endpoints

### Long-term
- [ ] Multi-turn conversations with memory
- [ ] Support for multiple languages in answers
- [ ] Fine-tune embedding model for domain-specific use
- [ ] Implement hybrid search (semantic + keyword)
- [ ] Add evaluation metrics (accuracy, latency)

## 📝 Project Structure

```
RecallVision/
├── data/                      # Data directory (created automatically)
│   ├── scraped_text.txt      # Scraped Wikipedia article
│   └── chroma_db/            # Vector database
├── data_collection.py        # Task 1: Wikipedia scraping
├── create_vector_db.py       # Task 2: Vector database creation
├── asr_service.py            # Task 3: ASR FastAPI service
├── translation.py            # Task 4: Translation function
├── rag_pipeline.py           # Task 5: Complete RAG pipeline
├── app.py                    # Bonus: Streamlit UI
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore file
└── README.md                # This file
```

## 🤝 Contributing

This project was created as part of an AI/ML assignment. Feel free to fork and improve!



## 🙏 Acknowledgments

- **OpenAI Whisper**: ASR model
- **Sarvam AI**: Translation API
- **LangChain**: RAG framework
- **ChromaDB**: Vector database
- **Streamlit**: UI framework
- **HuggingFace**: Embedding models

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

