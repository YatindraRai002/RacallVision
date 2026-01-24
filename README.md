# 🎙️ RecallVision

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![NVIDIA](https://img.shields.io/badge/NVIDIA_AI-76B900?style=for-the-badge&logo=nvidia)
![Whisper](https://img.shields.io/badge/OpenAI_Whisper-412991?style=for-the-badge&logo=openai)

**RecallVision** is a production-grade, voice-enabled **RAG (Retrieval-Augmented Generation)** assistant. It combines the power of **OpenAI Whisper** for state-of-the-art speech recognition, **FAISS** for efficient vector retrieval, and **NVIDIA's Llama 3.1** for intelligent response generation.

---

## ✨ Features

- **🗣️ Voice-First Interaction**: Record or upload audio to chat naturally.
- **🤖 Advanced ASR**: Powered by `Whisper` (or NeMo) for accurate multilingual transcription.
- **📚 RAG Pipeline**: Retrieves context from your Wikipedia-sourced knowledge base using `FAISS`.
- **⚡ Real-time Translation**: Automatically translates non-English queries using `Sarvam AI`.
- **🎨 Modern UI**: Beautiful, responsive interface built with `Streamlit`.
- **🛠️ Modular Backend**: FastAPI-based ASR service separated from the main application logic.

---

## 🚀 Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.10+**
2.  **FFmpeg** (CRITICAL for Audio Processing)
    > ⚠️ **Important**: The ASR service will fail with a `500 Error` if FFmpeg is missing.
    
    *   **Windows**: Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or use our included script (see Installation).
    *   **Linux**: `sudo apt install ffmpeg`
    *   **Mac**: `brew install ffmpeg`

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/RecallVision.git
cd RecallVision
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg (Windows Users)
If you don't have FFmpeg installed globally:
```bash
python scripts/install_ffmpeg_manual.py
```
*Note: We have patched the application to automatically detect FFmpeg installed by this script at `C:\ffmpeg_tool`.*

### 5. Configuration
Create a `.env` file in the root directory:
```env
# API Keys
SARVAM_API_KEY=your_sarvam_key
LLM_API_KEY=your_nvidia_key

# Model Settings
ASR_MODEL_TYPE=whisper
ASR_MODEL_NAME=base
LLM_PROVIDER=nvidia
```

---

## 🏃‍♂️ Usage

You need to run **two separate terminals** to start the full application.

### Terminal 1: ASR Service (Backend)
This service handles audio transcription.
```bash
# Make sure your virtual environment is activated
python -m src.asr_service.main
```
Wait until you see: `Uvicorn running on http://0.0.0.0:8001`

### Terminal 2: Streamlit App (Frontend)
This is the user interface.
```bash
# Make sure your virtual environment is activated
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🔧 Troubleshooting

### "Internal Server Error" (500) when recording
*   **Cause**: Missing or undetected FFmpeg.
*   **Fix**:
    1.  Run `python scripts/check_whisper.py` to verify detection.
    2.  Ensure you restarted your terminals after installing FFmpeg.
    3.  If using the manual script, ensure the `src/asr_service/asr_model.py` patch is present (it adds `C:\ffmpeg_tool` to PATH).

### "Method Not Allowed" on port 8001
*   **Cause**: Visiting `http://localhost:8001/transcribe` in a browser.
*   **Fix**: This is normal! The endpoint only accepts `POST` requests. Visit `http://localhost:8001/health` instead to check status.

---

## 🏗️ Architecture

```
User Voice 🎤 
  │
  ▼
[Streamlit UI] ──(Audio File)──▶ [FastAPI ASR Service]
                                      │
                                      ▼
                                  [Whisper Model]
                                      │
                                   (Text)
                                      ▼
[RAG Pipeline] ◀──(Query)─────────────┘
  │
  ├──▶ [Translation (Sarvam)]
  ├──▶ [Vector DB (FAISS)] ──▶ Retrieve Context
  └──▶ [LLM (NVIDIA)] ──▶ Generate Answer
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a Pull Request.
