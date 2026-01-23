from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os

from src.utils.logger import get_logger
from src.config import config

logger = get_logger(__name__)

# initialize FastAPI app
app = FastAPI(
    title="RecallVision ASR Service",
    description="Audio transcription service using NeMo/Whisper ASR",
    version="2.0.0"
)

# choose ASR model based on config
if config.ASR_MODEL_TYPE == "nemo":
    from src.asr_service.nemo_asr_model import NeMoASRModel
    asr_model = NeMoASRModel()
    logger.info("Using NeMo ASR model for multilingual support")
else:
    from src.asr_service.asr_model import ASRModel
    asr_model = ASRModel()
    logger.info("Using Whisper ASR model")


class TranscriptionResponse(BaseModel):
    text: str
    language: str


@app.on_event("startup")
async def startup_event():
    """Load the Whisper model when the service starts"""
    logger.info("Starting ASR service...")
    asr_model.load()
    logger.info("ASR service ready")


@app.get("/")
async def root():
    return {
        "service": "RecallVision ASR Service",
        "version": "2.0.0",
        "model": config.ASR_MODEL_NAME,
        "endpoints": {
            "/transcribe": "POST - Transcribe audio file",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": asr_model.is_loaded,
        "model_name": asr_model.model_name
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe an audio file to text using Whisper"""
    try:
        # basic validation
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # read file content
        content = await audio.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        # check file size limit
        if file_size_mb > config.ASR_MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {config.ASR_MAX_FILE_SIZE_MB}MB"
            )
        
        # get file extension
        file_ext = os.path.splitext(audio.filename)[1]
        if not file_ext:
            file_ext = ".wav"  # default to wav
        
        # validate audio format
        valid_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac']
        if file_ext.lower() not in valid_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format. Supported: {', '.join(valid_extensions)}"
            )
        
        logger.info(f"Transcribing file: {audio.filename} ({file_size_mb:.2f}MB)")
        
        # do the actual transcription
        transcribed_text, detected_language = asr_model.transcribe_bytes(content, file_ext)
        
        logger.info(f"Transcription successful. Language: {detected_language}")
        
        return TranscriptionResponse(
            text=transcribed_text,
            language=detected_language
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting ASR service...")
    logger.info(f"API documentation: http://localhost:8001/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
