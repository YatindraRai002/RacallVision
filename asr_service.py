# Task 3: ASR service with Whisper
# Using whisper-base model for CPU compatibility

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
import uvicorn
from pydantic import BaseModel


# Initialize FastAPI app
app = FastAPI(
    title="ASR Service",
    description="Audio transcription service using Whisper ASR",
    version="1.0.0"
)

# Load Whisper model (lazy loading on first request)
model = None


class TranscriptionResponse(BaseModel):
    """Response model for transcription"""
    text: str
    language: str


def load_model(model_name="base"):
    """
    Load the Whisper model.
    
    Args:
        model_name (str): Whisper model size (tiny, base, small, medium, large)
        
    Returns:
        whisper.Whisper: Loaded model
    """
    global model
    if model is None:
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        print("Model loaded successfully")
    return model


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model("base")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ASR Service is running",
        "endpoints": {
            "/transcribe": "POST - Transcribe audio file",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio file to text.
    
    Args:
        audio (UploadFile): Audio file (supports mp3, wav, m4a, etc.)
        
    Returns:
        TranscriptionResponse: Transcribed text and detected language
    """
    try:
        # Validate file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Get file extension
        file_ext = os.path.splitext(audio.filename)[1]
        
        # Create temporary file to store uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Write uploaded file to temp file
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Load model if not already loaded
            whisper_model = load_model("base")
            
            # Transcribe audio
            print(f"Transcribing audio file: {audio.filename}")
            result = whisper_model.transcribe(temp_file_path)
            
            transcribed_text = result["text"].strip()
            detected_language = result.get("language", "unknown")
            
            print(f"Transcription complete. Language: {detected_language}")
            print(f"Text: {transcribed_text[:100]}...")
            
            return TranscriptionResponse(
                text=transcribed_text,
                language=detected_language
            )
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


def main():
    """Run the FastAPI server"""
    print("Starting ASR Service...")
    print("API documentation available at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
