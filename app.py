# Streamlit UI for RAG chatbot

import streamlit as st
import os
import tempfile
from rag_pipeline import RAGPipeline
from audio_recorder_streamlit import audio_recorder
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="RecallVision - Voice RAG Chatbot",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.5s;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .bot-message {
        background: #f0f2f6;
        color: #333;
        margin-right: 20%;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .info-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'pipeline_initialized' not in st.session_state:
        st.session_state.pipeline_initialized = False


def initialize_pipeline(asr_endpoint, db_path, llm_provider):
    """Initialize the RAG pipeline"""
    try:
        with st.spinner("🔄 Initializing RAG pipeline..."):
            pipeline = RAGPipeline(
                asr_endpoint=asr_endpoint,
                vector_db_path=db_path,
                llm_provider=llm_provider
            )
            st.session_state.pipeline = pipeline
            st.session_state.pipeline_initialized = True
            st.success("✅ Pipeline initialized successfully!")
            return True
    except Exception as e:
        st.error(f"❌ Failed to initialize pipeline: {e}")
        return False


def display_chat_message(role, content, show_details=False, details=None):
    """Display a chat message with styling"""
    css_class = "user-message" if role == "user" else "bot-message"
    icon = "🎤" if role == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {role.upper()}</strong><br>
        {content}
    </div>
    """, unsafe_allow_html=True)
    
    if show_details and details:
        with st.expander("📊 View Processing Details"):
            st.json(details)


def process_audio_input(audio_bytes, pipeline):
    """Process audio input through the RAG pipeline"""
    try:
       
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        with st.spinner("🔄 Processing your query..."):
            results = pipeline.process_audio_query(temp_audio_path)
        
        os.remove(temp_audio_path)
        
        return results
    
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
        return None


def main():
    """Main Streamlit app"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎙️ RecallVision</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Voice-Enabled RAG Chatbot with Wikipedia Knowledge</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        asr_endpoint = st.text_input(
            "ASR Endpoint",
            value="http://localhost:8000/transcribe",
            help="URL of the ASR service"
        )
        
        db_path = st.text_input(
            "Vector DB Path",
            value="data/faiss_db",
            help="Path to FAISS database"
        )
        
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["groq", "cohere", "nvidia"],
            help="Choose your LLM provider"
        )
        
        st.markdown("---")
        
        # Mode selection
        demo_mode = st.checkbox("📝 Text Demo Mode (No ASR needed)", value=True)
        
        st.markdown("---")
        
        # Initialize button
        if st.button("🚀 Initialize Pipeline"):
            if demo_mode:
                initialize_pipeline("", db_path, llm_provider)
            else:
                initialize_pipeline(asr_endpoint, db_path, llm_provider)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main content area
    if not st.session_state.pipeline_initialized:
        st.info("👈 Please initialize the pipeline from the sidebar to get started!")
        
        # Show setup instructions
        st.markdown("""
        ### 🚀 Quick Start Guide
        
        1. **Enable Text Demo Mode** in sidebar (recommended)
        2. **Initialize Pipeline**: Click the button in the sidebar
        3. **Ask Questions**: Type your question in the chat below!
        
        **Sample Questions:**
        - What is artificial intelligence?
        - Tell me about machine learning
        - Explain deep learning
        """)
    
    else:
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(
                message["role"],
                message["content"],
                show_details=message.get("show_details", False),
                details=message.get("details")
            )
        
        # Text input option (always available)
        st.markdown("### 💬 Ask Your Question")
        
        text_query = st.text_input("Type your question here:", placeholder="e.g., What is machine learning?")
        
        if st.button("🚀 Submit Question") and text_query:
            try:
                with st.spinner("🔄 Processing..."):
                    # Retrieve context
                    context_chunks = st.session_state.pipeline.retrieve_context(text_query, top_k=2)
                    
                    # Generate answer
                    answer = st.session_state.pipeline.generate_answer(text_query, context_chunks)
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "user",
                        "content": text_query,
                        "show_details": False
                    })
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "show_details": True,
                        "details": {
                            "query": text_query,
                            "context_chunks": context_chunks
                        }
                    })
                    
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
        
        st.markdown("---")
        
        # Audio input options (optional)
        with st.expander("🎤 Advanced: Audio Input (Requires ASR Service)"):
            st.warning("⚠️ Audio features require ASR service running on port 8000")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Option 1: Record Audio**")
                audio_bytes = audio_recorder(
                    text="Click to record",
                    recording_color="#667eea",
                    neutral_color="#6aa36f",
                    icon_name="microphone",
                    icon_size="2x"
                )
                
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    
                    if st.button("🚀 Process Recording"):
                        results = process_audio_input(audio_bytes, st.session_state.pipeline)
                        
                        if results:
                            st.session_state.messages.append({
                                "role": "user",
                                "content": results["transcription"],
                                "show_details": False
                            })
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": results["answer"],
                                "show_details": True,
                                "details": {
                                    "transcription": results["transcription"],
                                    "translation": results["translation"],
                                    "context_chunks": results["context"]
                                }
                            })
                            
                            st.rerun()
            
            with col2:
                st.markdown("**Option 2: Upload Audio File**")
                uploaded_file = st.file_uploader(
                    "Choose an audio file",
                    type=["wav", "mp3", "m4a", "ogg"],
                    help="Upload an audio file with your question"
                )
                
                if uploaded_file:
                    st.audio(uploaded_file)
                    
                    if st.button("🚀 Process Upload"):
                        audio_bytes = uploaded_file.read()
                        results = process_audio_input(audio_bytes, st.session_state.pipeline)
                        
                        if results:
                            st.session_state.messages.append({
                                "role": "user",
                                "content": results["transcription"],
                                "show_details": False
                            })
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": results["answer"],
                                "show_details": True,
                                "details": {
                                    "transcription": results["transcription"],
                                    "translation": results["translation"],
                                    "context_chunks": results["context"]
                                }
                            })
                            
                            st.rerun()


if __name__ == "__main__":
    main()
