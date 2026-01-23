"""
RecallVision - Interactive Chat UI
Beautiful Streamlit interface for voice-enabled RAG chatbot
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import sys

# add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag import RAGPipeline
from src.utils import get_logger

logger = get_logger(__name__)

# page config
st.set_page_config(
    page_title="RecallVision Chat",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom CSS for modern chat UI
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* User message */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        margin-left: 20%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Assistant message */
    .assistant-message {
        background: #f7f7f8;
        color: #2d3748;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: 20%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Context preview */
    .context-preview {
        background: #fff5e6;
        border-left: 4px solid #ffa500;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 0.9em;
        color: #666;
    }
    
    /* Header */
    .app-header {
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f7f7f8;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'pipeline' not in st.session_state:
    with st.spinner("🔄 Initializing RAG pipeline..."):
        try:
            # explicitly set ASR endpoint to port 8001
            st.session_state.pipeline = RAGPipeline(asr_endpoint="http://localhost:8001/transcribe")
            st.success("✅ Pipeline initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize pipeline: {str(e)}")
            st.stop()

# header
st.markdown("""
<div class="app-header">
    <h1>🎙️ RecallVision Chat</h1>
    <p style="color: #666; margin: 0;">Voice-Enabled RAG Assistant</p>
</div>
""", unsafe_allow_html=True)

# sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # input mode
    input_mode = st.radio(
        "Input Mode",
        ["🎤 Audio Upload", "📝 Text Input", "🔴 Record Audio"],
        index=0
    )
    
    st.markdown("---")
    
    # model info
    st.markdown("### 📊 Model Info")
    st.info(f"""
    **LLM**: NVIDIA Llama 3.1
    **Embeddings**: MiniLM-L6-v2
    **ASR**: Whisper Base
    **Database**: 167 chunks
    """)
    
    st.markdown("---")
    
    # clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Upload audio files (.wav, .mp3, .m4a)
    - Ask about Python or AI
    - View retrieved context
    """)

# main chat area
chat_container = st.container()

# input area
st.markdown("---")

if input_mode == "🎤 Audio Upload":
    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
        help="Upload an audio file with your question"
    )
    
    if uploaded_file and st.button("🚀 Process Audio", use_container_width=True):
        with st.spinner("🎧 Processing your audio..."):
            # save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                # process through RAG pipeline
                results = st.session_state.pipeline.process_audio_query(tmp_path)
                
                # add to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': results['transcription'],
                    'language': results.get('detected_language', 'unknown')
                })
                
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': results['answer'],
                    'context': results['context_chunks']
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing audio: {str(e)}")
            finally:
                # cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

elif input_mode == "📝 Text Input":
    text_input = st.text_area(
        "Type your question",
        placeholder="Ask me anything about Python or Artificial Intelligence...",
        height=100
    )
    
    if st.button("💬 Send", use_container_width=True) and text_input:
        with st.spinner("🤔 Thinking..."):
            try:
                # process through RAG pipeline
                results = st.session_state.pipeline.process_text_query(text_input)
                
                # add to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': text_input
                })
                
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': results['answer'],
                    'context': results['context_chunks']
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

elif input_mode == "🔴 Record Audio":
    st.info("🎤 Audio recording feature")
    
    # check if audio-recorder-streamlit is available
    try:
        from audio_recorder_streamlit import audio_recorder
        
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#667eea",
            icon_name="microphone",
            icon_size="3x"
        )
        
        if audio_bytes and st.button("🚀 Process Recording", use_container_width=True):
            with st.spinner("🎧 Processing your recording..."):
                # save recording temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    # process through RAG pipeline
                    results = st.session_state.pipeline.process_audio_query(tmp_path)
                    
                    # add to chat history
                    st.session_state.chat_history.append({
                        'type': 'user',
                        'content': results['transcription'],
                        'language': results.get('detected_language', 'unknown')
                    })
                    
                    st.session_state.chat_history.append({
                        'type': 'assistant',
                        'content': results['answer'],
                        'context': results['context_chunks']
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing recording: {str(e)}")
                finally:
                    # cleanup temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
    
    except ImportError:
        st.warning("⚠️ Audio recording requires `audio-recorder-streamlit` package")
        st.code("pip install audio-recorder-streamlit", language="bash")

# display chat history
with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 50px; color: #666;">
            <h3>👋 Welcome to RecallVision!</h3>
            <p>Upload an audio file or type a question to get started.</p>
            <p>I can answer questions about <strong>Python</strong> and <strong>Artificial Intelligence</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg['type'] == 'user':
                lang_badge = f" ({msg['language']})" if 'language' in msg else ""
                st.markdown(f"""
                <div class="user-message">
                    <strong>You{lang_badge}:</strong><br>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            
            elif msg['type'] == 'assistant':
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 RecallVision:</strong><br>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # show context in expander
                if 'context' in msg and msg['context']:
                    with st.expander("📚 View Retrieved Context"):
                        for i, chunk in enumerate(msg['context'], 1):
                            st.markdown(f"""
                            <div class="context-preview">
                                <strong>Chunk {i}:</strong><br>
                                {chunk[:300]}{'...' if len(chunk) > 300 else ''}
                            </div>
                            """, unsafe_allow_html=True)

# footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Powered by <strong>Whisper</strong> • <strong>FAISS</strong> • <strong>NVIDIA LLM</strong></p>
</div>
""", unsafe_allow_html=True)
