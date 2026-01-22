# Simple demo without ASR requirement

import streamlit as st
import os
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="RecallVision Demo", page_icon="🎙️", layout="wide")

st.title("🎙️ RecallVision - RAG Demo")
st.caption("Voice-Enabled RAG Chatbot (Text Demo Mode)")

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    db_path = st.text_input("Vector DB Path", value="data/faiss_db")
    llm_provider = st.selectbox("LLM Provider", ["nvidia", "groq", "cohere"], index=0)
    
    st.markdown("---")
    st.info("💡 **Demo Mode**: Enter text queries directly (no ASR needed)")
    
    if st.button("🚀 Initialize Pipeline"):
        try:
            with st.spinner("Loading..."):
                # Initialize without ASR endpoint
                st.session_state.pipeline = RAGPipeline(
                    asr_endpoint="",  # Not needed for text demo
                    vector_db_path=db_path,
                    llm_provider=llm_provider
                )
            st.success("✅ Pipeline ready!")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main area
if st.session_state.pipeline is None:
    st.info("👈 Click 'Initialize Pipeline' in the sidebar to start")
    
    st.markdown("""
    ### 🚀 Quick Start
    
    1. **Initialize Pipeline** - Click the button in sidebar
    2. **Ask Questions** - Type your question below
    3. **Get Answers** - Based on the Wikipedia article about AI
    
    **Sample Questions:**
    - What is artificial intelligence?
    - Tell me about machine learning
    - What is deep learning?
    - Explain computer vision
    """)
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "details" in msg:
                with st.expander("📊 Details"):
                    st.json(msg["details"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about AI..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve context
                    context_chunks = st.session_state.pipeline.retrieve_context(prompt, top_k=2)
                    
                    # Generate answer
                    answer = st.session_state.pipeline.generate_answer(prompt, context_chunks)
                    
                    st.write(answer)
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "details": {"context": context_chunks}
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")
