"""
RecallVision - Prompt Templates
RAG prompt engineering for accurate, grounded responses
"""

from typing import List


def create_rag_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Create RAG prompt with retrieved context.
    
    Args:
        query: User question
        context_chunks: Retrieved context chunks
        
    Returns:
        Formatted prompt string
    """
    context = "\n\n".join([f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
    
    prompt = f"""You are a helpful AI assistant. Answer the question based STRICTLY on the provided context.

IMPORTANT RULES:
1. Use ONLY the information from the context below
2. Do NOT use external knowledge or make assumptions
3. If the context doesn't contain enough information to answer, say: "I don't have enough information from the retrieved documents to answer this question."
4. Be concise and accurate
5. Cite which context section you're using when relevant

Context:
{context}

Question: {query}

Answer: Provide a clear, concise answer based only on the information in the context above."""
    
    return prompt


def create_simple_prompt(query: str, context: str) -> str:
    """
    Create a simple RAG prompt.
    
    Args:
        query: User question
        context: Combined context
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer: Provide a clear, concise answer based only on the information in the context. If the context doesn't contain enough information, say so."""
    
    return prompt
