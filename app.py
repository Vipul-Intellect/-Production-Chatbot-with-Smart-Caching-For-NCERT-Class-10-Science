import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path

# Load env variables (for local testing, Streamlit Cloud uses Secrets)
load_dotenv(Path(__file__).parent / ".env")

# We import the orchestrator directly! No FastAPI, no Render needed!
from rag_app.orchestrator import chat, generate_session_id
from rag_app.storage.app_store import create_session

st.set_page_config(page_title="NCERT Science Assistant", page_icon="🔬")

st.title("NCERT Class 10 Science Assistant")

# Initialize session locally instead of hitting an API
if "session_id" not in st.session_state:
    session_id = generate_session_id()
    create_session(session_id)
    st.session_state.session_id = session_id
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    if st.button("New Conversation"):
        st.session_state.clear()
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            st.caption(msg["meta"])

# Chat input
if prompt := st.chat_input("Ask a question about NCERT Class 10 Science..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response Directly (No API Call)
    with st.chat_message("assistant"):
        with st.spinner("Searching NCERT database & running AI models..."):
            try:
                # Call the orchestrator directly!
                data = chat(st.session_state.session_id, prompt)
                
                reply = data.get("reply", "Error")
                cache_hit = data.get("cache_hit", False)
                latency = data.get("latency_ms", 0)
                citations = data.get("citations", [])
                
                # Format citations
                c_text = "  \n".join([f"**{c['label']}**: {c['source']}" for c in citations])
                
                st.markdown(reply)
                
                meta_str = f"Cache: {'HIT' if cache_hit else 'MISS'} | Latency: {latency} ms"
                if c_text:
                    meta_str += f"  \n\n**Sources:**  \n{c_text}"
                    
                st.caption(meta_str)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "meta": meta_str
                })
                    
            except Exception as e:
                st.error(f"An internal error occurred: {e}")
