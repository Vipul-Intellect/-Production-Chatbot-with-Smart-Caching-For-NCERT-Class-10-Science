import streamlit as st
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="NCERT Science Assistant", page_icon="🔬")

st.title("NCERT Class 10 Science Assistant")

# Initialize session
if "session_id" not in st.session_state:
    try:
        resp = requests.post(f"{API_URL}/session", timeout=5)
        resp.raise_for_status()
        st.session_state.session_id = resp.json()["session_id"]
        st.session_state.messages = []
    except requests.exceptions.RequestException:
        st.error(f"Could not connect to the backend API at {API_URL}.")
        st.stop()

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

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Searching NCERT database..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat", 
                    json={"session_id": st.session_state.session_id, "message": prompt},
                    timeout=60
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    
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
                else:
                    st.error(f"API Error: {resp.status_code} - {resp.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error("Failed to communicate with the backend API.")
