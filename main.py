from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path

# Ensure env variables like GEMINI_API_KEY are loaded
load_dotenv(Path(__file__).parent / ".env")

from rag_app.orchestrator import chat, generate_session_id
from rag_app.storage.app_store import create_session
from rag_app.observability.logger import logger

app = FastAPI(title="NCERT Science RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/session")
def create_session_endpoint():
    session_id = generate_session_id()
    create_session(session_id)
    return {"session_id": session_id}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not req.session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")
        
    try:
        response = chat(req.session_id, req.message)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        logger.error(f"RuntimeError in API: {re}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.error(f"Unhandled exception in API: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
