import os
import json
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from rag_app.generation.generator import GeminiRestClient
from rag_app.cache.semantic_cache import write_to_semantic_cache
from rag_app.storage.sqlite_store import get_sqlite_connection
from rag_app.observability.logger import logger

def get_parents(limit=10):
    conn = get_sqlite_connection()
    cur = conn.cursor()
    # Randomly select chunks to avoid hitting the same ones
    cur.execute("SELECT * FROM parents WHERE status = 'ACTIVE' ORDER BY RANDOM() LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def generate_qa_pairs(client, model, content):
    system = "You are an AI that generates exactly one high-quality Question and Answer pair based on the NCERT text. Output strictly in JSON format: {\"question\": \"...\", \"answer\": \"...\"}."
    try:
        response = client.generate_content(
            model=model,
            contents=content,
            system_instruction=system
        )
        text = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
    except Exception as e:
        logger.error(f"Generation failed: {e}")
    return None

def validate_qa_pair(client, model, question, answer, context):
    system = """You are a validation model. Evaluate if the answer is grounded in context, answers the question, and has no outside claims. 
Respond ONLY with 'PASS' or 'FAIL'."""
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    try:
        response = client.generate_content(
            model=model,
            contents=prompt,
            system_instruction=system
        )
        text = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        return "PASS" in text.upper()
    except Exception as e:
        logger.error(f"Validation failed: {e}")
    return False

def seed():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")
        
    client = GeminiRestClient(api_key)
    gen_model = os.getenv("QUESTION_GENERATION_MODEL", "gemini-3.5-flash-lite")
    val_model = os.getenv("VALIDATION_MODEL", "gemini-3.5-flash-lite")
    
    logger.info(f"Starting Cache Seeding... Gen Model: {gen_model} | Val Model: {val_model}")
    
    parents = get_parents(10)
    for parent in parents:
        content = parent["text"]
        
        import json
        try:
            metadata = json.loads(parent.get("metadata", "{}"))
        except:
            metadata = {}
            
        source_file = metadata.get("source_file", "Unknown Chapter")
        
        qa = generate_qa_pairs(client, gen_model, content)
        if not qa:
            continue
            
        q = qa.get("question")
        a = qa.get("answer")
        
        # Append Citation format
        a = f"{a} [C1]"
        
        is_valid = validate_qa_pair(client, val_model, q, a, content)
        if is_valid:
            logger.info(f"PASS: {q}")
            response_dict = {
                "reply": a,
                "citations": [{"label": "C1", "source": source_file}],
                "cache_hit": False
            }
            write_to_semantic_cache(q, response_dict)
        else:
            logger.warning(f"FAIL: {q}")

if __name__ == "__main__":
    seed()
