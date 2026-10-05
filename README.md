# NCERT Class 10 Science Assistant (AI Intern Assignment)

## 1. Local Execution Setup
Follow these steps to run the complete pipeline and application locally.

1. **Clone the repository:**
   ```bash
   git clone <https://github.com/Vipul-Intellect/-Production-Chatbot-with-Smart-Caching-For-NCERT-Class-10-Science>
   cd -Production-Chatbot-with-Smart-Caching-For-NCERT-Class-10-Science
   ```

2. **Set up Python Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Rename `.env.example` to `.env` and insert your Gemini API Key.
   ```env
   GEMINI_API_KEY="your_api_key_here"
   GEMINI_MODEL="gemini-3.5-flash-lite"
   ```

4. **Data Ingestion & Embedding:**
   Build the SQLite database and FAISS index from the NCERT PDFs located in `data/pdf/`.
   ```bash
   python scripts/ingest_pdfs.py
   python scripts/embed_chunks.py
   ```

5. **(Optional) Cache Pre-seeding:**
   To pre-populate the Smart Semantic Cache offline with generated questions:
   ```bash
   python scripts/seed_cache.py
   ```

6. **Run the Frontend (Streamlit):**
   The primary Streamlit app uses a monolithic architecture to fit within cloud free-tier RAM limits and provides the conversational UI.
   ```bash
   streamlit run app.py
   ```

7. **(Optional) Run the Backend API (FastAPI):**
   If you wish to test the API endpoints (`POST /session`, `POST /chat`) programmatically as required by the assignment specification.
   ```bash
   uvicorn main:app --reload
   ```

---

## 2. One-Page Explainer: Architecture & Approach

### 2.1 How the Chatbot and Cache Work
The chatbot operates on a two-stage Retrieval-Augmented Generation (RAG) pipeline combined with a **Smart Semantic Cache**.
- **Smart Semantic Cache:** Before the expensive RAG pipeline executes, the system embeds the user query and searches a secondary **Cache FAISS Index**. If a semantically similar question is found (e.g., "What is refraction?" matches "Define refraction"), the system validates it against strict safety rules (numerical parameters and contradictory terms). If it passes, it returns the cached response instantly (< 100ms) from Redis (or a graceful SQLite fallback), entirely bypassing the LLM. 
- **Offline Pre-seeding:** The cache is pre-populated using a strict Dual-Model architecture. A Generator model pulls NCERT chunks to draft questions, and a Validator model strictly reviews them for zero hallucination before injecting them into the cache.
- **RAG Pipeline:** If a Cache Miss occurs, we retrieve the Top-15 child chunks from FAISS, rerank them using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`), and fetch the final Top-6 parent contexts from our SQLite database. This context, alongside chat history, is sent to `gemini-3.5-flash-lite` to generate a strictly cited answer (e.g., `[C1]`), which is then dynamically written back to the Cache for future users.

### 2.2 What We Cache and What We Never Cache
The cache is designed to prioritize extreme safety over broad matching.

**What we CACHE (and serve):**
- **Semantically Equivalent Questions:** Exact matches and paraphrases (e.g., "What is refraction?" and "What does refraction mean?") successfully hit the cache.
- **Strictly Grounded Answers:** We only dynamically write to the cache if the LLM generated an answer that successfully used the provided NCERT citations.

**What we NEVER CACHE (and force a cache miss):**
- **Differing Numerical Parameters:** The safety filter strictly extracts numbers using regex. A question like "Focal length when R = 20 cm" will safely MISS the cache for "Focal length when R = 30 cm".
- **Contradictory Terms:** A safety regex prevents dangerous conceptual collisions, ensuring "Image by a concave mirror" NEVER matches "Image by a convex mirror".
- **Conversation-dependent Follow-ups:** The system detects contextual phrases (e.g., "Explain it more simply", "What about its laws?") and completely bypasses the cache to preserve conversational flow.

### 2.3 What Didn't Work (Challenges & Iterations)
- **Redis Availability & Cloud Limits:** The assignment architecture relies on Redis for ultra-fast, sub-500ms cache payload retrieval. However, running local Redis on Windows or deploying it freely on platforms like Render presented connectivity issues (e.g., Error 10061). To guarantee the Semantic Cache remained 100% functional, we engineered a **robust SQLite Fallback mechanism** that safely stores and retrieves cache payloads directly from `data/rag.db` if Redis is unreachable.
- **FAISS vs SQLite Separation:** Initially, we tried storing massive parent chunk texts directly inside FAISS metadata. This bloated RAM usage. We migrated to a dual-store architecture: FAISS only holds lightweight vectors for fast mathematical similarity search, while SQLite acts as the robust storage for heavy text.
- **Vector Query Pollution:** When maintaining conversation history, prepending the entire chat history to the FAISS search string mathematically ruined the semantic vector. The fix was isolating the FAISS search to *only* use the raw user question, and injecting chat history immediately after retrieval, solely for the LLM.

### 2.4 Request Flowchart
```text
User Question
      |
      v
Smart Cache Lookup (FAISS Semantic Match + Safety Rules)
      |
   +--+--+
   |     |
 HIT   MISS
   |     |
   |     v
   |   FAISS Vector Search (Child Chunks)
   |     |
   |     v
   |   Cross-Encoder Reranking
   |     |
   |     v
   |   SQLite Context Fetch (Parent Chunks)
   |     |
   |     v
   |   Gemini 3.5 Flash-Lite (LLM)
   |     |
   |     v
   |   Citation & Grounding Validation
   |     |
   |   +--+--+
   |   |     |
   | PASS   FAIL
   |   |     |
   |   v     v
   | Write   Discard (Return Error)
   | to Cache
   |   |
   +---+
    |
    v
 Final Answer + Citations
```


what validation?
At minimum:
1. Grounding — answer is supported by retrieved NCERT context.
2. Citation validity — cited chapter/section actually contains the supporting information.
3. Question-answer relevance — answer actually answers the user's question.
4. No unsupported claims — reject if the LLM adds outside information.
5. Cache suitability — don't cache highly contextual/follow-up questions that depend on previous conversation.
