# NCERT Class 10 Science Assistant (AI Intern Assignment)

## 1. Local Execution Setup
Follow these steps to run the complete pipeline and application locally.

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
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

5. **Run the Frontend (Streamlit):**
   The primary Streamlit app uses a monolithic architecture to fit within cloud free-tier RAM limits and provides the conversational UI.
   ```bash
   streamlit run app.py
   ```

6. **(Optional) Run the Backend API (FastAPI):**
   If you wish to test the API endpoints (`POST /session`, `POST /chat`) programmatically as required by the assignment specification.
   ```bash
   uvicorn main:app --reload
   ```

---

## 2. One-Page Explainer: Architecture & Approach

### 2.1 How the Chatbot and Cache Work
The chatbot operates on a two-stage Retrieval-Augmented Generation (RAG) pipeline combined with a Question-Centric Semantic Cache.
- **RAG Pipeline:** When a question is asked, it is converted into a vector embedding using `sentence-transformers/all-MiniLM-L6-v2`. We retrieve the Top-15 child chunks from FAISS, rerank them using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`), and fetch the final Top-6 corresponding parent contexts from our SQLite database. This context, alongside the chat history, is sent to `gemini-3.5-flash-lite`. Gemini is strictly constrained by a system prompt to answer *only* from the provided context and must explicitly append citation labels (e.g., `[C1]`).
- **Semantic Smart Cache:** Before the expensive RAG pipeline executes, the system searches the cache (powered by SQLite) for previous identical or semantically identical questions. If a match is found, the system instantly returns the validated, cached response with `< 500ms` latency, entirely bypassing the LLM.

### 2.2 What We Cache and What We Never Cache
The cache is designed for extreme safety to prevent serving incorrect answers to similar-sounding but fundamentally different questions.

**What we CACHE (and serve):**
- **Semantically identical questions:** "What is refraction?" and "What does refraction mean?" will trigger a cache hit because the core intent is identical.
- **Fully Grounded Answers:** We only write to the cache if the LLM successfully generated an answer that passed the citation validation (i.e., it didn't hallucinate and properly cited the NCERT text).

**What we NEVER CACHE (and force a cache miss):**
- **Different numerical parameters:** The vector embeddings are highly sensitive. A question like "Focal length when R = 20 cm" vs "Focal length when R = 30 cm" will bypass the cache because the semantic overlap is not exact for the specific entities.
- **Conversation-dependent follow-ups:** Short follow-up queries like "What about its laws?" or "Explain it more simply" rely heavily on the previous conversation history. We do not cache these isolated phrases as global reusable answers because their meaning changes depending on who asks it and what was asked previously.

### 2.3 What Didn't Work (Challenges & Iterations)
- **FAISS vs SQLite Separation:** Initially, we tried storing the massive parent chunk text directly inside FAISS metadata. This bloated the RAM usage. We migrated to a dual-store architecture: FAISS only holds lightweight child-chunk vectors for fast mathematical similarity search, while SQLite acts as the robust storage for the heavy parent-chunk text and metadata.
- **Cloud Free-Tier RAM Limits:** The original architecture strictly separated FastAPI and Streamlit into two different cloud services. However, running a Cross-Encoder and FAISS on Render's 512MB free-tier caused out-of-memory (OOM) crashes. The solution was migrating to a monolithic Streamlit Community Cloud app (`app.py`), leveraging its generous 1GB RAM limit by importing the `orchestrator` directly into Streamlit, bypassing HTTP network latency and saving memory.
- **Vector Query Pollution:** When maintaining conversation history, prepending the entire chat history to the FAISS search string mathematically ruined the semantic vector, causing FAISS to retrieve irrelevant chunks. The fix was isolating the FAISS search to *only* use the raw user question, and injecting the chat history immediately after retrieval, solely for the LLM's context.

### 2.4 Request Flowchart
```text
User Question
      |
      v
Smart Cache Lookup (SQLite String/Vector Match)
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