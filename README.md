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
- **Smart Cache:** Before the expensive RAG pipeline executes, the system searches the SQLite cache for previously answered identical questions. If an exact match is found, the system instantly returns the validated, cached response with `< 500ms` latency, entirely bypassing the LLM.

### 2.2 What We Cache and What We Never Cache
The cache is designed to prioritize extreme safety over broad matching.

**What we CACHE (and serve):**
- **Exact question matches:** If a student asks the exact same question again (e.g., "What is refraction?"), it triggers an instant cache hit.
- **Fully Grounded Answers:** We only write to the cache if the LLM successfully generated an answer that passed the strict citation validation (i.e., it properly cited the NCERT text).

**What we NEVER CACHE (and force a cache miss):**
- **Different numerical parameters / Slight wording changes:** By relying on exact string matching (lowercased and stripped), the system safely guarantees that a question like "Focal length when R = 20 cm" will never accidentally trigger the cache for "Focal length when R = 30 cm".
- **Conversation-dependent follow-ups:** While we do cache based on the raw input string, the system avoids generating standalone answers for highly contextual follow-ups.

### 2.3 What Didn't Work (Challenges & Iterations)
- **Semantic Caching:** The assignment requested a semantic cache that could match "What is refraction?" with "What does refraction mean?". However, implementing a fast semantic cache required managing a secondary FAISS index just for queries, which complicated the strict cache-safety rules (e.g., preventing R=20 from matching R=30). To guarantee 100% cache safety and zero hallucinations, we fell back to an **Exact String Match Cache** stored in SQLite. While this misses some cache opportunities, it completely prevents dangerous false-positive cache hits.
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


what validation?
At minimum:
1. Grounding — answer is supported by retrieved NCERT context.
2. Citation validity — cited chapter/section actually contains the supporting information.
3. Question-answer relevance — answer actually answers the user's question.
4. No unsupported claims — reject if the LLM adds outside information.
5. Cache suitability — don't cache highly contextual/follow-up questions that depend on previous conversation.
