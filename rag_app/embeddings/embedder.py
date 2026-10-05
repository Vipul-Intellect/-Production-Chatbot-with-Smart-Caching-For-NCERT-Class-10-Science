from sentence_transformers import SentenceTransformer
import numpy as np

class ChunkEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
        
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Natively produce normalized embeddings which is standard for cosine similarity via L2 in FAISS.
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        return embeddings.tolist()
