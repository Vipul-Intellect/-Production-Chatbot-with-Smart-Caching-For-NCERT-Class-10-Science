import faiss
import pickle
import numpy as np
from pathlib import Path

class FAISSIndex:
    def __init__(self, dimension: int, index_path: str | Path, mapping_path: str | Path):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.index = faiss.IndexFlatL2(dimension)
        self.index_to_child_id: dict[int, str] = {}
        self._load()
        
    def _load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.mapping_path.exists():
            with open(self.mapping_path, 'rb') as f:
                self.index_to_child_id = pickle.load(f)
                
    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.mapping_path, 'wb') as f:
            pickle.dump(self.index_to_child_id, f)
            
    def add_vectors(self, vectors: list[list[float]], child_ids: list[str]):
        if not vectors:
            return
        
        vectors_np = np.array(vectors, dtype=np.float32)
        start_idx = self.index.ntotal
        
        self.index.add(vectors_np)
        
        for i, child_id in enumerate(child_ids):
            self.index_to_child_id[start_idx + i] = child_id
