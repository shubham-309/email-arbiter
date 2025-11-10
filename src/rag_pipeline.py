import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
from src.utils import get_embedding, client
from src.data_layer import normalize_to_dict  # Not needed, but for type

DIMENSION = 1536  # text-embedding-3-small

class RAGIndex:
    def __init__(self, index_path: str = "outputs/rag_index.faiss"):
        self.index_path = index_path
        self.index = None
        self.documents = []  # List of (thread_root, thread_text)
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.index_path + '.pkl', 'rb') as f:
                self.documents = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(DIMENSION)

    def add_documents(self, full_threads: Dict[str, List[Dict]], emails: List[Dict]):
        """Explicit: Use passed full_threads and emails."""
        for root, thread_emails in full_threads.items():
            thread_text = " ".join([e['body_clean'] for e in thread_emails])
            emb = get_embedding(thread_text)
            self.index.add(np.array([emb], dtype=np.float32))
            self.documents.append((root, thread_text))
        self._persist()

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + '.pkl', 'wb') as f:
            pickle.dump(self.documents, f)

    def query(self, question: str, k: int = 3) -> str:
        if self.index is None or len(self.documents) == 0:
            return "No index built yet."
        q_emb = np.array([get_embedding(question)], dtype=np.float32)
        distances, indices = self.index.search(q_emb, k)
        context = "\n".join([self.documents[i][1] for i in indices[0] if i < len(self.documents)])
        prompt = f"Answer based on context: {context}\nQuestion: {question}"
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content