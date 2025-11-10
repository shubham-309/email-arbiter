import faiss
import numpy as np
from openai import OpenAI
from typing import List, Dict, Tuple

client = OpenAI()
dimension = 1536  # text-embedding-3-small

class RAGIndex:
    def __init__(self):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []  # List of (thread_id, email_dict)

    def add_documents(self, threads: Dict[str, List[Dict]]):
        for root, email_ids in threads.items():
            thread_text = " ".join([e['body_clean'] for e in [next(em for em in EMAILS if em['id'] == eid) for eid in email_ids]])  # EMAILS global or pass
            emb = get_embedding(thread_text)
            self.index.add(np.array([emb]))
            self.documents.append((root, thread_text))

    def query(self, question: str, k: int = 3) -> str:
        q_emb = get_embedding(question)
        distances, indices = self.index.search(np.array([q_emb]), k)
        context = "\n".join([self.documents[i][1] for i in indices[0]])
        prompt = f"Answer based on context: {context}\nQuestion: {question}"
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

# Global for POC; in prod, pass df
# EMAILS = load_emails('data/raw/emails.jsonl').to_dict('records')  # Load once