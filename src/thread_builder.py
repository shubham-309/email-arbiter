import networkx as nx
from openai import OpenAI
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

def build_thread_graph(emails: List[Dict]) -> Dict[str, List[str]]:
    """Build reply graph using references/in_reply_to."""
    G = nx.DiGraph()
    for email in emails:
        G.add_node(email['id'])
        if email['in_reply_to']:
            G.add_edge(email['in_reply_to'], email['id'])
        for ref in email['references']:
            if ref != email['id']:
                G.add_edge(ref, email['id'], weight=0.5)  # Lower weight for refs

    # LLM Arbitration for ambiguities (e.g., orphan emails)
    orphans = [n for n in G.nodes if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    for orphan in orphans:
        best_match = arbitrate_with_llm(emails, orphan)
        if best_match:
            G.add_edge(best_match, orphan, weight=0.8)

    # Extract threads (weakly connected components)
    threads = {}
    for component in nx.weakly_connected_components(G):
        root = min(component, key=lambda x: next(e['date'] for e in emails if e['id'] == x))
        thread_emails = [e['id'] for e in emails if e['id'] in component]
        sorted_thread = sorted(thread_emails, key=lambda x: next(e['date'] for e in emails if e['id'] == x))
        threads[root] = sorted_thread
    return threads

def arbitrate_with_llm(emails: List[Dict], orphan_id: str) -> Optional[str]:
    """Use OpenAI to infer parent via semantic match."""
    orphan = next(e for e in emails if e['id'] == orphan_id)
    candidates = [e for e in emails if e['date'] < orphan['date']]
    if not candidates:
        return None
    
    # Embed subjects/bodies
    orphan_emb = get_embedding(orphan['subject'] + ' ' + orphan['body_clean'])
    cand_embs = [get_embedding(c['subject'] + ' ' + c['body_clean']) for c in candidates]
    similarities = cosine_similarity([orphan_emb], cand_embs)[0]
    best_idx = np.argmax(similarities)
    
    # LLM confirm if similarity > 0.7
    if similarities[best_idx] > 0.7:
        prompt = f"Is email {orphan_id} a reply to {candidates[best_idx]['id']}? Subjects: {orphan['subject']} vs {candidates[best_idx]['subject']}. Yes/No."
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        if "Yes" in response.choices[0].message.content:
            return candidates[best_idx]['id']
    return None

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding