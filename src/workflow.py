from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import os
import json
import pickle
from src.data_layer import load_emails, normalize_to_dict
from src.thread_builder import build_thread_graph
from src.event_tagger import tag_events
from src.rag_pipeline import RAGIndex
from src.utils import client

class AgentState(TypedDict):
    emails_df: 'pd.DataFrame'  # Type hint
    emails: List[Dict]
    threads: Dict[str, List[str]]
    full_threads: Dict[str, List[Dict]]
    tagged_threads: Dict[str, List[Dict]]
    rag_index: RAGIndex
    query: str
    response: str

def load_node(state: AgentState) -> Dict[str, Any]:
    if os.path.exists("outputs/emails.pkl"):
        with open("outputs/emails.pkl", 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_emails("data/raw/emails.jsonl")
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/emails.pkl", 'wb') as f:
            pickle.dump(df, f)
    return {"emails_df": df, "emails": normalize_to_dict(df)}

def thread_node(state: AgentState) -> Dict[str, Any]:
    threads = build_thread_graph(state["emails"])
    # Fixed: Correct list comp for full_threads
    full_threads = {
        root: [next(e for e in state["emails"] if e["id"] == tid) for tid in thread]
        for root, thread in threads.items()
    }
    # Cache threads
    with open("outputs/threads.json", "w") as f:
        json.dump(threads, f, indent=2, default=str)
    return {"threads": threads, "full_threads": full_threads}

def tag_node(state: AgentState) -> Dict[str, Any]:
    tagged = {}
    for root, emails in state["full_threads"].items():
        tags = tag_events(emails)
        # Map tags back to emails
        tag_dict = {t["id"]: t["event"] for t in tags}
        tagged[root] = [{"email": e, "tag": tag_dict[e["id"]]} for e in emails]
    # Cache tagged
    with open("outputs/tagged_threads.json", "w") as f:
        json.dump(tagged, f, indent=2, default=str)
    return {"tagged_threads": tagged}

def rag_build_node(state: AgentState) -> Dict[str, Any]:
    index = RAGIndex()
    if len(index.documents) == 0:  # Only build if not cached
        index.add_documents(state["full_threads"], state["emails"])
    return {"rag_index": index}

def query_node(state: AgentState) -> Dict[str, Any]:
    if "query" not in state or not state["query"]:
        return {"response": "No query provided."}
    response = state["rag_index"].query(state["query"])
    return {"response": response}

# Graph Setup
workflow = StateGraph(AgentState)

# Add nodes (return dicts for merge)
workflow.add_node("load", load_node)
workflow.add_node("thread", thread_node)
workflow.add_node("tag", tag_node)
workflow.add_node("rag_build", rag_build_node)
workflow.add_node("query", query_node)

# Edges (linear flow; add conditional for query if needed)
workflow.set_entry_point("load")
workflow.add_edge("load", "thread")
workflow.add_edge("thread", "tag")
workflow.add_edge("tag", "rag_build")
workflow.add_edge("rag_build", "query")
workflow.add_edge("query", END)

app = workflow.compile()

# Run Example
if __name__ == "__main__":
    initial_state = {"query": "Who approved the paint issue proposal?"}
    result = app.invoke(initial_state)
    print("Response:", result["response"])
    print("Threads cached:", os.path.exists("outputs/threads.json"))