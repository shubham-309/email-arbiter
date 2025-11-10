from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from src.data_layer import load_emails, normalize_to_dict
from src.thread_builder import build_thread_graph
from src.event_tagger import tag_events
from src.rag_pipeline import RAGIndex

class AgentState(TypedDict):
    emails_df: pd.DataFrame
    emails: List[Dict]
    threads: Dict[str, List[str]]
    tagged_threads: Dict[str, List[Dict]]
    rag_index: RAGIndex
    query: str
    response: str

def load_node(state: AgentState) -> AgentState:
    df = load_emails("data/raw/emails.jsonl")
    return {"emails_df": df, "emails": normalize_to_dict(df)}

def thread_node(state: AgentState) -> AgentState:
    threads = build_thread_graph(state["emails"])
    full_threads = {root: [next(e for e in state["emails"] if e["id"] in ids) for ids in [ [id for id in thread] ]] for root, thread in threads.items()}
    return {"threads": threads, "full_threads": full_threads}

def tag_node(state: AgentState) -> AgentState:
    tagged = {}
    for root, emails in state["full_threads"].items():
        tags = tag_events(emails)
        tagged[root] = [{"email": e, "tag": next(t["event"] for t in tags if t["id"] == e["id"])} for e in emails]
    return {"tagged_threads": tagged}

def rag_build_node(state: AgentState) -> AgentState:
    index = RAGIndex()
    index.add_documents(state["threads"])
    faiss.write_index(index.index, "outputs/rag_index.faiss")
    return {"rag_index": index}

def query_node(state: AgentState) -> AgentState:
    # Assume query in state; in prod, conditional edge
    response = state["rag_index"].query(state["query"])
    return {"response": response}

# Graph
workflow = StateGraph(AgentState)
workflow.add_node("load", load_node)
workflow.add_node("thread", thread_node)
workflow.add_node("tag", tag_node)
workflow.add_node("rag_build", rag_build_node)
workflow.add_node("query", query_node)

workflow.add_edge("load", "thread")
workflow.add_edge("thread", "tag")
workflow.add_edge("tag", "rag_build")
workflow.add_edge("rag_build", "query")
workflow.add_edge("query", END)

graph = workflow.compile()

# Run
if __name__ == "__main__":
    initial_state = {"query": "Who approved the paint issue proposal?"}
    result = graph.invoke(initial_state)
    print(result["response"])
    # Save threads
    with open("outputs/threads.json", "w") as f:
        json.dump(result["tagged_threads"], f, indent=2, default=str)