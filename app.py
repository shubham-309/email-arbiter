import streamlit as st
from src.workflow import app

st.title("Email Agent Demo")
query = st.text_input("Ask a question:")
if st.button("Query"):
    initial_state = {"query": query}
    result = app.invoke(initial_state)
    st.write(result["response"])

if st.sidebar.button("Rebuild Cache"):
    for f in ["outputs/emails.pkl", "outputs/threads.json", "outputs/tagged_threads.json", "outputs/rag_index.faiss*"]:
        if os.path.exists(f):
            os.remove(f)
    st.sidebar.success("Cache cleared! Run query to rebuild.")