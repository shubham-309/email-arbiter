import streamlit as st
from src.workflow import graph

st.title("Email Agent Demo")
query = st.text_input("Ask a question:")
if st.button("Query"):
    initial_state = {"query": query}
    result = graph.invoke(initial_state)
    st.write(result["response"])

# Sidebar: Run pipeline
if st.sidebar.button("Rebuild Index"):
    st.sidebar.success("Pipeline run!")