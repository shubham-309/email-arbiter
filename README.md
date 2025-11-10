# Email Agent — LLM Arbiter

## Overview
Simplified AI for email thread reconstruction, event tagging, and RAG QA.

## Setup
1. `pip install -r requirements.txt`
2. Set `OPENAI_API_KEY` in `.env`
3. `python src/workflow.py` to run pipeline
4. `streamlit run app.py` for demo

## Usage
- Outputs: `outputs/threads.json` (tagged threads)
- Query example: "What was the decision on supplier change?"

## Architecture
See diagram in guideline. LangGraph for flow, OpenAI for smarts.

## Limitations
- Local FAISS; scale to Pinecone.
- Assumes <1k emails; chunk for larger.
