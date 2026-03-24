#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import os
from utils.chunking import chunk_documents
from utils.embeddings import EmbeddingModel
from utils.vector_store import VectorStore
from utils.llm import LocalLLM

def load_documents():
    docs = []
    for fname in os.listdir("data"):
        if fname.endswith(".md"):
            with open(f"data/{fname}", "r", encoding="utf-8") as f:
                docs.append({"text": f.read(), "source": fname})
    return docs

@st.cache_resource
def build_index():
    docs = load_documents()
    chunks = chunk_documents(docs)
    embed_model = EmbeddingModel()
    embeddings = embed_model.embed([c["text"] for c in chunks])
    vector_store = VectorStore(embeddings.shape[1])
    vector_store.add(embeddings, chunks)
    return embed_model, vector_store

def main():
    st.set_page_config(page_title="Construction Assistant", layout="wide")
    st.title(" Construction Assistant (RAG)")
    st.markdown("Ask about Indecimal's policies, packages, quality, or customer journey.")

    embed_model, vector_store = build_index()
    llm = LocalLLM()   # uses flan-t5-small by default

    query = st.text_input("Your question:", placeholder="e.g., What are the package prices?")

    if query:
        with st.spinner("Searching documents..."):
            q_emb = embed_model.embed([query])
            retrieved = vector_store.search(q_emb, k=3)

        st.subheader(" Retrieved Context")
        for i, chunk in enumerate(retrieved):
            st.markdown(f"**Chunk {i+1}** (from `{chunk['source']}`)")
            st.write(chunk["text"])
            st.divider()

        with st.spinner("Generating answer (local LLM)..."):
            context = "\n\n".join([c["text"] for c in retrieved])
            prompt = f"""You are an assistant for a construction marketplace. Answer the user's question **only** using the provided context. If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""
            answer = llm.generate(prompt)

        st.subheader("Generated Answer")
        st.write(answer)

if __name__ == "__main__":
    main()

