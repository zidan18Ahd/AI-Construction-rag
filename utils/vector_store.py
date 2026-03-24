#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
    
    def add(self, embeddings, chunks):
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)
    
    def search(self, query_emb, k=3):
        distances, indices = self.index.search(query_emb.astype(np.float32), k)
        retrieved = [self.chunks[i] for i in indices[0]]
        return retrieved

