import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStore:
    def __init__(self, embedding_model: OpenAIEmbeddings, vector_db_path: str = "vector_db"):
        self.embedding_model = embedding_model
        self.vector_db_path = vector_db_path
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Create directory if it doesn't exist
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Load existing index if available
        self.load_index()
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(texts)
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Initialize or update FAISS index
        if self.index is None:
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend(metadatas)
        
        # Save updated index
        self.save_index()
    
    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(distances[0][i])
                })
        
        return results
    
    def save_index(self):
        """Save FAISS index and metadata"""
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.vector_db_path, "index.faiss"))
        
        with open(os.path.join(self.vector_db_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(self.vector_db_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
    
    def load_index(self):
        """Load existing FAISS index and metadata"""
        index_path = os.path.join(self.vector_db_path, "index.faiss")
        docs_path = os.path.join(self.vector_db_path, "documents.pkl")
        meta_path = os.path.join(self.vector_db_path, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)