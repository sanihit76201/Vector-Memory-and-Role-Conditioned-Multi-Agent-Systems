"""Vector-based episodic memory with semantic retrieval."""

from collections import deque
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseMemory


class VectorEpisodicMemory(BaseMemory):
    """
    Semantic similarity-based memory with vector embeddings.
    
    Stores reflections with semantic embeddings and retrieves
    the most semantically similar reflections using cosine similarity.
    """
    
    def __init__(self, llm, max_size: int = 100):
        """
        Initialize vector episodic memory.
        
        Args:
            llm: BaseLLMModel instance with get_embedding() method
            max_size: Maximum number of reflections to store
        """
        self.llm = llm
        self.reflections = deque(maxlen=max_size)
        self.embeddings = deque(maxlen=max_size)
    
    def add_reflection(self, reflection: str):
        """Add reflection with its semantic embedding."""
        emb = self.llm.get_embedding(reflection)
        self.reflections.append(reflection)
        self.embeddings.append(emb)
    
    def get_relevant_memories(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve top-k most semantically similar reflections.
        
        Args:
            query: Query text for semantic similarity search
            k: Number of similar reflections to retrieve
        """
        if not self.reflections:
            return []
        
        q_emb = self.llm.get_embedding(query)
        sims = cosine_similarity(
            q_emb.reshape(1, -1), 
            np.array(list(self.embeddings))
        )[0]
        top_k = np.argsort(sims)[-k:]
        
        return [list(self.reflections)[i] for i in sorted(top_k)]
    
    def clear(self):
        """Clear all reflections and embeddings."""
        self.reflections.clear()
        self.embeddings.clear()
    
    def __len__(self) -> int:
        return len(self.reflections)
