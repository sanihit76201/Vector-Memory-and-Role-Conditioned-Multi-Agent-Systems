"""Temporal (recency-based) memory implementation."""

from collections import deque
from typing import List
from .base import BaseMemory


class TemporalMemory(BaseMemory):
    """
    FIFO sliding window memory (recency-based retrieval).
    
    Stores reflections in chronological order and retrieves
    the most recent k reflections.
    """
    def __init__(self, max_size: int = 10):
        """
        Initialize temporal memory.
        
        Args:
            max_size: Maximum number of reflections to store
        """
        self.reflections = deque(maxlen=max_size)
    
    def add_reflection(self, reflection: str):
        """Add reflection to end of queue."""
        self.reflections.append(reflection)
    
    def get_relevant_memories(self, query: str = "", k: int = 3) -> List[str]:
        """
        Return last k reflections (most recent).
        
        Args:
            query: Ignored (recency-based, not semantic)
            k: Number of recent reflections to return
        """
        return list(self.reflections)[-k:]
    
    def clear(self):
        """Clear all reflections."""
        self.reflections.clear()
    
    def __len__(self) -> int:
        return len(self.reflections)
