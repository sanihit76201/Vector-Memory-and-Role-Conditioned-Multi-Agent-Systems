"""Abstract base class for memory implementations."""

from abc import ABC, abstractmethod
from typing import List


class BaseMemory(ABC):
    """
    Abstract base class for memory modules.
    
    All memory implementations must inherit from this class.
    """
    
    @abstractmethod
    def add_reflection(self, reflection: str):
        """Store a reflection in memory."""
        pass
    
    @abstractmethod
    def get_relevant_memories(self, query: str = "", k: int = 3) -> List[str]:
        """Retrieve relevant memories."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all stored memories."""
        pass
    
    def __len__(self) -> int:
        """Return number of stored memories."""
        return 0
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(size={len(self)})"
