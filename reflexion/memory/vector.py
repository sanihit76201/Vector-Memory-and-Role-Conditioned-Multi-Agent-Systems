"""Vector-based episodic memory with semantic retrieval and configurable
retention policy.

── TMLR REVISION — retention-policy ablation ───────────────────────────────
Previously the only retention mechanism was FIFO (deque(maxlen=N)), and the
only thing ever varied was capacity (max_size) and retrieval count (k) —
never the eviction MECHANISM itself. This patch adds two alternative
mechanisms, selectable via retention_policy:

  'fifo'       (default, UNCHANGED) — evict oldest-inserted entry when full.
               Uses the original deque(maxlen=N) auto-eviction exactly as
               before; this path is untouched so all existing published
               results are reproduced exactly.

  'lru'        — evict the entry LEAST RECENTLY RETRIEVED (not least
               recently inserted). An entry's "last used" timestamp is
               updated whenever it appears in a get_relevant_memories()
               result. Tests whether protecting frequently-useful memories
               from eviction (regardless of insertion age) helps.

  'importance' — evict the LOWEST-importance_score entry (ties broken by
               insertion age, oldest first). add_reflection() accepts an
               optional importance_score (default 0.0); callers can tag
               reflections they know are valuable (e.g. ones that preceded
               a success) to protect them from eviction.

retention_policy defaults to 'fifo' everywhere, so VectorEpisodicMemory(llm)
with no extra args is unchanged.
"""

from collections import deque
from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseMemory


class VectorEpisodicMemory(BaseMemory):
    """
    Semantic similarity-based memory with vector embeddings.

    Stores reflections with semantic embeddings and retrieves
    the most semantically similar reflections using cosine similarity.

    Args:
        llm: BaseLLMModel instance with get_embedding() method
        max_size: Maximum number of reflections to store
        retention_policy: 'fifo' (default, unchanged), 'lru', or 'importance'
    """

    def __init__(self, llm, max_size: int = 100,
                 retention_policy: str = 'fifo'):
        if retention_policy not in ('fifo', 'lru', 'importance'):
            raise ValueError(
                f"retention_policy must be 'fifo', 'lru', or 'importance', "
                f"got {retention_policy!r}"
            )
        self.llm = llm
        self.max_size = max_size
        self.retention_policy = retention_policy

        if retention_policy == 'fifo':
            # ── ORIGINAL path — unchanged ──────────────────────────────
            self.reflections = deque(maxlen=max_size)
            self.embeddings = deque(maxlen=max_size)
        else:
            # Manual capacity management needed for arbitrary-index
            # eviction (deque(maxlen=N) can only drop from the left).
            self.reflections = []
            self.embeddings = []
            # Index-aligned metadata with self.reflections
            self.last_used: List[int] = []     # for 'lru': logical timestamp
            self.importance: List[float] = []  # for 'importance': score
            self._clock = 0                    # logical time counter

    def add_reflection(self, reflection: str,
                        importance_score: Optional[float] = None):
        """Add reflection with its semantic embedding.

        Args:
            reflection: text to store
            importance_score: only used when retention_policy=='importance'.
                Defaults to 0.0 if not provided. Ignored (accepted for
                call-site compatibility) under 'fifo' and 'lru'.
        """
        emb = self.llm.get_embedding(reflection)

        if self.retention_policy == 'fifo':
            # ── ORIGINAL path — unchanged ──────────────────────────────
            self.reflections.append(reflection)
            self.embeddings.append(emb)
            return

        # ── 'lru' / 'importance' — manual eviction before insert ───────
        if len(self.reflections) >= self.max_size:
            evict_idx = self._select_eviction_index()
            del self.reflections[evict_idx]
            del self.embeddings[evict_idx]
            del self.last_used[evict_idx]
            del self.importance[evict_idx]

        self.reflections.append(reflection)
        self.embeddings.append(emb)
        self.last_used.append(self._clock)
        self.importance.append(
            importance_score if importance_score is not None else 0.0
        )

    def _select_eviction_index(self) -> int:
        """Pick which stored entry to evict, per retention_policy."""
        if self.retention_policy == 'lru':
            # Evict the entry with the OLDEST "last used" timestamp
            # (least recently retrieved). Ties -> earliest such index.
            return int(np.argmin(self.last_used))

        if self.retention_policy == 'importance':
            # Evict the LOWEST-importance entry. Ties -> earliest
            # (oldest) such index.
            min_score = min(self.importance)
            for i, score in enumerate(self.importance):
                if score == min_score:
                    return i

        raise RuntimeError(f"Unknown retention_policy: {self.retention_policy}")

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

        if self.retention_policy == 'lru':
            # Mark these entries as "just used" so they're protected
            # from eviction relative to entries that aren't retrieved.
            self._clock += 1
            for i in top_k:
                self.last_used[i] = self._clock

        return [list(self.reflections)[i] for i in sorted(top_k)]

    def clear(self):
        """Clear all reflections and embeddings."""
        self.reflections.clear()
        self.embeddings.clear()
        if self.retention_policy != 'fifo':
            self.last_used.clear()
            self.importance.clear()
            self._clock = 0

    def __len__(self) -> int:
        return len(self.reflections)