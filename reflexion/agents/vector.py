"""Extension 1: VectorReflexionAgent with FAISS semantic retrieval.

FULL CHAIN ANALYSIS (traced from base class source)
====================================================

ReflexionAgent.__init__(memory_mode='vector'):
    → self.memory = VectorEpisodicMemory(llm)   ✅ correct

ReflexionAgent.solve_task():
    → retrieves from self.memory with task['prompt']  ✅ uses vector search
    → builds prompt with mem_ctx                       ✅ memories reach LLM
    → calls self.llm.call_llm(prompt)                 ✅ generation uses memories

CONCLUSION:
    The base class already wires VectorEpisodicMemory correctly end-to-end.
    Previous VectorReflexionAgent was doing redundant duplicate work:
      - Double-instantiating VectorEpisodicMemory (base + us)
      - Double-retrieving memories (base retrieves k=3, we retrieved k=5 and discarded)
      - Injecting into augmented_task['prompt'] which base embeds in its own template

WHAT THIS CLASS SHOULD DO:
    1. Remove double instantiation
    2. Override ONLY what differs from base: top-k=5 vs base k=3
    3. Add memories_used / memory_pool to result dict for benchmark tracking
    4. Keep reflection storage (base stores k=3 generic; we store richer messages)
"""

import logging
from typing import Dict, List

from reflexion.memory import VectorEpisodicMemory
from reflexion.agents.base import ReflexionAgent

logger = logging.getLogger(__name__)


class VectorReflexionAgent(ReflexionAgent):
    """
    Reflexion agent with semantic episodic memory (VectorEpisodicMemory).

    The base ReflexionAgent already:
      - Creates VectorEpisodicMemory when memory_mode='vector'
      - Retrieves semantically similar memories per task prompt
      - Injects those memories into the LLM prompt
      - Stores failure reflections back into memory

    This subclass extends that with:
      - TOP_K=5 instead of base k=3 (more context for complex tasks)
      - Richer failure reflections (task_id + error + actionable hint)
      - memories_used + memory_pool in result dict (for benchmark metrics)
      - Explicit memory type verification on init (fail fast if misconfigured)
    """

    TOP_K_MEMORIES = 5   # base uses k=3; we retrieve more context

    def __init__(self, llm, max_trials: int = 3) -> None:
        # Base class creates VectorEpisodicMemory via memory_mode='vector'
        super().__init__(llm, memory_mode="vector", max_trials=max_trials)

        # Verify memory type — fail fast rather than silently use wrong memory
        if not isinstance(self.memory, VectorEpisodicMemory):
            raise TypeError(
                f"Expected VectorEpisodicMemory, got {type(self.memory).__name__}. "
                f"Check ReflexionAgent.__init__() memory_mode branching."
            )

        logger.info(
            "VectorReflexionAgent ready | memory=%s | top_k=%d",
            type(self.memory).__name__,
            self.TOP_K_MEMORIES,
        )

    # ------------------------------------------------------------------ #
    #  Override solve_task to use TOP_K=5 and richer reflections          #
    # ------------------------------------------------------------------ #

    def solve_task(self, task: Dict, verbose: bool = False) -> Dict:
        """
        Solve task using VectorEpisodicMemory with TOP_K=5 retrieval.

        We re-implement the core loop (rather than calling super()) because:
          - We need k=5 not k=3 in get_relevant_memories()
          - We need richer reflection strings for better future retrieval
          - We need memories_used in the result dict for benchmark metrics

        The logic mirrors ReflexionAgent.solve_task() exactly — only the
        retrieval k, reflection format, and result keys differ.
        """
        task_id = task["task_id"]

        for trial in range(self.max_trials):

            # ── Retrieve top-5 (not base k=3) semantic memories ───────
            memories: List[str] = self.memory.get_relevant_memories(
                task["prompt"], k=self.TOP_K_MEMORIES
            )
            mem_ctx = "\n".join(f"- {m}" for m in memories) if memories else "None"

            # ── Prompt mirrors base class template exactly ─────────────
            prompt = f"""You are an expert Python programmer. Complete this function:

{task['prompt']}

Past reflections (TOP-{self.TOP_K_MEMORIES} semantically similar — learn from these):
{mem_ctx}

Requirements:
1. Complete the function implementation
2. Handle all edge cases
3. Make sure all test cases pass
4. Output ONLY the Python code, no markdown, no explanations

Your code:"""

            try:
                logger.info("🔄 Trial %d/%d — %s | memories=%d",
                            trial + 1, self.max_trials, task_id, len(memories))

                # ── Generate ───────────────────────────────────────────
                code = self.llm.call_llm(prompt, max_tokens=2048)

                if isinstance(code, list):
                    code = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in code
                )

                # ── Clean markdown ─────────────────────────────────────
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0].strip()
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0].strip()

                # ── Evaluate ───────────────────────────────────────────
                results = self.evaluator.evaluate(
                    code, task["entry_point"], task["test"]
                )

                if results["passed"]:
                    logger.info("✅ %s solved in %d trials | memories_used=%d",
                                task_id, trial + 1, len(memories))
                    return {
                        "task_id":       task_id,
                        "success":       True,
                        "trials":        trial + 1,
                        "code":          code,
                        "agent_type":    "VectorReflexion",
                        "memories_used": len(memories),
                        "memory_pool":   len(self.memory),
                    }

                # ── Richer failure reflection ──────────────────────────
                # Base stores: "Trial N failed: <error>"
                # We store:    task context + error + actionable hint
                # → future similar tasks get more useful retrieved context
                reflection = (
                    f"Task '{task_id}' trial {trial+1} failed: {results['error']}. "
                    f"Hint: review edge cases, type handling, boundary conditions."
                )
                self.memory.add_reflection(reflection)
                logger.warning("❌ %s trial %d failed | pool=%d",
                               task_id, trial + 1, len(self.memory))

            except KeyboardInterrupt:
                logger.error("⚠️  Interrupted")
                raise
            except Exception as exc:
                reflection = (
                    f"Task '{task_id}' trial {trial+1} exception: {str(exc)[:120]}. "
                    f"Check input types and function signature."
                )
                self.memory.add_reflection(reflection)
                logger.error("❌ Exception in %s: %s", task_id, exc)

        logger.warning("❌ %s failed after %d trials | pool=%d",
                       task_id, self.max_trials, len(self.memory))
        return {
            "task_id":       task_id,
            "success":       False,
            "trials":        self.max_trials,
            "agent_type":    "VectorReflexion",
            "memories_used": 0,
            "memory_pool":   len(self.memory),
        }

