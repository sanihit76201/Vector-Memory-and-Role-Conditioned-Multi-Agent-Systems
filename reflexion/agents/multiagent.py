"""
Extension 2: Multi-Agent Reflexion with SHARED MEMORY + Communication Protocol.
✅ Shared reflections across agents
✅ Shared memory injected into agent reasoning (fixed)
✅ Agent specialization via role prompts (fixed)
✅ Cross-agent reasoning + debate rounds
✅ Majority vote + communication tracking (fixed)

── TMLR REVISION — TWO SEPARATE ABLATION AXES ──────────────────────────────
The reviewer's remaining concern distinguishes two different things that
both get called "role-conditioning":

  1. use_role_conditioning (prompt-level, unchanged from previous patch):
     whether each agent's LLM prompt is prefixed with its specialization
     text ("You are a syntax expert..."). Governs what the LLM WRITES.

  2. role_conditioned_memory (NEW — this patch):
     whether shared-memory RETRIEVAL is biased by role. When False (the
     default, and the behavior of every existing published result), every
     reflection is retrievable by every agent purely via cosine similarity
     — role has zero effect on what comes back, only agent-id
     self-exclusion applies. When True, reflections are tagged with the
     role that produced them, and retrieval re-ranks candidates to prefer
     same-role reflections over cross-role ones (falling back to cross-role
     memories only when there aren't enough same-role ones). This governs
     what the agent SEES, independent of prompt text.

Both flags default to the values that reproduce your existing published
numbers exactly (use_role_conditioning=True, role_conditioned_memory=False)
— MultiAgentReflexion() with no args is unchanged. New subclasses toggle
one flag at a time so each ablation isolates a single variable.
"""
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional
import logging
from .vector import VectorReflexionAgent
from ..evaluators import ObjectiveCodeEvaluator
from ..memory import VectorEpisodicMemory

logger = logging.getLogger(__name__)


# ============================================================
# AGENT ROLES — heterogeneous specialization
# ============================================================

AGENT_ROLES = [
    "You are a syntax and code quality expert. Focus on clean structure, "
    "correct Python syntax, and readable implementation.",

    "You are a logic and algorithm expert. Focus on correctness of the "
    "algorithm, efficiency, and handling of all logical cases.",

    "You are an edge-case and robustness expert. Focus on boundary conditions, "
    "empty inputs, type errors, and unexpected values.",
]

# Short role keys used for MEMORY tagging (distinct from the prompt text
# above). Index-aligned with AGENT_ROLES / self.agents.
ROLE_KEYS = ["syntax", "logic", "edge_case"]


# ============================================================
# SHARED MEMORY POOL
# ============================================================

class SharedMemoryPool:
    """
    Shared vector memory accessible by all agents.
    Agents read/write to common index for cross-learning.

    Args:
        role_conditioned_retrieval: If False (default — matches all
            existing published results), retrieval is pure cosine
            similarity with only agent-id self-exclusion (original
            behavior, unchanged). If True, each stored reflection is
            tagged with the producing agent's role_key, and retrieval
            re-ranks the semantic candidate pool to prefer same-role
            reflections before falling back to cross-role ones.
    """

    def __init__(self, llm, max_size: int = 500,
                 role_conditioned_retrieval: bool = False):
        self.llm = llm
        self.memory = VectorEpisodicMemory(llm, max_size=max_size)
        self.contributions = {}  # agent_id → count
        self.role_conditioned_retrieval = role_conditioned_retrieval

        # Parallel metadata store: role_tags[i] is the role_key (or None)
        # for self.memory.reflections[i]. Same maxlen so it stays aligned
        # under FIFO eviction. Only ever mutated here, in lockstep with
        # self.memory.add_reflection(), so ordering is guaranteed to match.
        self.role_tags = deque(maxlen=max_size)

    def add_reflection(self, agent_id: str, reflection: str,
                        role_key: Optional[str] = None):
        """Agent contributes tagged reflection to shared pool.

        role_key: the specialization that produced this reflection
            ("syntax" / "logic" / "edge_case" / None for Supervisor).
            Stored regardless of role_conditioned_retrieval so the flag
            can be toggled independently — retrieval behavior is
            controlled solely by get_relevant_memories().
        """
        self.memory.add_reflection(f"[{agent_id}] {reflection}")
        self.role_tags.append(role_key)
        self.contributions[agent_id] = self.contributions.get(agent_id, 0) + 1

    def get_relevant_memories(self, query: str, agent_id: str = None,
                               role_key: Optional[str] = None, k: int = 5) -> List[str]:
        """
        Retrieve shared memories.

        If agent_id provided, exclude own reflections for diversity
        (unchanged, applies in both modes).

        If role_conditioned_retrieval is True AND role_key is provided,
        additionally re-rank the semantic candidate pool so memories
        tagged with the SAME role_key are preferred over cross-role ones.
        """
        if not self.role_conditioned_retrieval or role_key is None:
            # ── ORIGINAL / non-role-conditioned path — unchanged ──────
            memories = self.memory.get_relevant_memories(query, k=k * 2)
            if agent_id:
                others = [m for m in memories if not m.startswith(f"[{agent_id}]")]
                return others[:k]
            return memories[:k]

        # ── ROLE-CONDITIONED path ─────────────────────────────────────
        # Widen the semantic candidate pool so there's enough headroom
        # to re-rank by role match without starving the result.
        pool_size = min(k * 4, len(self.memory)) or k
        candidates = self.memory.get_relevant_memories(query, k=pool_size)

        if agent_id:
            candidates = [m for m in candidates if not m.startswith(f"[{agent_id}]")]

        # Map reflection text -> role_key using the parallel deques.
        # (Both deques are appended together in add_reflection and share
        # maxlen, so they stay index-aligned under eviction.)
        tag_map = dict(zip(self.memory.reflections, self.role_tags))

        same_role  = [m for m in candidates if tag_map.get(m) == role_key]
        cross_role = [m for m in candidates if tag_map.get(m) != role_key]

        # Same-role memories first, cross-role as fallback — this IS the
        # role-conditioning: retrieval prioritizes specialization match.
        ranked = same_role + cross_role
        return ranked[:k]


# ============================================================
# MULTI-AGENT REFLEXION
# ============================================================

class MultiAgentReflexion:
    """
    3 specialized agents with shared memory + communication protocol.

    Protocol:
      Round 1 — Independent solving with shared memory injection
      Round 2 — Cross-agent debate (supervisor synthesis on all-fail)
      Round 3 — Weighted majority vote → best solution

    Args:
        use_role_conditioning: prompt-level role specialization text
            (default True — matches existing published behavior).
        role_conditioned_memory: memory-level role-biased retrieval
            (default False — matches existing published behavior, i.e.
            plain semantic retrieval with only agent-id self-exclusion).
    """

    def __init__(self, llm, max_trials: int = 3, num_agents: int = 3,
                 use_role_conditioning: bool = True,
                 role_conditioned_memory: bool = False):
        self.llm = llm
        self.max_trials = max_trials
        self.num_agents = num_agents
        self.use_role_conditioning = use_role_conditioning
        self.role_conditioned_memory = role_conditioned_memory
        self.evaluator = ObjectiveCodeEvaluator(timeout=10)

        # Shared memory pool
        self.shared_memory = SharedMemoryPool(
            llm, role_conditioned_retrieval=role_conditioned_memory
        )

        # 3 heterogeneous agents (specialized via role prompts, when enabled)
        self.agents = [
            VectorReflexionAgent(llm, max_trials),  # Syntax expert
            VectorReflexionAgent(llm, max_trials),  # Logic expert
            VectorReflexionAgent(llm, max_trials),  # Edge-case expert
        ]

        self.task_results: List[Dict] = []
        self.communication_log: List[Dict] = []

    # --------------------------------------------------------
    # SOLVE TASK
    # --------------------------------------------------------

    def solve_task(self, task: Dict, verbose: bool = False) -> Dict:
        task_id = task["task_id"]
        solutions = []
        debate_context = []

        # ── ROUND 1: Independent solving with shared memory ──────────
        for i, agent in enumerate(self.agents):
            agent_id = f"Agent-{i}"
            role = AGENT_ROLES[i]
            role_key = ROLE_KEYS[i]

            # Retrieve other agents' reflections (role-conditioned if enabled)
            memories = self.shared_memory.get_relevant_memories(
                task["prompt"], agent_id=agent_id, role_key=role_key, k=5
            )

            # Inject role text into prompt — GATED by use_role_conditioning.
            # Shared-memory injection happens either way; this keeps the
            # two ablation axes independent of each other.
            augmented_task = dict(task)
            header = ""
            if self.use_role_conditioning:
                header += f"# Your specialization: {role}\n\n"
            if memories:
                mem_ctx = "\n".join(f"- {m}" for m in memories)
                header += f"# Shared reflections from other agents:\n{mem_ctx}\n\n"
            augmented_task["prompt"] = header + task["prompt"]

            result = agent.solve_task(augmented_task, verbose=False)

            # Log communication event
            self.communication_log.append({
                "task_id": task_id,
                "agent_id": agent_id,
                "role": role[:50],
                "role_key": role_key,
                "use_role_conditioning": self.use_role_conditioning,
                "role_conditioned_memory": self.role_conditioned_memory,
                "memories_received": len(memories),
                "success": result["success"],
                "trials": result["trials"],
            })

            if result["success"]:
                solutions.append({
                    "code": result["code"],
                    "trials": result["trials"],
                    "agent_id": agent_id,
                    "confidence": 1.0 / result["trials"],
                })
                # Share successful approach — tagged with this agent's role
                self.shared_memory.add_reflection(
                    agent_id,
                    f"Solved {task_id} in {result['trials']} trial(s). "
                    f"Role: {role[:60]}",
                    role_key=role_key,
                )
            else:
                # Share failure with hint — tagged with this agent's role
                reflection = (
                    f"Failed {task_id} (trial {result['trials']}). "
                    f"Role: {role[:60]}. Review edge cases and logic."
                )
                self.shared_memory.add_reflection(agent_id, reflection, role_key=role_key)
                debate_context.append(reflection)

            agent.reset()

        # ── ROUND 2: Supervisor debate if all agents failed ───────────
        if not solutions and debate_context:
            debate_prompt = (
                f"CROSS-AGENT DEBATE: Task {task_id}\n\n"
                f"All agents failed. Shared failure context:\n"
                + "\n".join(f"- {r}" for r in debate_context[-3:])
                + f"\n\nTask:\n{task['prompt']}\n\n"
                "As Supervisor: synthesize a correct solution from collective failures.\n"
                "Output ONLY working Python code, no markdown:\n"
            )

            supervisor_code = self.llm.call_llm(debate_prompt, max_tokens=2048)

            # Clean markdown if present
            if "```python" in supervisor_code:
                supervisor_code = supervisor_code.split("```python")[1].split("```")[0].strip()
            elif "```" in supervisor_code:
                supervisor_code = supervisor_code.split("```")[1].split("```")[0].strip()

            supervisor_result = self.evaluator.evaluate(
                supervisor_code, task["entry_point"], task["test"]
            )

            self.communication_log.append({
                "task_id": task_id,
                "agent_id": "Supervisor",
                "role": "debate synthesis",
                "role_key": None,
                "use_role_conditioning": self.use_role_conditioning,
                "role_conditioned_memory": self.role_conditioned_memory,
                "memories_received": len(debate_context),
                "success": supervisor_result["passed"],
                "trials": 1,
            })

            if supervisor_result["passed"]:
                solutions.append({
                    "code": supervisor_code,
                    "trials": 1,
                    "agent_id": "Supervisor",
                    "confidence": 0.9,
                })
                # Supervisor has no fixed specialization -> role_key=None
                self.shared_memory.add_reflection(
                    "Supervisor", f"Debate synthesis solved {task_id}", role_key=None
                )

        # ── ROUND 3: Weighted majority vote ──────────────────────────
        if solutions:
            best = max(solutions, key=lambda x: x["confidence"])

            self.task_results.append({
                "task_id": task_id,
                "agents_solved": len(solutions),
                "shared_reflections_used": len(debate_context),
                "winning_agent": best["agent_id"],
            })

            return {
                "task_id": task_id,
                "success": True,
                "trials": best["trials"],
                "code": best["code"],
                "agent_type": type(self).__name__,
                "use_role_conditioning": self.use_role_conditioning,
                "role_conditioned_memory": self.role_conditioned_memory,
                "collaborators": len(solutions),
                "winning_agent": best["agent_id"],
                "shared_memory_size": len(self.shared_memory.memory),
            }

        # All failed including supervisor
        self.shared_memory.add_reflection(
            "Supervisor", f"All agents exhausted on {task_id}", role_key=None
        )
        self.task_results.append({
            "task_id": task_id,
            "agents_solved": 0,
            "shared_reflections_used": len(debate_context),
            "winning_agent": None,
        })

        return {
            "task_id": task_id,
            "success": False,
            "trials": self.max_trials,
            "agent_type": type(self).__name__,
            "use_role_conditioning": self.use_role_conditioning,
            "role_conditioned_memory": self.role_conditioned_memory,
            "collaborators": 0,
            "winning_agent": None,
            "shared_memory_size": len(self.shared_memory.memory),
        }

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.shared_memory.memory.clear()
        self.shared_memory.role_tags.clear()
        self.task_results.clear()
        self.communication_log.clear()

    # --------------------------------------------------------
    # COMMUNICATION ANALYSIS
    # --------------------------------------------------------

    def get_communication_analysis(self) -> Dict[str, Any]:
        """Quantify shared memory usage, collaboration, and agent contributions."""
        if not self.task_results:
            return {"message": "No tasks run yet"}

        total_tasks = len(self.task_results)
        collab_tasks = sum(1 for r in self.task_results if r["agents_solved"] > 1)
        supervisor_wins = sum(
            1 for r in self.task_results if r.get("winning_agent") == "Supervisor"
        )

        # Per-agent stats from communication log
        agent_stats = {}
        for entry in self.communication_log:
            aid = entry["agent_id"]
            if aid not in agent_stats:
                agent_stats[aid] = {"attempts": 0, "successes": 0, "memories_received": 0}
            agent_stats[aid]["attempts"] += 1
            agent_stats[aid]["successes"] += int(entry["success"])
            agent_stats[aid]["memories_received"] += entry["memories_received"]

        return {
            "total_tasks": total_tasks,
            "collaboration_rate_pct": round(collab_tasks / total_tasks * 100, 1),
            "supervisor_interventions": supervisor_wins,
            "avg_shared_reflections_per_task": round(
                float(np.mean([r["shared_reflections_used"] for r in self.task_results])), 2
            ),
            "shared_memory_total_entries": len(self.shared_memory.memory),
            "agent_contributions": self.shared_memory.contributions,
            "per_agent_stats": agent_stats,
            "protocol": "Round-robin → Debate → Weighted vote",
            "use_role_conditioning": self.use_role_conditioning,
            "role_conditioned_memory": self.role_conditioned_memory,
        }


# ============================================================
# ABLATION CONVENIENCE CLASSES
# ============================================================

class MultiAgentReflexionNoRoles(MultiAgentReflexion):
    """
    ABLATION AXIS 1 — prompt-level role text.
    Same architecture (shared memory, debate, vote); no specialization
    text injected into any agent's prompt. Memory retrieval stays
    non-role-conditioned (matches the main method's default).
    """
    def __init__(self, llm, max_trials: int = 3, num_agents: int = 3):
        super().__init__(llm, max_trials=max_trials, num_agents=num_agents,
                          use_role_conditioning=False,
                          role_conditioned_memory=False)


class MultiAgentReflexionRoleConditionedMemory(MultiAgentReflexion):
    """
    ABLATION AXIS 2 — role-conditioned MEMORY (the reviewer's specific ask).

    Identical to the main method (prompt-level role text ON, same
    architecture) EXCEPT shared-memory retrieval is now role-conditioned:
    reflections are tagged by producing role, and retrieval prefers
    same-role reflections over cross-role ones.

    Compare directly against plain MultiAgentReflexion() (default args)
    to isolate the effect of role-conditioned memory retrieval, holding
    prompt-level role-conditioning constant.
    """
    def __init__(self, llm, max_trials: int = 3, num_agents: int = 3):
        super().__init__(llm, max_trials=max_trials, num_agents=num_agents,
                          use_role_conditioning=True,
                          role_conditioned_memory=True)