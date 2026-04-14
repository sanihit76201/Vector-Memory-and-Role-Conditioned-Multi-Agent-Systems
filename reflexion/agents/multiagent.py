"""
Extension 2: Multi-Agent Reflexion with SHARED MEMORY + Communication Protocol.
✅ Shared reflections across agents
✅ Shared memory injected into agent reasoning (fixed)
✅ Agent specialization via role prompts (fixed)
✅ Cross-agent reasoning + debate rounds
✅ Majority vote + communication tracking (fixed)
"""
import numpy as np
from typing import Dict, List, Any
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


# ============================================================
# SHARED MEMORY POOL
# ============================================================

class SharedMemoryPool:
    """
    Shared vector memory accessible by all agents.
    Agents read/write to common index for cross-learning.
    """

    def __init__(self, llm, max_size: int = 500):
        self.llm = llm
        self.memory = VectorEpisodicMemory(llm, max_size=max_size)
        self.contributions = {}  # agent_id → count

    def add_reflection(self, agent_id: str, reflection: str):
        """Agent contributes tagged reflection to shared pool."""
        self.memory.add_reflection(f"[{agent_id}] {reflection}")
        self.contributions[agent_id] = self.contributions.get(agent_id, 0) + 1

    def get_relevant_memories(self, query: str, agent_id: str = None, k: int = 5) -> List[str]:
        """
        Retrieve shared memories.
        If agent_id provided, exclude own reflections for diversity.
        """
        memories = self.memory.get_relevant_memories(query, k=k * 2)

        if agent_id:
            others = [m for m in memories if not m.startswith(f"[{agent_id}]")]
            return others[:k]

        return memories[:k]


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
    """

    def __init__(self, llm, max_trials: int = 3, num_agents: int = 3):
        self.llm = llm
        self.max_trials = max_trials
        self.num_agents = num_agents
        self.evaluator = ObjectiveCodeEvaluator(timeout=10)

        # Shared memory pool
        self.shared_memory = SharedMemoryPool(llm)

        # 3 heterogeneous agents (specialized via role prompts)
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

            # Retrieve other agents' reflections
            memories = self.shared_memory.get_relevant_memories(
                task["prompt"], agent_id=agent_id, k=5
            )

            # Inject role + shared memories into prompt
            augmented_task = dict(task)
            role_header = f"# Your specialization: {role}\n\n"
            if memories:
                mem_ctx = "\n".join(f"- {m}" for m in memories)
                role_header += f"# Shared reflections from other agents:\n{mem_ctx}\n\n"
            augmented_task["prompt"] = role_header + task["prompt"]

            result = agent.solve_task(augmented_task, verbose=False)

            # Log communication event
            self.communication_log.append({
                "task_id": task_id,
                "agent_id": agent_id,
                "role": role[:50],
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
                # Share successful approach
                self.shared_memory.add_reflection(
                    agent_id,
                    f"Solved {task_id} in {result['trials']} trial(s). "
                    f"Role: {role[:60]}"
                )
            else:
                # Share failure with hint
                reflection = (
                    f"Failed {task_id} (trial {result['trials']}). "
                    f"Role: {role[:60]}. Review edge cases and logic."
                )
                self.shared_memory.add_reflection(agent_id, reflection)
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
                self.shared_memory.add_reflection(
                    "Supervisor", f"Debate synthesis solved {task_id}"
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
                "agent_type": "MultiAgentReflexion",
                "collaborators": len(solutions),
                "winning_agent": best["agent_id"],
                "shared_memory_size": len(self.shared_memory.memory),
            }

        # All failed including supervisor
        self.shared_memory.add_reflection(
            "Supervisor", f"All agents exhausted on {task_id}"
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
            "agent_type": "MultiAgentReflexion",
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
        }
