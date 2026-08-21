"""HotpotQA Reflexion agents — Baseline, Vector, and MultiAgent."""

import logging
from typing import Dict, List, Optional

from reflexion.benchmarks.hotpotqa import HotpotQALoader
from reflexion.evaluators.hotpotqa_eval import HotpotQAEvaluator
from reflexion.memory.temporal import TemporalMemory
from reflexion.memory.vector import VectorEpisodicMemory

logger = logging.getLogger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────────────

BASELINE_PROMPT = """You are an expert at multi-hop question answering.
Answer the question by reasoning over the provided context paragraphs.

Question: {question}

Context:
{context}

Past reflections (learn from previous mistakes):
{memories}

Instructions:
1. Read all context paragraphs carefully
2. Identify the relevant facts across multiple paragraphs
3. Reason step by step
4. Output your final answer on the last line as: "Answer: <your answer>"

Your response:"""


VECTOR_PROMPT = """You are an expert at multi-hop question answering.
Answer the question by reasoning over the provided context paragraphs.

Question: {question}

Context:
{context}

Semantically relevant past reflections (TOP-{k} similar — learn from these):
{memories}

Instructions:
1. Read all context paragraphs carefully
2. Identify the relevant facts across multiple paragraphs
3. Note what went wrong in past reflections and avoid those mistakes
4. Reason step by step
5. Output your final answer on the last line as: "Answer: <your answer>"

Your response:"""


GENERATOR_PROMPT = """You are an expert researcher finding answers in text.
Your role: Generate an initial answer by identifying key facts.

Question: {question}

Context:
{context}

Past generator reflections:
{memories}

Instructions:
1. Find the key facts needed across paragraphs
2. Produce an initial answer
3. Output: "Answer: <your answer>"

Your response:"""


CRITIC_PROMPT = """You are a critical reviewer checking answers for correctness.
Your role: Review the answer WITHOUT providing a new answer yourself.

Question: {question}

Context:
{context}

Proposed answer: {answer}

Past critic reflections:
{memories}

Instructions:
1. Check if the answer is supported by the context
2. Identify any factual errors or missing reasoning steps
3. Point out specifically what is wrong or what is missing
4. Do NOT provide the correct answer — only critique

Your critique:"""


VERIFIER_PROMPT = """You are a verification specialist synthesising a final answer.
Your role: Use the generator's answer and critic's feedback to produce the correct answer.

Question: {question}

Context:
{context}

Generator's answer: {generator_answer}
Critic's feedback: {critique}

Past verifier reflections:
{memories}

Instructions:
1. Consider the critic's feedback carefully
2. Fix any identified errors
3. Produce the final, corrected answer
4. Output: "Answer: <your answer>"

Your response:"""


# ── Base HotpotQA Agent ───────────────────────────────────────────────────────

class HotpotQAReflexionAgent:
    """
    Baseline Reflexion agent for HotpotQA using TemporalMemory.
    """

    def __init__(self, llm, max_trials: int = 3):
        self.llm        = llm
        self.max_trials = max_trials
        self.evaluator  = HotpotQAEvaluator()
        self.memory     = TemporalMemory(max_size=10)

    def solve_task(self, task: Dict, verbose: bool = False) -> Dict:
        task_id  = task["task_id"]
        question = task["question"]
        gold     = task["answer"]
        context  = HotpotQALoader.format_context(task["context"])

        for trial in range(self.max_trials):
            memories = self.memory.get_relevant_memories(k=3)
            mem_ctx  = "\n".join(f"- {m}" for m in memories) if memories else "None"

            prompt = BASELINE_PROMPT.format(
                question=question,
                context=context,
                memories=mem_ctx,
            )

            try:
                logger.info("Trial %d/%d — %s", trial+1, self.max_trials, task_id)
                response = self.llm.call_llm(prompt, max_tokens=512)

                if isinstance(response, list):
                    response = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in response
                    )

                result = self.evaluator.evaluate(response, gold)

                if result["passed"]:
                    logger.info("✅ %s solved in %d trials", task_id, trial+1)
                    return {
                        "task_id"   : task_id,
                        "success"   : True,
                        "trials"    : trial + 1,
                        "em"        : result["em"],
                        "f1"        : result["f1"],
                        "prediction": result["prediction"],
                        "gold"      : gold,
                        "agent_type": "HotpotQA_Baseline",
                    }

                # Store reflection
                reflection = (
                    f"Trial {trial+1} failed on '{question[:80]}'. "
                    f"{result['error']}"
                )
                self.memory.add_reflection(reflection)
                logger.warning("❌ Trial %d failed | %s", trial+1, result["error"])

            except KeyboardInterrupt:
                raise
            except Exception as e:
                reflection = f"Trial {trial+1} error: {str(e)[:100]}"
                self.memory.add_reflection(reflection)
                logger.error("❌ Exception: %s", e)

        logger.warning("❌ %s failed after %d trials", task_id, self.max_trials)
        return {
            "task_id"   : task_id,
            "success"   : False,
            "trials"    : self.max_trials,
            "em"        : 0.0,
            "f1"        : 0.0,
            "prediction": "",
            "gold"      : gold,
            "agent_type": "HotpotQA_Baseline",
        }

    def reset(self):
        self.memory.clear()


# ── Vector HotpotQA Agent ─────────────────────────────────────────────────────

class HotpotQAVectorAgent:
    """
    VectorEpisodicMemory Reflexion agent for HotpotQA (Extension 1).
    Uses semantic retrieval with TOP_K=5.
    """

    TOP_K = 5

    def __init__(self, llm, max_trials: int = 3):
        self.llm        = llm
        self.max_trials = max_trials
        self.evaluator  = HotpotQAEvaluator()
        self.memory     = VectorEpisodicMemory(llm, max_size=500)

    def solve_task(self, task: Dict, verbose: bool = False) -> Dict:
        task_id  = task["task_id"]
        question = task["question"]
        gold     = task["answer"]
        context  = HotpotQALoader.format_context(task["context"])

        for trial in range(self.max_trials):
            memories = self.memory.get_relevant_memories(question, k=self.TOP_K)
            mem_ctx  = "\n".join(f"- {m}" for m in memories) if memories else "None"

            prompt = VECTOR_PROMPT.format(
                question=question,
                context=context,
                memories=mem_ctx,
                k=self.TOP_K,
            )

            try:
                logger.info("Trial %d/%d — %s | memories=%d",
                            trial+1, self.max_trials, task_id, len(memories))
                response = self.llm.call_llm(prompt, max_tokens=512)

                if isinstance(response, list):
                    response = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in response
                    )

                result = self.evaluator.evaluate(response, gold)

                if result["passed"]:
                    logger.info("✅ %s solved in %d trials | pool=%d",
                                task_id, trial+1, len(self.memory))
                    return {
                        "task_id"      : task_id,
                        "success"      : True,
                        "trials"       : trial + 1,
                        "em"           : result["em"],
                        "f1"           : result["f1"],
                        "prediction"   : result["prediction"],
                        "gold"         : gold,
                        "memories_used": len(memories),
                        "memory_pool"  : len(self.memory),
                        "agent_type"   : "HotpotQA_Vector",
                    }

                reflection = (
                    f"Task '{task_id}' trial {trial+1} failed: {result['error']}. "
                    f"Question: {question[:80]}. "
                    f"Hint: check all context paragraphs for supporting facts."
                )
                self.memory.add_reflection(reflection)
                logger.warning("❌ Trial %d failed | pool=%d", trial+1, len(self.memory))

            except KeyboardInterrupt:
                raise
            except Exception as e:
                reflection = (
                    f"Task '{task_id}' trial {trial+1} exception: {str(e)[:100]}"
                )
                self.memory.add_reflection(reflection)
                logger.error("❌ Exception: %s", e)

        return {
            "task_id"      : task_id,
            "success"      : False,
            "trials"       : self.max_trials,
            "em"           : 0.0,
            "f1"           : 0.0,
            "prediction"   : "",
            "gold"         : gold,
            "memories_used": 0,
            "memory_pool"  : len(self.memory),
            "agent_type"   : "HotpotQA_Vector",
        }

    def reset(self):
        self.memory.clear()


# ── MultiAgent HotpotQA ───────────────────────────────────────────────────────

class HotpotQAMultiAgent:
    """
    Generator-Critic-Verifier pipeline for HotpotQA (Extension 2).
    Shared VectorEpisodicMemory pool with role-prefixed reflections.
    """

    TOP_K = 3

    def __init__(self, llm, max_trials: int = 3):
        self.llm        = llm
        self.max_trials = max_trials
        self.evaluator  = HotpotQAEvaluator()
        self.memory     = VectorEpisodicMemory(llm, max_size=500)

    def _get_memories(self, query: str, role: str) -> str:
        """Retrieve role-conditioned memories."""
        role_query = f"[{role}] {query}"
        memories   = self.memory.get_relevant_memories(role_query, k=self.TOP_K)
        return "\n".join(f"- {m}" for m in memories) if memories else "None"

    def _call(self, prompt: str) -> str:
        response = self.llm.call_llm(prompt, max_tokens=512)
        if isinstance(response, list):
            response = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in response
            )
        return response.strip()

    def solve_task(self, task: Dict, verbose: bool = False) -> Dict:
        task_id  = task["task_id"]
        question = task["question"]
        gold     = task["answer"]
        context  = HotpotQALoader.format_context(task["context"])

        for trial in range(self.max_trials):
            try:
                # ── Generator ─────────────────────────────────────────
                gen_memories = self._get_memories(question, "Generator")
                gen_prompt   = GENERATOR_PROMPT.format(
                    question=question,
                    context=context,
                    memories=gen_memories,
                )
                gen_response = self._call(gen_prompt)
                gen_result   = self.evaluator.evaluate(gen_response, gold)
                gen_answer   = gen_result["prediction"]

                logger.info("Trial %d/%d Generator: '%s' (F1=%.2f)",
                            trial+1, self.max_trials,
                            gen_answer[:50], gen_result["f1"])

                # ── Critic ────────────────────────────────────────────
                crit_memories = self._get_memories(question, "Critic")
                crit_prompt   = CRITIC_PROMPT.format(
                    question=question,
                    context=context,
                    answer=gen_answer,
                    memories=crit_memories,
                )
                critique = self._call(crit_prompt)
                logger.info("Critic: '%s'", critique[:80])

                # ── Verifier ──────────────────────────────────────────
                ver_memories = self._get_memories(question, "Verifier")
                ver_prompt   = VERIFIER_PROMPT.format(
                    question=question,
                    context=context,
                    generator_answer=gen_answer,
                    critique=critique,
                    memories=ver_memories,
                )
                ver_response = self._call(ver_prompt)
                ver_result   = self.evaluator.evaluate(ver_response, gold)
                final_answer = ver_result["prediction"]

                logger.info("Verifier: '%s' (F1=%.2f)",
                            final_answer[:50], ver_result["f1"])

                # ── Evaluate final answer ─────────────────────────────
                if ver_result["passed"]:
                    logger.info("✅ %s solved in %d trials", task_id, trial+1)
                    return {
                        "task_id"           : task_id,
                        "success"           : True,
                        "trials"            : trial + 1,
                        "em"                : ver_result["em"],
                        "f1"                : ver_result["f1"],
                        "prediction"        : final_answer,
                        "gold"              : gold,
                        "shared_memory_size": len(self.memory),
                        "agent_type"        : "HotpotQA_MultiAgent",
                    }

                # ── All three agents write reflections on failure ──────
                self.memory.add_reflection(
                    f"[Generator] Task '{task_id}' trial {trial+1}: "
                    f"Generated '{gen_answer}' for '{question[:60]}'. "
                    f"Was wrong. F1={gen_result['f1']:.2f}."
                )
                self.memory.add_reflection(
                    f"[Critic] Task '{task_id}' trial {trial+1}: "
                    f"Critique was: {critique[:100]}."
                )
                self.memory.add_reflection(
                    f"[Verifier] Task '{task_id}' trial {trial+1}: "
                    f"Final answer '{final_answer}' was wrong. "
                    f"Gold: '{gold}'. F1={ver_result['f1']:.2f}."
                )

                logger.warning("❌ Trial %d failed | pool=%d",
                               trial+1, len(self.memory))

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error("❌ Exception trial %d: %s", trial+1, e)
                self.memory.add_reflection(
                    f"[Generator] Task '{task_id}' exception: {str(e)[:100]}"
                )

        return {
            "task_id"           : task_id,
            "success"           : False,
            "trials"            : self.max_trials,
            "em"                : 0.0,
            "f1"                : 0.0,
            "prediction"        : "",
            "gold"              : gold,
            "shared_memory_size": len(self.memory),
            "agent_type"        : "HotpotQA_MultiAgent",
        }

    def reset(self):
        self.memory.clear()