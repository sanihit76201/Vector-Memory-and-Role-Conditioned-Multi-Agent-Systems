"""
Extension 1 — Reasoning Quality Benchmark

Measures reasoning quality of VectorReflexionAgent vs base ReflexionAgent
on a sample of HumanEval tasks.

Metrics:
✔ Pass rate (success/total)
✔ Average trials to solve
✔ Memories used per task
✔ Reasoning quality scores (constraint awareness, logical structure, etc.)
✔ JSON output for thesis tables
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import logging
import numpy as np

from reflexion.agents import ReflexionAgent, VectorReflexionAgent
from reflexion.config import SecureConfigLoader
from reflexion.llm import BaseLLMModel

logging.basicConfig(level=logging.WARNING)


# ============================================================
# SAMPLE HUMANEVAL TASKS
# ============================================================

TASKS = [
    {
        "task_id": "HumanEval/0",
        "prompt": "def has_close_elements(numbers: list, threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    \"\"\"\n",
        "entry_point": "has_close_elements",
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": "def separate_paren_groups(paren_string: str) -> list:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses.\n    Your goal is to separate those groups into separate strings and return the list of those.\n    \"\"\"\n",
        "entry_point": "separate_paren_groups",
        "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n    assert candidate('(()(())((())))') == ['(()(())((())))']\n",
    },
    {
        "task_id": "HumanEval/2",
        "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    an integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n    Return the decimal part of the number.\n    \"\"\"\n",
        "entry_point": "truncate_number",
        "test": "def check(candidate):\n    assert candidate(3.5) == 0.5\n    assert abs(candidate(1.33) - 0.33) < 1e-6\n    assert abs(candidate(123.456) - 0.456) < 1e-6\n",
    },
    {
        "task_id": "HumanEval/3",
        "prompt": "def below_zero(operations: list) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    \"\"\"\n",
        "entry_point": "below_zero",
        "test": "def check(candidate):\n    assert candidate([]) == False\n    assert candidate([1, 2, 3]) == False\n    assert candidate([1, 2, -4, 5]) == True\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True\n",
    },
    {
        "task_id": "HumanEval/4",
        "prompt": "def mean_absolute_deviation(numbers: list) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case).\n    \"\"\"\n",
        "entry_point": "mean_absolute_deviation",
        "test": "def check(candidate):\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.2) < 1e-6\n",
    },
]


# ============================================================
# RUN BENCHMARK
# ============================================================

def run_benchmark(agent, tasks, label):
    print(f"\n{'='*60}")
    print(f"🧪 {label}")
    print(f"{'='*60}")

    results = []
    for task in tasks:
        result = agent.solve_task(task)
        results.append(result)
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {result['task_id']} | "
              f"trials={result['trials']} | "
              f"memories_used={result.get('memories_used', 'N/A')}")

    return results


# ============================================================
# SUMMARISE RESULTS
# ============================================================

def summarise(results, label):
    total  = len(results)
    passed = sum(1 for r in results if r["success"])
    pass_rate = passed / total * 100

    trials     = [r["trials"] for r in results if r["success"]]
    avg_trials = float(np.mean(trials)) if trials else float(results[0]["trials"])

    memories     = [r.get("memories_used", 0) for r in results]
    avg_memories = float(np.mean(memories))

    reasoning_scores = []
    for r in results:
        if "reasoning_quality" in r:
            reasoning_scores.append(r["reasoning_quality"]["completeness_score"])

    avg_reasoning = float(np.mean(reasoning_scores)) if reasoning_scores else None

    print(f"\n📊 {label} Summary:")
    print(f"   Pass rate:    {passed}/{total} ({pass_rate:.1f}%)")
    print(f"   Avg trials:   {avg_trials:.2f}")
    print(f"   Avg memories: {avg_memories:.2f}")
    if avg_reasoning is not None:
        print(f"   Avg reasoning: {avg_reasoning:.3f}")

    summary = {
        "label": label,
        "total_tasks": total,
        "passed": passed,
        "pass_rate_pct": pass_rate,
        "avg_trials": avg_trials,
        "avg_memories_used": avg_memories,
    }
    if avg_reasoning is not None:
        summary["avg_reasoning_completeness"] = avg_reasoning

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Extension 1: Reasoning Quality Benchmark")
    print(f"   Tasks: {len(TASKS)}")

    # Load config from repo root .env
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    config = SecureConfigLoader().load_from_env_file(env_path)

    llm = BaseLLMModel(
        config["openrouter_api_key"],
        config["openrouter_model"]
    )

    # ── Base agent (temporal memory) ────────────────────────
    base_agent   = ReflexionAgent(llm, memory_mode="temporal")
    base_results = run_benchmark(base_agent, TASKS, "Base ReflexionAgent (Temporal)")
    base_summary = summarise(base_results, "Base ReflexionAgent")

    # ── Vector agent (semantic memory) ──────────────────────
    vector_agent   = VectorReflexionAgent(llm)
    vector_results = run_benchmark(vector_agent, TASKS, "VectorReflexionAgent (Semantic)")
    vector_summary = summarise(vector_results, "VectorReflexionAgent")

    # ── Comparison ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("📈 COMPARISON")
    print(f"{'='*60}")
    print(f"  Pass rate:  Base={base_summary['pass_rate_pct']:.1f}%  "
          f"Vector={vector_summary['pass_rate_pct']:.1f}%")
    print(f"  Avg trials: Base={base_summary['avg_trials']:.2f}  "
          f"Vector={vector_summary['avg_trials']:.2f}")

    # ── Save Results — into results/seed_runs/ ───────────────
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'seed_runs'
    )
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "extension1_reasoning_benchmark.json")

    with open(output_path, "w") as f:
        json.dump({
            "tasks_evaluated": len(TASKS),
            "base": {
                "summary": base_summary,
                "results": base_results,
            },
            "vector": {
                "summary": vector_summary,
                "results": vector_results,
            }
        }, f, indent=2, default=str)

    print(f"\n💾 Saved: {output_path}")
    print("✅ EXTENSION 1 COMPLETE")


if __name__ == "__main__":
    main()