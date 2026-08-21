"""
HotpotQA Experiment Runner
===========================
Runs Baseline, VectorReflexion, and MultiAgentReflexion on HotpotQA.

Usage:
    cd experiments
    python run_hotpotqa.py --tasks 100
    python run_hotpotqa.py --tasks 50 --extension vector
    python run_hotpotqa.py --tasks 100 --extension all
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import logging
import argparse
import numpy as np
from scipy import stats
from pathlib import Path

from reflexion.config import SecureConfigLoader
from reflexion.llm import BaseLLMModel
from reflexion.benchmarks.hotpotqa import HotpotQALoader
from reflexion.agents.hotpotqa_agent import (
    HotpotQAReflexionAgent,
    HotpotQAVectorAgent,
    HotpotQAMultiAgent,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def numpy_encoder(obj):
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    raise TypeError(f"Object {type(obj)} not serializable")


def run_agent(agent, tasks, name):
    """Run agent on all tasks and return results."""
    logger.info('\n%s\n Running: %s\n%s', "="*70, name, "="*70)
    results = []

    for i, task in enumerate(tasks):
        logger.info("Task %d/%d: %s", i+1, len(tasks), task["task_id"])
        result = agent.solve_task(task, verbose=False)
        result["agent_type"] = name
        results.append(result)

        if hasattr(agent, "reset"):
            agent.reset()

    return results


def compute_metrics(results):
    """Compute EM, F1, Pass rate, avg trials."""
    n       = len(results)
    passed  = sum(1 for r in results if r.get("success"))
    em      = np.mean([r.get("em", 0) for r in results])
    f1      = np.mean([r.get("f1", 0) for r in results])
    p1      = sum(1 for r in results if r.get("success") and r.get("trials") == 1)
    s_trials = [r["trials"] for r in results if r.get("success") and r.get("trials", 0) > 0]
    avg_t   = float(np.mean(s_trials)) if s_trials else 0.0
    fail_t1 = [r for r in results if not (r.get("success") and r.get("trials") == 1)]
    recov   = sum(1 for r in results if r.get("success") and r.get("trials", 0) > 1)
    rec     = recov / len(fail_t1) * 100 if fail_t1 else 100.0

    return {
        "n"         : n,
        "pass_rate" : round(passed / n * 100, 1),
        "pass_at_1" : round(p1 / n * 100, 1),
        "exact_match": round(float(em) * 100, 1),
        "f1"        : round(float(f1) * 100, 1),
        "avg_trials": round(avg_t, 2),
        "recovery"  : round(rec, 1),
    }


def print_metrics(baseline_r, ext_r, ext_name):
    """Print quantified metrics and statistical validation."""
    bm = compute_metrics(baseline_r)
    em = compute_metrics(ext_r)

    print("\n" + "="*70)
    print("HOTPOTQA PERFORMANCE METRICS")
    print("="*70)

    print(f"\n{'Metric':<25} {'Baseline':>12} {ext_name:>20}")
    print("-"*70)
    for key in ["pass_rate", "pass_at_1", "exact_match", "f1",
                "avg_trials", "recovery"]:
        print(f"  {key:<23} {bm[key]:>11} {em[key]:>19}")

    # Statistical validation
    b_bin = [1 if r.get("success") else 0 for r in baseline_r]
    e_bin = [1 if r.get("success") else 0 for r in ext_r]

    if len(b_bin) == len(e_bin) and len(b_bin) > 1:
        t_stat, p_val = stats.ttest_rel(e_bin, b_bin)
        diff    = np.array(e_bin) - np.array(b_bin)
        cohens  = float(diff.mean() / (diff.std() + 1e-9))
        delta   = round((np.mean(e_bin) - np.mean(b_bin)) * 100, 1)

        print(f"\n{'STATISTICAL VALIDATION':}")
        print("-"*70)
        print(f"  Delta Pass Rate : {delta:+.1f} pp")
        print(f"  t-statistic     : {t_stat:.3f}")
        print(f"  p-value         : {p_val:.4f} "
              f"{'*' if p_val < 0.05 else '(not significant)'}")
        print(f"  Cohen's d       : {cohens:.3f}")

    # F1 comparison
    b_f1 = np.mean([r.get("f1", 0) for r in baseline_r])
    e_f1 = np.mean([r.get("f1", 0) for r in ext_r])
    print(f"\n  F1 improvement  : {(e_f1 - b_f1)*100:+.1f} pp")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="HotpotQA Reflexion Benchmark")
    parser.add_argument("--tasks",     type=int, default=100,
                        help="Number of tasks (default 100)")
    parser.add_argument("--extension", default="all",
                        choices=["baseline", "vector", "multiagent", "all"],
                        help="Which extension to run")
    parser.add_argument("--outdir",    default="../results/seed_runs")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    try:
        config = SecureConfigLoader().load_from_env_file(env_path)
    except Exception as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    llm = BaseLLMModel(
        config["openrouter_api_key"],
        config["openrouter_model"],
        config["gemini_api_base"],
        config["rate_limit_delay"],
    )

    # ── Load tasks ────────────────────────────────────────────────
    logger.info("Loading HotpotQA tasks...")
    tasks = HotpotQALoader.load(num_samples=args.tasks)
    logger.info("Loaded %d tasks", len(tasks))

    # ETA estimate
    calls_per_task = {"baseline": 3, "vector": 3, "multiagent": 9, "all": 15}
    eta = len(tasks) * calls_per_task.get(args.extension, 15) * config["rate_limit_delay"] / 60
    logger.info("ETA: ~%.0f minutes", eta)
    input("\nPress ENTER to start...")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── Baseline (always runs) ────────────────────────────────────
    logger.info("\nRunning HotpotQA Baseline...")
    baseline_agent   = HotpotQAReflexionAgent(llm, max_trials=3)
    baseline_results = run_agent(baseline_agent, tasks, "HotpotQA_Baseline")
    all_results["hotpotqa_baseline"] = baseline_results

    # ── Vector Extension ──────────────────────────────────────────
    if args.extension in ("vector", "all"):
        logger.info("\nRunning HotpotQA VectorReflexion...")
        vector_agent   = HotpotQAVectorAgent(llm, max_trials=3)
        vector_results = run_agent(vector_agent, tasks, "HotpotQA_Vector")
        all_results["hotpotqa_vector"] = vector_results
        print_metrics(baseline_results, vector_results, "VectorReflexion")

    # ── MultiAgent Extension ──────────────────────────────────────
    if args.extension in ("multiagent", "all"):
        logger.info("\nRunning HotpotQA MultiAgent...")
        multi_agent    = HotpotQAMultiAgent(llm, max_trials=3)
        multi_results  = run_agent(multi_agent, tasks, "HotpotQA_MultiAgent")
        all_results["hotpotqa_multiagent"] = multi_results
        print_metrics(baseline_results, multi_results, "MultiAgentReflexion")

    # ── Save results ──────────────────────────────────────────────
    output = {
        "dataset"  : "HotpotQA",
        "num_tasks": len(tasks),
        "task_ids" : [t["task_id"] for t in tasks],
        "results"  : all_results,
    }

    out_path = os.path.join(args.outdir, "hotpotqa_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=numpy_encoder)

    logger.info("\nResults saved to: %s", out_path)
    logger.info("HOTPOTQA EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()