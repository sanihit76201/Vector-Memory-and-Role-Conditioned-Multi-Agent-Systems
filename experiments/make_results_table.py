import json
import os
from statistics import mean
from collections import defaultdict
from datetime import datetime

# Files that contain your runs
RESULT_FILES = [
    "../results/extension1_smart_agent.json",
    "../results/extension1_vector_agent.json", 
    "../results/extension2_multiagent_agent.json",
    "../results/extension5_optimized_agent.json",
    "../results/extension3_rl_agent.json",
]

AGENT_TYPES = {
    "Original Reflexion":      "Original_Working",
    "Modular Baseline":        "Modular_Baseline", 
    "Smart (Task Isolation)":  "Smart_TaskIsolation",
    "VectorReflexion":         "VectorReflexion",
    "MultiAgentReflexion":     "MultiAgentReflexion",
    "OptimizedReflexion":      "OptimizedReflexion",
    "RLReflexion":             "RLReflexion",
}

def collect_latest_runs_only():
    """Get ONLY most recent batch per agent (not cumulative)."""
    all_latest = {}
    
    for path in RESULT_FILES:
        if not os.path.exists(path):
            print(f"[WARN] Skipping: {path}")
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Group by agent_type, keep only MOST RECENT run per agent
        agent_runs = defaultdict(list)
        results = data.get("results", {})
        
        for agent_name, tasks in results.items():
            for task in tasks:
                agent_type = task.get("agent_type")
                if agent_type in AGENT_TYPES.values():
                    # Add run identifier (filename + task_id for uniqueness)
                    task['run_id'] = path
                    agent_runs[agent_type].append(task)
        
        # For each agent, take only LATEST run data
        for agent_type, tasks in agent_runs.items():
            if agent_type not in all_latest:
                all_latest[agent_type] = []
            all_latest[agent_type].extend(tasks[-20:])  # Last 20 tasks only!
    
    return all_latest

def summarize_agent(results):
    """Unchanged."""
    if not results:
        return 0.0, 0.0, 0.0, 0.0, 0, 0

    total = len(results)
    passed = [r for r in results if r.get("success")]
    pass3 = len(passed) / total * 100

    pass1 = [r for r in passed if r.get("trials") == 1]
    pass1_rate = len(pass1) / total * 100

    trials_success = [r.get("trials", 0) for r in passed]
    avg_trials = mean(trials_success) if trials_success else 0.0

    failed_t1 = [r for r in results if not (r.get("success") and r.get("trials") == 1)]
    recovered = [r for r in results if r.get("success") and r.get("trials", 0) > 1]
    recovery_rate = (len(recovered) / len(failed_t1) * 100) if failed_t1 else 0.0

    return pass3, pass1_rate, avg_trials, recovery_rate, len(passed), total

def main():
    all_results = collect_latest_runs_only()  # ✅ Latest batch only

    summary = {}
    for display_name, agent_type in AGENT_TYPES.items():
        results = all_results.get(agent_type, [])
        pass3, pass1, avg_trials, recovery, passed, total = summarize_agent(results)
        summary[display_name] = {
            "pass3": pass3, "pass1": pass1, "avg_trials": avg_trials,
            "recovery": recovery, "passed": passed, "total": total,
        }

    # MAIN TABLE - LAST BATCH ONLY
    print("Table: HumanEval – LAST RUN BATCHES ONLY\n")
    print("| Agent                  | Pass@3 | Pass@1 | Avg trials | #Solved / #Tasks |")
    print("|------------------------|--------|--------|------------|------------------|")
    
    for name in AGENT_TYPES.keys():
        s = summary[name]
        print(f"| {name:22} | {s['pass3']:5.1f}% | {s['pass1']:5.1f}% | {s['avg_trials']:10.2f} | {s['passed']:3d} / {s['total']:3d}       |")

    # Metrics section unchanged
    print("\nQuantified metrics vs Modular Baseline:\n")
    base = summary["Modular Baseline"]
    base_p3 = base["pass3"] or 1e-9
    print(f"- Modular Baseline: {base['passed']}/{base['total']} tasks (latest batch)")
    print()

    for name in AGENT_TYPES.keys():
        if name == "Modular Baseline": continue
        s = summary[name]
        if s["total"] == 0: 
            print(f"{name}: no recent data.")
            continue

        abs_impr = s["pass3"] - base["pass3"]
        rel_impr = abs_impr / base_p3 * 100.0

        print(f"{name}:")
        print(f"  - Tasks: {s['total']}  (solved {s['passed']} / {s['total']})")
        print(f"  - Pass@3: {s['pass3']:.1f}%  (Δ vs baseline = {abs_impr:+.1f} pp, {rel_impr:+.1f} %)")
        print(f"  - Pass@1: {s['pass1']:.1f}%")
        print(f"  - Avg trials: {s['avg_trials']:.2f}")
        print(f"  - Recovery rate after failed trial 1: {s['recovery']:.1f}%\n")

if __name__ == "__main__":
    main()
