import json
import os
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import matplotlib.patches as mpatches

# Same RESULT_FILES and AGENT_TYPES as your table script
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

def collect_all_results():
    collected = {agent_type: [] for agent_type in AGENT_TYPES.values()}
    
    for path in RESULT_FILES:
        if not os.path.exists(path):
            print(f"[WARN] File not found (skipping): {path}")
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        all_results = data.get("results", {})
        for _key, lst in all_results.items():
            for row in lst:
                atype = row.get("agent_type")
                if atype in collected:
                    collected[atype].append(row)
    return collected

def summarize_agent(results):
    if not results:
        return 0.0, 0.0, 0.0, 0, 0
        
    total = len(results)
    passed = [r for r in results if r.get("success")]
    pass3 = len(passed) / total * 100
    
    pass1 = [r for r in passed if r.get("trials") == 1]
    pass1_rate = len(pass1) / total * 100
    
    trials_success = [r.get("trials", 0) for r in passed]
    avg_trials = mean(trials_success) if trials_success else 0.0
    
    return pass3, pass1_rate, avg_trials, len(passed), total

def main():
    all_results = collect_all_results()
    
    # Collect data for plotting
    agent_names = list(AGENT_TYPES.keys())
    pass3_data = []
    pass1_data = []
    trials_data = []
    solved_data = []
    
    for name in agent_names:
        agent_type = AGENT_TYPES[name]
        results = all_results.get(agent_type, [])
        p3, p1, avg_t, passed, total = summarize_agent(results)
        pass3_data.append(p3)
        pass1_data.append(p1)
        trials_data.append(avg_t)
        solved_data.append(f"{passed}/{total}")
    
    # Create 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Pass@3 Bar Chart (main metric)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    bars1 = ax1.bar(range(len(agent_names)), pass3_data, color=colors[:len(agent_names)], 
                    edgecolor='black', linewidth=1.2, alpha=0.8)
    ax1.set_ylabel('Pass@3 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Pass@3 Accuracy\n(HumanEval)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(0, 110)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pass3_data[i]:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Pass@1 Bar Chart
    bars2 = ax2.bar(range(len(agent_names)), pass1_data, color=colors[:len(agent_names)], 
                    edgecolor='black', linewidth=1.2, alpha=0.8)
    ax2.set_ylabel('Pass@1 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Pass@1 (First Trial)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylim(0, 100)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pass1_data[i]:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Avg Trials (inverted - lower is better)
    ax3.bar(range(len(agent_names)), trials_data, color=colors[:len(agent_names)], 
            edgecolor='black', linewidth=1.2, alpha=0.8)
    ax3.set_ylabel('Avg Trials per Task', fontsize=12, fontweight='bold')
    ax3.set_title('Trial Efficiency\n(Lower = Better)', fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylim(0, max(trials_data)*1.2)
    
    for i, val in enumerate(trials_data):
        ax3.text(i, val + 0.05, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.tick_params(axis='x', rotation=45)
    
    # Common X labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(range(len(agent_names)))
        ax.set_xticklabels([name.replace(' ', '\n') for name in agent_names], fontsize=10)
    
    # Add baseline line
    baseline_idx = agent_names.index("Modular Baseline")
    for ax in [ax1, ax2]:
        ax.axvline(x=baseline_idx, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
    
    plt.tight_layout()
    plt.savefig('reflexion_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print table for reference
    print("\n📊 Table saved as 'reflexion_results.png'")
    print("Table: HumanEval – Final Results")
    print("| Agent | Pass@3 | Pass@1 | Avg Trials | Solved/Total |")
    print("|-------|--------|--------|------------|--------------|")
    for i, name in enumerate(agent_names):
        print(f"| {name[:20]:20} | {pass3_data[i]:6.1f}% | {pass1_data[i]:6.1f}% | {trials_data[i]:9.2f} | {solved_data[i]:11} |")

if __name__ == "__main__":
    main()
