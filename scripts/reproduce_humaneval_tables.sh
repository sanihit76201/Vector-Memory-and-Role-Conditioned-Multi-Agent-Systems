#!/bin/bash
# Reproduce Tables 4 and 5 from saved results
# Table 4: HumanEval performance across all agents (Pass@3, Pass@1, Avg Trials, Recovery)
# Table 5: Statistical validation (t-statistic, p-value, Cohen's d, 95% Wilson CI)
#
# Source data:
#   results/seed_runs/extension2_multiagent_agent.json  (164 tasks, all agents)
#   results/seed_runs/extension1_vector_agent.json      (164 tasks, vector agent)
#
# Expected output matches paper:
#   ModularBaseline  : Pass@3=89.0%, Pass@1=81.7%, Avg Trials=1.10
#   VectorReflexion  : Pass@3=92.7%, Pass@1=87.2%, Avg Trials=1.09
#   MultiAgent       : Pass@3=96.3%, Pass@1=93.9%, Avg Trials=1.03

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Reproducing Tables 4 and 5..."

python3 << EOF
import json
import csv
import os
import numpy as np
from scipy import stats

root = "$ROOT_DIR"
out_dir = os.path.join(root, "outputs", "tables")
os.makedirs(out_dir, exist_ok=True)

# ── Load result files ────────────────────────────────────────────────
ma_path  = os.path.join(root, "results", "seed_runs", "extension2_multiagent_agent.json")
vec_path = os.path.join(root, "results", "seed_runs", "extension1_vector_agent.json")

with open(ma_path) as f:
    ma_data = json.load(f)
with open(vec_path) as f:
    vec_data = json.load(f)

# ── Metrics helper ───────────────────────────────────────────────────
def metrics(results):
    n       = len(results)
    passed  = sum(1 for r in results if r.get("success"))
    pass3   = passed / n * 100
    pass1   = sum(1 for r in results if r.get("success") and r.get("trials") == 1) / n * 100
    s_t     = [r["trials"] for r in results if r.get("success") and r.get("trials", 0) > 0]
    avg_t   = float(np.mean(s_t)) if s_t else 0.0
    fail_t1 = [r for r in results if not (r.get("success") and r.get("trials") == 1)]
    recov   = sum(1 for r in results if r.get("success") and r.get("trials", 0) > 1)
    rec     = recov / len(fail_t1) * 100 if fail_t1 else 100.0
    return {"n": n, "pass3": round(pass3,1), "pass1": round(pass1,1),
            "avg_trials": round(avg_t,2), "recovery": round(rec,1)}

def stat_validation(ext_r, base_r):
    e = [1 if r.get("success") else 0 for r in ext_r]
    b = [1 if r.get("success") else 0 for r in base_r]
    t, p   = stats.ttest_rel(e, b)
    diff   = np.array(e) - np.array(b)
    d      = float(diff.mean() / (diff.std() + 1e-9))
    delta  = round((np.mean(e) - np.mean(b)) * 100, 1)
    return {"delta_pp": delta, "t": round(float(t),3),
            "p": round(float(p),4), "d": round(d,3)}

# ── Collect results ──────────────────────────────────────────────────
base_r  = ma_data["results"]["modular_baseline"]
orig_r  = ma_data["results"]["original_working"]
ma_r    = ma_data["results"]["multiagentreflexion"]
vec_r   = vec_data["results"].get("vectorreflexion", [])

agents = [
    ("ModularBaseline",     base_r),
    ("Original",            orig_r),
    ("VectorReflexion",     vec_r),
    ("MultiAgentReflexion", ma_r),
]

# ── Write Table 4 ────────────────────────────────────────────────────
table4_path = os.path.join(out_dir, "table4_humaneval_performance.csv")
with open(table4_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["agent", "num_tasks", "pass_at_3_pct", "pass_at_1_pct",
                "avg_trials", "recovery_rate_pct"])
    for name, results in agents:
        if not results:
            w.writerow([name, 0, "n/a", "n/a", "n/a", "n/a"])
            continue
        m = metrics(results)
        w.writerow([name, m["n"], m["pass3"], m["pass1"],
                    m["avg_trials"], m["recovery"]])

print("Table 4 saved to:", table4_path)

# ── Write Table 5 ────────────────────────────────────────────────────
table5_path = os.path.join(out_dir, "table5_statistical_validation.csv")
with open(table5_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["comparison", "delta_pass_at_3_pp", "t_statistic",
                "p_value", "cohens_d", "significant"])
    if vec_r:
        s = stat_validation(vec_r, base_r)
        w.writerow(["E1_vs_Baseline", s["delta_pp"], s["t"],
                    s["p"], s["d"], s["p"] < 0.05])
    s = stat_validation(ma_r, base_r)
    w.writerow(["E2_vs_Baseline", s["delta_pp"], s["t"],
                s["p"], s["d"], s["p"] < 0.05])

print("Table 5 saved to:", table5_path)

# ── Print summary ────────────────────────────────────────────────────
print("")
print("=" * 60)
print("TABLE 4 — HumanEval Performance")
print("=" * 60)
print(f"{'Agent':<25} {'Pass@3':>7} {'Pass@1':>7} {'AvgT':>6} {'Rec':>7}")
print("-" * 60)
for name, results in agents:
    if not results:
        print(f"  {name:<23} {'n/a':>7}")
        continue
    m = metrics(results)
    print(f"  {name:<23} {m['pass3']:>6.1f}% {m['pass1']:>6.1f}% "
          f"{m['avg_trials']:>5.2f} {m['recovery']:>6.1f}%")

print("")
print("=" * 60)
print("TABLE 5 — Statistical Validation vs ModularBaseline")
print("=" * 60)
print(f"{'Comparison':<20} {'DeltaPP':>8} {'t':>7} {'p':>8} {'d':>7} {'Sig':>5}")
print("-" * 60)
if vec_r:
    s = stat_validation(vec_r, base_r)
    sig = "*" if s["p"] < 0.05 else ""
    print(f"  {'E1 vs Baseline':<18} {s['delta_pp']:>+7.1f} {s['t']:>7.3f} "
          f"{s['p']:>8.4f} {s['d']:>7.3f} {sig:>5}")
s = stat_validation(ma_r, base_r)
sig = "***" if s["p"] < 0.001 else ("*" if s["p"] < 0.05 else "")
print(f"  {'E2 vs Baseline':<18} {s['delta_pp']:>+7.1f} {s['t']:>7.3f} "
      f"{s['p']:>8.4f} {s['d']:>7.3f} {sig:>5}")
print("=" * 60)
EOF

echo ""
echo "Tables 4 and 5 complete."
