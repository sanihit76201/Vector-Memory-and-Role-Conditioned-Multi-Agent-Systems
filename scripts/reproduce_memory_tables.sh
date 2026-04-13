set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Reproducing Tables 2 and 3..."

python3 << EOF
import json
import csv
import os
import numpy as np

root = "$ROOT_DIR"
out_dir = os.path.join(root, "outputs", "tables")
os.makedirs(out_dir, exist_ok=True)

# ── Table 2: Long-horizon benchmark ─────────────────────────────────
lh_path = os.path.join(root, "results", "seed_runs", "extension1b_long_horizon.json")

table2_path = os.path.join(out_dir, "table2_long_horizon.csv")
if os.path.exists(lh_path):
    with open(lh_path) as f:
        lh_data = json.load(f)

    summary = lh_data.get("summary", {})
    t_sum   = summary.get("temporal", {})
    v_sum   = summary.get("vector",   {})

    with open(table2_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "temporal", "vector", "delta"])
        rows = [
            ("dependency_recall_pct",
             t_sum.get("mean_dep_recall_pct", 0),
             v_sum.get("mean_dep_recall_pct", 0)),
            ("session5_success_pct",
             t_sum.get("mean_s5_success_pct", 0),
             v_sum.get("mean_s5_success_pct", 0)),
            ("avg_memories_retrieved",
             t_sum.get("mean_avg_memories", 0),
             v_sum.get("mean_avg_memories", 0)),
            ("avg_retrieval_latency_ms",
             t_sum.get("mean_retrieval_ms", 0),
             v_sum.get("mean_retrieval_ms", 0)),
        ]
        for metric, t_val, v_val in rows:
            delta = round(v_val - t_val, 2)
            w.writerow([metric, round(t_val,2), round(v_val,2), delta])

    print("Table 2 saved to:", table2_path)

    print("")
    print("=" * 60)
    print("TABLE 2 — Long-Horizon Memory Benchmark")
    print("=" * 60)
    print(f"{'Metric':<35} {'Temporal':>10} {'Vector':>10} {'Delta':>8}")
    print("-" * 60)
    for metric, t_val, v_val in rows:
        delta = v_val - t_val
        print(f"  {metric:<33} {t_val:>9.1f} {v_val:>9.1f} {delta:>+7.1f}")
    print("=" * 60)
else:
    print(f"WARNING: {lh_path} not found.")
    print("Run: python experiments/extension1_vector_memory/run_long_horizon_benchmark.py")

# ── Table 3: Memory efficiency ───────────────────────────────────────
me_path = os.path.join(root, "results", "seed_runs", "extension2_memory_efficiency.json")

table3_path = os.path.join(out_dir, "table3_memory_efficiency.csv")
if os.path.exists(me_path):
    with open(me_path) as f:
        me_data = json.load(f)

    temporal_r = me_data.get("temporal", [])
    vector_r   = me_data.get("vector",   [])

    with open(table3_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool_size", "temporal_latency_ms", "vector_latency_ms",
                    "vector_sigma_ms"])
        for t, v in zip(temporal_r, vector_r):
            w.writerow([
                t.get("memory_size", t.get("size", "")),
                round(t.get("retrieval_latency_ms_mean",
                      t.get("avg_latency_ms", 0)), 4),
                round(v.get("retrieval_latency_ms_mean",
                      v.get("avg_latency_ms", 0)), 3),
                round(v.get("retrieval_latency_ms_std",
                      v.get("std_latency_ms", 0)), 3),
            ])

    print("")
    print("Table 3 saved to:", table3_path)

    print("")
    print("=" * 60)
    print("TABLE 3 — Retrieval Latency vs Pool Size")
    print("=" * 60)
    print(f"{'Pool Size':>10} {'Temporal (ms)':>15} {'Vector (ms)':>13} {'±σ':>8}")
    print("-" * 60)
    for t, v in zip(temporal_r, vector_r):
        size    = t.get("memory_size", t.get("size", ""))
        t_lat   = t.get("retrieval_latency_ms_mean", t.get("avg_latency_ms", 0))
        v_lat   = v.get("retrieval_latency_ms_mean", v.get("avg_latency_ms", 0))
        v_sigma = v.get("retrieval_latency_ms_std",  v.get("std_latency_ms", 0))
        print(f"  {size:>8} {t_lat:>14.4f} {v_lat:>13.2f} {v_sigma:>7.2f}")
    print("=" * 60)
else:
    print(f"WARNING: {me_path} not found.")
    print("Run: python experiments/extension1_vector_memory/run_memory_efficiency.py")
EOF

echo ""
echo "Tables 2 and 3 complete."
