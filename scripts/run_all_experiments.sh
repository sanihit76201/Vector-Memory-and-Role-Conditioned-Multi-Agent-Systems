#!/bin/bash
# Run all main experiments from scratch
# Paper: Scalable Reflexion: Semantic Memory and Multi-Agent Collaboration
#
# WARNING: This will make real API calls to OpenRouter (Gemini 2.5 Flash).
# Estimated cost: ~$0.25 for all experiments combined.
# Estimated time: ~8 hours total (dominated by API rate limiting at 0.5s/call).
#
# For fast reproduction from pre-saved results use:
#   bash scripts/reproduce_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "Scalable Reflexion — Full Experiment Runner"
echo "============================================================"
echo ""
echo "This will run:"
echo "  1. ModularBaseline    on 164 HumanEval tasks (~1.5 hours)"
echo "  2. VectorReflexion    on 164 HumanEval tasks (~1.5 hours)"
echo "  3. MultiAgentReflexion on 164 HumanEval tasks (~4.5 hours)"
echo "  4. Long-horizon memory benchmark             (~10 minutes)"
echo "  5. Memory efficiency scaling benchmark       (~5 minutes)"
echo ""
echo "WARNING: This requires a valid OpenRouter API key in .env"
echo "WARNING: Estimated API cost ~\$0.25"
echo "WARNING: Estimated total time ~8 hours"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."

# Check .env exists
if [ ! -f "$ROOT_DIR/.env" ]; then
    echo "ERROR: .env file not found."
    echo "Copy config.json.template to .env and fill in your API key."
    exit 1
fi

# Create output directories
mkdir -p "$ROOT_DIR/results/seed_runs"
mkdir -p "$ROOT_DIR/outputs/tables"
mkdir -p "$ROOT_DIR/outputs/figures"

cd "$ROOT_DIR/experiments"

# ── Experiment 1: ModularBaseline + VectorReflexion (Extension 1) ───
echo ""
echo "============================================================"
echo "Experiment 1/3: VectorReflexion (Extension 1)"
echo "============================================================"
python3 run_comparison.py --extension vector --tasks 164
echo "Extension 1 complete. Results saved to results/seed_runs/"

# ── Experiment 2: ModularBaseline + MultiAgentReflexion (Extension 2)
echo ""
echo "============================================================"
echo "Experiment 2/3: MultiAgentReflexion (Extension 2)"
echo "============================================================"
python3 run_comparison.py --extension multiagent --tasks 164
echo "Extension 2 complete. Results saved to results/seed_runs/"

# ── Experiment 3: Long-horizon memory benchmark ──────────────────────
echo ""
echo "============================================================"
echo "Experiment 3/5: Long-Horizon Memory Benchmark"
echo "============================================================"
cd "$ROOT_DIR/experiments/extension1_vector_memory"
python3 run_long_horizon_benchmark.py --trials 3
echo "Long-horizon benchmark complete."

# ── Experiment 4: Memory efficiency scaling ──────────────────────────
echo ""
echo "============================================================"
echo "Experiment 4/5: Memory Efficiency Scaling"
echo "============================================================"
python3 run_memory_efficiency.py
echo "Memory efficiency benchmark complete."

# ── Move results to seed_runs ────────────────────────────────────────
cd "$ROOT_DIR"
echo ""
echo "Moving results to results/seed_runs/..."
for f in results/*.json; do
    [ -f "$f" ] && mv "$f" results/seed_runs/
done

# ── Reproduce tables and figures from new results ────────────────────
echo ""
echo "============================================================"
echo "Generating tables and figures..."
echo "============================================================"
bash "$SCRIPT_DIR/reproduce_all.sh"

echo ""
echo "============================================================"
echo "All experiments complete!"
echo "============================================================"
echo ""
echo "Results saved to : results/seed_runs/"
echo "Tables saved to  : outputs/tables/"
echo "Figures saved to : outputs/figures/"
