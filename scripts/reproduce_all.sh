#!/bin/bash
# Master reproduction script
# Paper: Scalable Reflexion: Semantic Memory and Multi-Agent Collaboration
# Author: Dagani Jesu Sanihit, Manipal Institute of Technology
#
# This script reproduces all main tables and figures from the paper
# using pre-saved result files in results/seed_runs/
#
# Estimated time: 2-3 minutes (reads from saved results, no API calls)
# For full re-run from scratch: bash scripts/run_all_experiments.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "Scalable Reflexion — Full Reproduction"
echo "============================================================"
echo ""
echo "This script will reproduce:"
echo "  - Table 2: Long-horizon memory benchmark results"
echo "  - Table 3: Memory efficiency and retrieval latency"
echo "  - Table 4: HumanEval performance across all agents"
echo "  - Table 5: Statistical validation (t-test, Cohen's d, 95% CI)"
echo "  - Figure 5: Session-level task success rate"
echo "  - Figure 6: Retrieval latency vs pool size"
echo "  - Figure 7: HumanEval Pass@3 and Pass@1 bar chart"
echo "  - Figure 8: Pass@3 with 95% Wilson CIs"
echo ""
echo "Estimated time: 2-3 minutes"
echo "============================================================"
echo ""

# Check Python is available
python3 --version > /dev/null 2>&1 || {
    echo "ERROR: python3 not found. Please install Python 3.9+."
    exit 1
}

# Check results directory exists
if [ ! -d "$ROOT_DIR/results/seed_runs" ]; then
    echo "ERROR: results/seed_runs/ not found."
    echo "Please run bash scripts/run_all_experiments.sh first."
    exit 1
fi

# Create output directories
mkdir -p "$ROOT_DIR/outputs/tables"
mkdir -p "$ROOT_DIR/outputs/figures"

# Step 1: HumanEval performance tables (Tables 4-5)
echo "------------------------------------------------------------"
echo "Step 1/3: HumanEval Tables (Tables 4 and 5)"
echo "------------------------------------------------------------"
bash "$SCRIPT_DIR/reproduce_humaneval_tables.sh"

# Step 2: Memory benchmark tables (Tables 2-3)
echo ""
echo "------------------------------------------------------------"
echo "Step 2/3: Memory Benchmark Tables (Tables 2 and 3)"
echo "------------------------------------------------------------"
bash "$SCRIPT_DIR/reproduce_memory_tables.sh"

# Step 3: All figures (Figures 5-8)
echo ""
echo "------------------------------------------------------------"
echo "Step 3/3: Figures (Figures 5, 6, 7, 8)"
echo "------------------------------------------------------------"
bash "$SCRIPT_DIR/reproduce_figures.sh"

echo ""
echo "============================================================"
echo "Reproduction complete!"
echo "============================================================"
echo ""
echo "Generated outputs:"
echo "  outputs/tables/table2_long_horizon.csv"
echo "  outputs/tables/table3_memory_efficiency.csv"
echo "  outputs/tables/table4_humaneval_performance.csv"
echo "  outputs/tables/table5_statistical_validation.csv"
echo "  outputs/figures/figure5_session_success.png"
echo "  outputs/figures/figure6_latency_scaling.png"
echo "  outputs/figures/figure7_humaneval_bars.png"
echo "  outputs/figures/figure8_wilson_ci.png"
echo ""
echo "All numbers should match Tables 2-5 and Figures 5-8 in the paper."
echo "============================================================"
