# Vector Memory and Role-Conditioned Multi-Agent Systems: Two Extensions to Improve Reflexion for Language Model Self-Improvement

**Dagani Jesu Sanihit** and **Sanjay Singh** � Manipal Institute of Technology, Udupi, Karnataka, India


This repository contains the official implementation for the paper **"Vector Memory and Role-Conditioned Multi-Agent Systems: Two Extensions to Improve Reflexion for Language Model Self-Improvement"**, which proposes two orthogonal extensions to the [Reflexion](https://arxiv.org/abs/2303.11366) framework (Shinn et al., NeurIPS 2023).

---

## Overview

[Reflexion](https://arxiv.org/abs/2303.11366) improves LLM performance through verbal self-reflection and iterative trial refinement. This work addresses two structural limitations:

| Limitation | Our Solution |
|---|---|
| FIFO sliding window evicts semantically relevant older memories | **Extension 1:** `VectorEpisodicMemory` — Sentence-BERT embeddings with cosine-similarity retrieval |
| Single agent conflates generation, critique, and verification | **Extension 2:** Generator–Critic–Verifier pipeline with shared role-conditioned memory pool |

Both extensions operate **purely at inference time** — no fine-tuning or gradient updates required.

---

## Key Results

### HumanEval (164 tasks, Gemini 2.5 Flash)

| Agent | Pass@3 | Pass@1 | Avg Trials | Recovery |
|---|---|---|---|---|
| ModularBaseline | 89.0% | 81.7% | 1.10 | 40.0% |
| Original Reflexion† | 93.3% | 86.0% | 1.10 | 52.2% |
| VectorReflexion (E1) | 92.7% | 87.2% | 1.09 | 42.9% |
| **MultiAgentReflexion (E2)** | **96.3%** | **93.9%** | **1.03** | **40.0%** |

†Original Reflexion contains a code-cleaning bug neutralised by certain model outputs.

### Statistical Validation

| Comparison | ΔPass@3 | t | p | Cohen's d |
|---|---|---|---|---|
| E1 vs Baseline | +3.7 pp | 2.144 | 0.033* | 0.127 |
| E2 vs Baseline | +7.9 pp | 3.746 | <0.001*** | 0.301 |

95% Wilson CIs: Baseline [84.2%, 93.9%] · E1 [88.7%, 96.7%] · E2 [93.4%, 99.2%]

### Long-Horizon Memory Benchmark (5 sessions, 13 tasks, 9 distractors)

| Metric | Temporal (FIFO) | VectorEpisodicMemory | Δ |
|---|---|---|---|
| Dependency Recall | 50.0% | 100.0% | +50.0 pp |
| Session-5 Success | 0.0% | 100.0% | +100 pp |
| Avg Retrieval Latency | 0.002 ms | 14.3 ms | +14.3 ms |

Zero variance across all 3 independent trials for both conditions.

---

## Repository Structure

```
Vector-Memory-and-Role-Conditioned-Multi-Agent-Systems/
├── reflexion/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # ReflexionAgent (modular baseline)
│   │   ├── original.py          # OriginalReflexionAgent (Shinn et al. replica)
│   │   ├── vector.py            # VectorReflexionAgent (Extension 1)
│   │   └── multiagent.py        # MultiAgentReflexion (Extension 2)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseMemory abstract class
│   │   ├── temporal.py          # TemporalMemory (FIFO baseline)
│   │   └── vector.py            # VectorEpisodicMemory (Extension 1)
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   └── humaneval.py         # HumanEval dataset loader (164 tasks)
│   ├── evaluators/
│   │   ├── __init__.py
│   │   └── code.py              # ObjectiveCodeEvaluator (subprocess, 10s timeout)
│   ├── reflection/
│   │   └── optimizer.py         # Reflection generation utilities
│   ├── config.py                # SecureConfigLoader (env-based API key management)
│   ├── llm.py                   # BaseLLMModel with exponential backoff + embeddings
│   └── memory.py                # SharedMemoryPool for multi-agent use
│
├── experiments/
│   ├── extension1_vector_memory/
│   │   ├── long_horizon_benchmark.py  # 5-session, 13-task long-horizon benchmark
│   │   ├── memory_efficiency.py       # Retrieval latency vs pool size (100–50k)
│   │   ├── reasoning_benchmark.py     # Reasoning quality analysis
│   │   └── retrieval_analysis.py      # Retrieval quality (precision@5)
│   ├── make_results_table.py          # Generate paper tables from saved results
│   ├── run_comparison.py              # Main benchmark entrypoint (Extensions 1 & 2)
│   ├── run_humaneval.py               # Standalone HumanEval runner
│   └── visualize_results.py           # Generate paper figures
│
├── configs/
│   ├── humaneval_baseline_seed42.yaml
│   ├── humaneval_vector_seed42.yaml
│   ├── humaneval_multiagent_seed42.yaml
│   └── memory_efficiency_benchmark.yaml
│
├── scripts/
│   ├── reproduce_all.sh               # Master: reproduce all tables and figures
│   ├── reproduce_humaneval_tables.sh  # Tables 4 and 5
│   ├── reproduce_memory_tables.sh     # Tables 2 and 3
│   ├── reproduce_figures.sh           # Figures 5, 6, 7, 8
│   └── run_all_experiments.sh         # Full rerun from scratch (~8 hours)
│
├── data/
│   └── dataset_instructions.txt       # Dataset access and preprocessing notes
│
├── environment/
│   ├── requirements.txt               # pip dependencies
│   ├── environment.yml                # conda dependencies
│   └── hardware_notes.txt             # Hardware specs and runtime estimates
│
├── results/
│   ├── aggregated_results.csv         # All paper numbers in one file
│   └── seed_runs/                     # Per-experiment raw JSON outputs
│       ├── extension1_vector_agent.json
│       ├── extension2_multiagent_agent.json
│       ├── extension2_memory_efficiency.json
│       ├── extension1_reasoning_benchmark.json
│       ├── humaneval_results.json
│       ├── modular_vs_original.json
│       ├── comparison_results.json
│       └── section4_4_memory_efficiency.json
│
├── HumanEval.jsonl.gz                 # HumanEval dataset (164 tasks, included)
├── config.json.template               # API key configuration template
├── reflexion_framework.ipynb          # Interactive notebook walkthrough
├── .gitignore
└── README.md
```

---

## System Requirements

### Hardware

All experiments were run on a **standard CPU machine**. No GPU is required.

- CPU: Any modern CPU
- RAM: 8 GB recommended
- Storage: ~500 MB (embedding model downloaded automatically on first run)
- GPU: Not required — embeddings run on CPU via PyTorch

See `environment/hardware_notes.txt` for full memory and runtime details.

### Software

- Python 3.9+
- All dependencies listed in `environment/requirements.txt`

---

## Setup

### Using pip

```bash
git clone https://github.com/sanihit76201/Vector-Memory-and-Role-Conditioned-Multi-Agent-Systems.git
cd Vector-Memory-and-Role-Conditioned-Multi-Agent-Systems
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r environment/requirements.txt
```

### Using conda

```bash
conda env create -f environment/environment.yml
conda activate reflexion
```

### API Configuration

```bash
cp config.json.template .env
```

Edit `.env`:

```
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=google/gemini-2.5-flash
GEMINI_API_BASE=https://openrouter.ai/api/v1
RATE_LIMIT_DELAY=0.5
```

> **Never commit your `.env` file.** It is listed in `.gitignore`.

All LLM calls use **Google Gemini 2.5 Flash** via [OpenRouter](https://openrouter.ai).
Sentence-BERT embeddings (`all-MiniLM-L6-v2`, ~90 MB) run locally on CPU — no GPU required.
The embedding model is downloaded automatically from HuggingFace on first run.

---

## Data Access

The HumanEval dataset (`HumanEval.jsonl.gz`) is included in the repository root.
No manual download is required.

The long-horizon memory benchmark is fully procedural — generated at runtime with no
external data needed.

See `data/dataset_instructions.txt` for full details on both benchmarks.

---

## How to Reproduce the Main Results

### Option A: Reproduce from stored results (fast, ~3 minutes, no API calls)

Regenerate all tables and figures from the pre-saved JSON results:

```bash
# Reproduce everything
bash scripts/reproduce_all.sh

# Or reproduce individual components
bash scripts/reproduce_humaneval_tables.sh   # Tables 4 and 5
bash scripts/reproduce_memory_tables.sh      # Tables 2 and 3
bash scripts/reproduce_figures.sh            # Figures 5, 6, 7, 8
```

Outputs are saved to `outputs/tables/` and `outputs/figures/`.

### Option B: Rerun experiments from scratch (~8 hours, requires API key)

```bash
bash scripts/run_all_experiments.sh
```

Or run individual extensions:

```bash
cd experiments

# Extension 1: VectorReflexion (164 tasks, ~1.5 hours)
python run_comparison.py --extension vector --tasks 164

# Extension 2: MultiAgentReflexion (164 tasks, ~4.5 hours)
python run_comparison.py --extension multiagent --tasks 164
```

For a quick smoke test (10 tasks, ~5 minutes):

```bash
python run_comparison.py --extension vector --tasks 10
python run_comparison.py --extension multiagent --tasks 10
```

### Long-Horizon Memory Benchmark

```bash
cd experiments/extension1_vector_memory
python long_horizon_benchmark.py --trials 3
```

Runs 3 independent trials of the 5-session, 13-task benchmark comparing
TemporalMemory vs. VectorEpisodicMemory. Results saved to `results/seed_runs/`.

### Memory Efficiency Benchmark

```bash
cd experiments/extension1_vector_memory
python memory_efficiency.py
```

Measures retrieval latency across pool sizes 100–50,000. Takes ~2 minutes.

---

## Mapping Between Paper Results and Scripts

| Paper Element | Reproduction Script | Source Data |
|---|---|---|
| Table 2 (Long-horizon benchmark) | `scripts/reproduce_memory_tables.sh` | `results/seed_runs/section4_4_memory_efficiency.json` |
| Table 3 (Retrieval latency) | `scripts/reproduce_memory_tables.sh` | `results/seed_runs/extension2_memory_efficiency.json` |
| Table 4 (HumanEval performance) | `scripts/reproduce_humaneval_tables.sh` | `results/seed_runs/extension2_multiagent_agent.json` |
| Table 5 (Statistical validation) | `scripts/reproduce_humaneval_tables.sh` | `results/seed_runs/extension2_multiagent_agent.json` |
| Figure 5 (Session success rate) | `scripts/reproduce_figures.sh` | `results/seed_runs/extension1_vector_agent.json` |
| Figure 6 (Latency vs pool size) | `scripts/reproduce_figures.sh` | `results/seed_runs/extension2_memory_efficiency.json` |
| Figure 7 (HumanEval bar chart) | `scripts/reproduce_figures.sh` | `results/seed_runs/extension2_multiagent_agent.json` |
| Figure 8 (Wilson CI plot) | `scripts/reproduce_figures.sh` | `results/seed_runs/extension2_multiagent_agent.json` |

---

## Architecture

### Extension 1: VectorEpisodicMemory

Replaces the FIFO sliding window with Sentence-BERT (`all-MiniLM-L6-v2`, 384-dim) embeddings.
At retrieval time, cosine similarity is computed between the current task query and all stored
reflections, returning the top-k most semantically relevant ones regardless of when they were stored:

```
sim(q, rᵢ) = φ(q)ᵀ φ(rᵢ) / (‖φ(q)‖ · ‖φ(rᵢ)‖)
```

Retrieval overhead is ~14 ms constant up to 50,000 entries — less than 2.8% of LLM call latency.

```python
from reflexion.memory.vector import VectorEpisodicMemory

memory = VectorEpisodicMemory(llm, max_size=1000)
memory.add_reflection("off-by-one error in loop boundary")

# Returns semantically relevant reflections, not just recent ones
relevant = memory.get_relevant_memories("current task prompt", k=5)
```

### Extension 2: Multi-Agent Reflexion

Three specialised agents share a single `VectorEpisodicMemory` pool:

```
Task → [Generator] → candidate code c
             ↓
        [Critic] → structured critique (no code generation)
             ↓
       [Verifier] → final submission c*
             ↓
        Evaluator → Pass / Fail → all three agents reflect
```

On failure, all three agents write role-prefixed reflections (`[Generator]`, `[Critic]`,
`[Verifier]`) back to the shared pool. Each agent queries the pool with a role-conditioned
prompt to bias retrieval toward role-relevant past experience.

```python
from reflexion.agents.multiagent import MultiAgentReflexion

agent = MultiAgentReflexion(llm, max_trials=3)
result = agent.solve_task(task)
# result = {"success": True, "trials": 1, "code": "..."}
```

---

## Hyperparameters

| Component | Parameter | Value |
|---|---|---|
| LLM | Model | `google/gemini-2.5-flash` |
| LLM | Temperature | 0.7 |
| LLM | Max tokens | 2048 |
| LLM | Inter-request delay | 0.5 s |
| Backoff | Max retries / Initial delay / Factor | 5 / 5 s / 2.5× |
| TemporalMemory | Buffer size / Top-k / Eviction | 10 / 3 / FIFO |
| VectorEpisodicMemory | Encoder | `all-MiniLM-L6-v2` |
| VectorEpisodicMemory | Embedding dim / Max pool / Top-k | 384 / 1,000 (2,000 for HumanEval) / 5 |
| MultiAgentReflexion | Top-k per agent | 3 |
| HumanEval | Max trials / Subprocess timeout | 3 / 10 s |
| Long-horizon benchmark | Sessions / Total tasks / Distractors / Trials | 5 / 13 / 9 / 3 |

No hyperparameter search was performed. All values were set a priori based on Reflexion
paper defaults where applicable.

---

## Experimental Protocol

- **Dataset:** All 164 HumanEval tasks used; no tasks excluded
- **Evaluation:** Per-task binary success indicator (1 if solved within 3 trials, 0 otherwise)
- **Statistical tests:** Paired t-tests on 164 paired binary indicators
- **Effect size:** Cohen's d using pooled standard deviation on paired differences
- **Confidence intervals:** 95% Wilson score CIs (`statsmodels`, method='wilson')
- **Significance threshold:** p < 0.05
- **Long-horizon benchmark:** 3 independent trials; results are deterministic (zero variance observed)

---

## Reproducibility Notes

- Minor numerical variation may occur across hardware and software environments
- Runtime varies with API response latency and network conditions
- The embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded automatically on first run
- The long-horizon benchmark is fully deterministic given the fixed task sequence
- `VectorEpisodicMemory` max_size is set to 2,000 for HumanEval to accommodate
  the full reflection budget of 164 × 3 × 3 = 1,476 role-prefixed entries

---

## Reviewer Checklist

- [x] Environment specification (`environment/requirements.txt`, `environment/environment.yml`)
- [x] Hardware notes (`environment/hardware_notes.txt`)
- [x] Dataset preparation instructions (`data/dataset_instructions.txt`)
- [x] Configuration files for all experiments (`configs/`)
- [x] Training and evaluation scripts (`experiments/`)
- [x] Reproduction scripts for all tables and figures (`scripts/`)
- [x] Pre-saved result files (`results/seed_runs/`)
- [x] Aggregated results CSV (`results/aggregated_results.csv`)
- [x] Interactive notebook walkthrough (`reflexion_framework.ipynb`)

---

## Citation

```bibtex
@article{sanihit2026vectormemory,
  title   = {Vector Memory and Role-Conditioned Multi-Agent Systems:
             Two Extensions to Improve Reflexion for Language Model
             Self-Improvement},
  author  = {Dagani Jesu Sanihit and Sanjay Singh},
  journal = {Transactions on Machine Learning Research},
  year    = {2026}
}
```

---

## Acknowledgements

This work builds upon [Reflexion](https://arxiv.org/abs/2303.11366) by Shinn et al. (NeurIPS 2023).
Sentence-BERT embeddings use the [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
model from Reimers & Gurevych (EMNLP 2019).
All LLM calls use Google Gemini 2.5 Flash via OpenRouter.

---

## License

This repository is released under the MIT License.
See [LICENSE](LICENSE) for details.

