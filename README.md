# Scalable Reflexion: Semantic Memory and Multi-Agent Collaboration for Verbal Reinforcement Learning

**Dagani Jesu Sanihit** — Manipal Institute of Technology  
*Transactions on Machine Learning Research (under review)*

---

## Overview

This repository contains the code for **two orthogonal extensions** to the [Reflexion](https://arxiv.org/abs/2303.11366) framework:

| Extension | Module | Key Result |
|---|---|---|
| **Extension 1** — VectorEpisodicMemory | `reflexion/memory/vector.py` | 100% long-horizon recall vs. 0% for FIFO (+50 pp, zero variance) |
| **Extension 2** — MultiAgentReflexion | `reflexion/agents/multiagent.py` | Pass@3 = 96.3%, Pass@1 = 93.9% on HumanEval (+7.9 pp, +9.8 pp) |

Both extensions operate **purely at inference time** — no fine-tuning or gradient updates required.

---

## Repository Structure

```
reflexion/
├── agents/
│   ├── base.py            # ModularBaseline — clean single-agent Reflexion
│   ├── multiagent.py      # Extension 2: Generator–Critic–Verifier pipeline
│   └── vector.py          # Extension 1: VectorReflexionAgent
│
├── memory/
│   ├── base.py            # BaseMemory abstract class
│   ├── temporal.py        # TemporalMemory (FIFO baseline)
│   └── vector.py          # VectorEpisodicMemory (Sentence-BERT retrieval)
│
├── benchmarks/
│   └── humaneval.py       # HumanEval task loader (164 tasks)
│
├── evaluators/
│   └── code.py            # ObjectiveCodeEvaluator (subprocess, 10s timeout)
│
├── reflection/
│   └── optimizer.py       # Reflection generation utilities
│
├── config.py              # SecureConfigLoader (env-based API key management)
├── llm.py                 # BaseLLMModel with exponential backoff
└── memory.py              # SharedMemoryPool for multi-agent use
│
experiments/
└── extension1_vector_memory/
    ├── long_horizon_benchmark.py   # 5-session, 13-task long-horizon benchmark
    ├── memory_efficiency.py        # Retrieval latency vs. pool size (100–50k)
    └── retrieval_analysis.py       # Retrieval quality analysis (precision@5)
│
results/
    ├── extension1_vector_agent.json       # VectorReflexion HumanEval results
    ├── extension2_multiagent_agent.json   # MultiAgent HumanEval results
    ├── humaneval_results.json             # All agents, full task-level results
    └── modular_vs_original.json          # Baseline vs. original Reflexion
│
run_humaneval.py       # Main HumanEval evaluation script
run_comparison.py      # Side-by-side agent comparison
HumanEval.jsonl.gz     # HumanEval dataset (164 tasks)
requirements.txt       # Python dependencies
.env.example           # API key template (copy to .env and fill in)
config.json.template   # Configuration template
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/[YOUR_USERNAME]/scalable-reflexion.git
cd scalable-reflexion
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
OPENROUTER_API_KEY=your_key_here
```

All LLM calls use **Google Gemini 2.5 Flash** via [OpenRouter](https://openrouter.ai).  
Sentence-BERT embeddings run locally on CPU — no GPU required.

---

## Reproducing Paper Results

### HumanEval Benchmark (Table 3 & 4)

Run all agents on 164 HumanEval tasks:

```bash
# Modular Baseline (TemporalMemory)
python run_humaneval.py --agent baseline

# Extension 1: VectorReflexionAgent
python run_humaneval.py --agent vector

# Extension 2: MultiAgentReflexion
python run_humaneval.py --agent multiagent

# Side-by-side comparison
python run_comparison.py
```

Results are saved to `results/`. Expected runtimes at 0.5s inter-request delay:
- Baseline / VectorAgent: ~25 minutes for 164 tasks
- MultiAgent: ~75 minutes (3× LLM calls per trial)

### Long-Horizon Memory Benchmark (Table 1 & Figure 3)

```bash
python experiments/extension1_vector_memory/long_horizon_benchmark.py
```

Runs 3 independent trials of the 5-session, 13-task benchmark comparing TemporalMemory vs. VectorEpisodicMemory.

### Memory Efficiency Benchmark (Table 2 & Figure 4)

```bash
python experiments/extension1_vector_memory/memory_efficiency.py
```

Measures retrieval latency across pool sizes 100–50,000. Takes ~2 minutes.

### Retrieval Quality Analysis (Section 5.2)

```bash
python experiments/extension1_vector_memory/retrieval_analysis.py
```

Injects 3 relevant reflections among 997 irrelevant entries and measures top-5 precision.

---

## Key Hyperparameters

| Component | Parameter | Value |
|---|---|---|
| LLM | Model | `google/gemini-2.5-flash` |
| LLM | Temperature | 0.7 |
| LLM | Max tokens | 2048 |
| LLM | Inter-request delay | 0.5s |
| Backoff | Max retries | 5 |
| Backoff | Initial delay | 5s |
| Backoff | Factor | 2.5× |
| TemporalMemory | Buffer size | 10 |
| TemporalMemory | Top-k | 3 |
| VectorEpisodicMemory | Encoder | `all-MiniLM-L6-v2` |
| VectorEpisodicMemory | Embedding dim | 384 |
| VectorEpisodicMemory | Max pool size | 1,000 |
| VectorEpisodicMemory | Top-k | 5 |
| MultiAgentReflexion | Top-k per agent | 3 |
| HumanEval | Max trials | 3 |
| HumanEval | Subprocess timeout | 10s |

---

## Architecture

### Extension 1: VectorEpisodicMemory

Replaces the FIFO sliding window with Sentence-BERT embedding retrieval:

```python
from reflexion.memory.vector import VectorEpisodicMemory

memory = VectorEpisodicMemory(max_size=1000, top_k=5)
memory.add_reflection("task_id", "reflection text about off-by-one error")

# Retrieves semantically relevant reflections, not just recent ones
relevant = memory.get_relevant_memories("current task prompt")
```

### Extension 2: MultiAgentReflexion

Three-agent Generator–Critic–Verifier pipeline with shared memory:

```python
from reflexion.agents.multiagent import MultiAgentReflexion

agent = MultiAgentReflexion(max_trials=3, top_k=3)
result = agent.solve(task_prompt, test_suite)
# result = {"success": True, "trials": 1, "code": "..."}
```

---

## Results Summary

### Long-Horizon Memory Benchmark

| Metric | TemporalMemory | VectorEpisodicMemory | Δ |
|---|---|---|---|
| Dependency Recall (%) | 50.0 | **100.0** | +50.0 pp |
| Session-5 Success (%) | 0.0 | **100.0** | +100 pp |
| Avg Retrieval Latency (ms) | 0.0 | 30.9 | +30.9 |

### HumanEval (164 tasks, Gemini 2.5 Flash)

| Agent | Pass@3 | Pass@1 | Avg Trials |
|---|---|---|---|
| ModularBaseline | 89.0% | 81.7% | 1.10 |
| VectorReflexion (E1) | 92.7% | 87.2% | 1.09 |
| **MultiAgentReflexion (E2)** | **96.3%** | **93.9%** | **1.03** |

Statistical validation: E1 vs Baseline *p*=0.033 (*d*=0.127); E2 vs Baseline *p*<0.001 (*d*=0.301).

---

## Citation

```bibtex
@article{sanihit2024scalablereflexion,
  title   = {Scalable Reflexion: Semantic Memory and Multi-Agent
             Collaboration for Verbal Reinforcement Learning},
  author  = {Sanihit, Dagani Jesu},
  journal = {Transactions on Machine Learning Research},
  year    = {2024}
}
```

---

## License

This repository is released under the MIT License.  
See [LICENSE](LICENSE) for details.
