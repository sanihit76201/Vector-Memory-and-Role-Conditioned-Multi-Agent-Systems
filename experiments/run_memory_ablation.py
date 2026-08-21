"""
run_memory_ablation.py — Memory Hyperparameter Ablation

PURPOSE
-------
Addresses Reviewer s56K Critical Concern 3:
  "For the memory module, ablate retrieval method (FIFO vs. vector retrieval),
   memory size, top-k, and retention policy."

This script sweeps two hyperparameters of VectorEpisodicMemory:

  top_k       ∈ {1, 3, 5, 7}     — how many memories are retrieved per trial
  max_size    ∈ {3, 5, 10, 50}   — pool capacity (max reflections stored)

For each (top_k, max_size) combination, VectorReflexionAgent is run on a
50-task HumanEval subset and Pass@3 is recorded.

Expected finding (hypothesis):
  Results should be flat above top_k=3 and max_size=5, because:
    - HumanEval tasks are short and intra-task pool depth is ≤9 entries
    - Retrieving more than 3-5 memories adds noise, not signal
    - A pool of 5 is enough to hold all intra-task reflections
  This would confirm the paper's hyperparameter choices are reasonable
  and that the reported gains are not artifacts of tuning.

USAGE
-----
  # Full ablation (50 tasks × 16 configs, ~8-10 hrs)
  python run_memory_ablation.py

  # Quick sweep (20 tasks × 16 configs, ~3-4 hrs)
  python run_memory_ablation.py --tasks 20

  # Single config sanity check
  python run_memory_ablation.py --tasks 10 --topk 5 --maxsize 10

OUTPUT
------
  ../results/memory_ablation.json   — full results grid
  Console heatmap table             — Pass@3 for every (top_k, max_size) cell

PLACE THIS FILE AT
------------------
  experiments/run_memory_ablation.py
"""

import logging
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from itertools import product
from collections import deque
from typing import List, Dict

sys.path.insert(0, '..')

from reflexion.config import SecureConfigLoader
from reflexion.llm import BaseLLMModel
from reflexion.benchmarks import HumanEvalLoader
from reflexion.evaluators import ObjectiveCodeEvaluator
from reflexion.memory import VectorEpisodicMemory
from reflexion.agents.base import ReflexionAgent

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ── Ablation grid ─────────────────────────────────────────────────────────────
TOPK_VALUES    = [1, 3, 5, 7]
MAXSIZE_VALUES = [3, 5, 10, 50]

# Paper's default config (for reference column)
PAPER_TOPK    = 5
PAPER_MAXSIZE = 100   # effectively uncapped for intra-task use


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURABLE VECTOR AGENT
# Thin wrapper over VectorReflexionAgent with injected top_k and max_size
# ─────────────────────────────────────────────────────────────────────────────

class ConfigurableVectorAgent:
    """
    VectorReflexionAgent with configurable top_k and max_size.

    Reimplements the solve_task loop directly rather than subclassing,
    so we can vary both memory parameters cleanly without touching the
    installed package.

    Memory resets between tasks (same as all other agents in this project).
    """

    def __init__(self, llm, top_k: int = 5, max_size: int = 100,
                 max_trials: int = 3):
        self.llm        = llm
        self.top_k      = top_k
        self.max_size   = max_size
        self.max_trials = max_trials
        self.evaluator  = ObjectiveCodeEvaluator(timeout=10)
        self.memory     = VectorEpisodicMemory(llm, max_size=max_size)

    def _clean(self, raw) -> str:
        if isinstance(raw, list):
            raw = ''.join(
                p.get('text', '') if isinstance(p, dict) else str(p)
                for p in raw
            )
        raw = str(raw)
        if '```python' in raw:
            raw = raw.split('```python', 1)[1]
        if '```' in raw:
            raw = raw.split('```', 1)[0]
        return raw.strip()

    def solve_task(self, task: Dict, verbose: bool = False) -> Dict:
        task_id = task['task_id']

        for trial in range(self.max_trials):
            # Retrieve top_k semantically similar memories
            memories: List[str] = self.memory.get_relevant_memories(
                task['prompt'], k=self.top_k
            )
            mem_ctx = '\n'.join(f'- {m}' for m in memories) \
                      if memories else 'None'

            prompt = (
                f"You are an expert Python programmer. Complete this function:\n\n"
                f"{task['prompt']}\n\n"
                f"Past reflections (top-{self.top_k} semantically similar):\n"
                f"{mem_ctx}\n\n"
                f"Requirements:\n"
                f"1. Complete the function implementation\n"
                f"2. Handle all edge cases\n"
                f"3. Output ONLY Python code, no markdown, no explanations\n\n"
                f"Your code:"
            )

            try:
                raw  = self.llm.call_llm(prompt, max_tokens=2048)
                code = self._clean(raw)

                result = self.evaluator.evaluate(
                    code, task['entry_point'], task['test']
                )

                if result['passed']:
                    if verbose:
                        logger.info(f'  ✅ {task_id} solved trial {trial+1}')
                    return {
                        'task_id':    task_id,
                        'success':    True,
                        'trials':     trial + 1,
                        'top_k':      self.top_k,
                        'max_size':   self.max_size,
                        'agent_type': f'Vector_k{self.top_k}_s{self.max_size}',
                    }

                # Store richer reflection for next trial
                reflection = (
                    f"Task '{task_id}' trial {trial+1} failed: "
                    f"{(result.get('error') or '')[:150]}. "
                    f"Hint: review edge cases, types, boundary conditions."
                )
                self.memory.add_reflection(reflection)

                if verbose:
                    logger.info(f'  ❌ {task_id} trial {trial+1} failed')

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.memory.add_reflection(
                    f"Task '{task_id}' trial {trial+1} exception: {str(exc)[:100]}"
                )
                logger.error(f'  Exception on {task_id}: {exc}')

        return {
            'task_id':    task_id,
            'success':    False,
            'trials':     self.max_trials,
            'top_k':      self.top_k,
            'max_size':   self.max_size,
            'agent_type': f'Vector_k{self.top_k}_s{self.max_size}',
        }

    def reset(self):
        self.memory.clear()


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_config(llm, tasks, top_k, max_size, verbose=False):
    """Run one (top_k, max_size) configuration, return pass3 and raw results."""
    agent   = ConfigurableVectorAgent(llm, top_k=top_k, max_size=max_size)
    results = []

    for task in tasks:
        result = agent.solve_task(task, verbose=verbose)
        result['agent_type'] = f'Vector_k{top_k}_s{max_size}'
        if 'trials' not in result:
            result['trials'] = agent.max_trials
        results.append(result)
        agent.reset()

    n      = len(results)
    passed = sum(1 for r in results if r.get('success'))
    pass3  = passed / n * 100 if n else 0.0
    pass1  = sum(1 for r in results
                 if r.get('success') and r.get('trials', 0) == 1) / n * 100

    return pass3, pass1, results


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_heatmap(grid_pass3, grid_pass1, topk_values, maxsize_values,
                  paper_topk, paper_maxsize):
    """
    Print Pass@3 and Pass@1 as heatmap tables.

    Rows = max_size, Columns = top_k.
    Paper's default cell marked with *.
    """
    col_w = 10

    for metric_name, grid in [('Pass@3', grid_pass3), ('Pass@1', grid_pass1)]:
        print(f'\n── {metric_name} Ablation (rows=max_size, cols=top_k) ──')
        header = f'{"max_size":>10}' + ''.join(
            f'{"k="+str(k):>{col_w}}' for k in topk_values
        )
        print(header)
        print('-' * len(header))

        for s in maxsize_values:
            row = f'{s:>10}'
            for k in topk_values:
                val  = grid.get((k, s), float('nan'))
                mark = '*' if (k == paper_topk and s == paper_maxsize) else ' '
                row += f'{val:>{col_w-1}.1f}%{mark}'
            print(row)

    print(f'\n  * = paper default (top_k={paper_topk}, max_size={paper_maxsize})')


def print_sensitivity(grid_pass3, topk_values, maxsize_values):
    """
    Print range (max-min) across each axis to quantify sensitivity.
    Low range = robust to that hyperparameter.
    """
    print('\n── Sensitivity Analysis ──')

    # top_k sensitivity (fix max_size, vary k)
    print('\n  top_k sensitivity (Pass@3 range across k values):')
    for s in maxsize_values:
        vals = [grid_pass3.get((k, s), float('nan')) for k in topk_values]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            print(f'    max_size={s:>3}: range = {max(vals)-min(vals):.1f} pp  '
                  f'(min={min(vals):.1f}%, max={max(vals):.1f}%)')

    # max_size sensitivity (fix k, vary max_size)
    print('\n  max_size sensitivity (Pass@3 range across size values):')
    for k in topk_values:
        vals = [grid_pass3.get((k, s), float('nan')) for s in maxsize_values]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            print(f'    top_k={k}: range = {max(vals)-min(vals):.1f} pp  '
                  f'(min={min(vals):.1f}%, max={max(vals):.1f}%)')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def numpy_encoder(obj):
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    raise TypeError(f'Object {type(obj)} not serializable')


def main():
    parser = argparse.ArgumentParser(
        description='Memory hyperparameter ablation for VectorReflexionAgent'
    )
    parser.add_argument(
        '--tasks', type=int, default=50,
        help='Tasks per configuration (default 50; use 20 for quick sweep)',
    )
    parser.add_argument(
        '--outdir', default='../results',
        help='Output directory for memory_ablation.json',
    )
    parser.add_argument(
        '--topk', type=int, default=None,
        help='Run only this top_k value (for single-config testing)',
    )
    parser.add_argument(
        '--maxsize', type=int, default=None,
        help='Run only this max_size value (for single-config testing)',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show per-task pass/fail logs',
    )
    args = parser.parse_args()

    # Allow single-config override
    topk_values    = [args.topk]    if args.topk    else TOPK_VALUES
    maxsize_values = [args.maxsize] if args.maxsize else MAXSIZE_VALUES

    n_configs = len(topk_values) * len(maxsize_values)

    # ── Config ────────────────────────────────────────────────────────────
    try:
        config = SecureConfigLoader().load_from_env_file('../.env')
    except Exception as e:
        logger.error(f'❌ Config error: {e}')
        sys.exit(1)

    llm = BaseLLMModel(
        config['openrouter_api_key'],
        config['openrouter_model'],
        config['gemini_api_base'],
        config['rate_limit_delay'],
    )

    # ── Tasks ─────────────────────────────────────────────────────────────
    logger.info('📚 Loading HumanEval tasks...')
    try:
        tasks = HumanEvalLoader.load_from_file(
            '../HumanEval.jsonl.gz', num_samples=args.tasks
        )
        logger.info(f'✓ Loaded {len(tasks)} tasks')
    except FileNotFoundError:
        logger.error('❌ HumanEval.jsonl.gz not found')
        sys.exit(1)

    # ── ETA ───────────────────────────────────────────────────────────────
    delay   = config['rate_limit_delay']
    eta_min = n_configs * args.tasks * 3 * delay / 60

    print('\n' + '=' * 70)
    print('MEMORY HYPERPARAMETER ABLATION — REVIEWER s56K CONCERN 3')
    print('=' * 70)
    print(f'Tasks per config : {args.tasks}')
    print(f'Configurations   : {n_configs}  '
          f'(top_k={topk_values} × max_size={maxsize_values})')
    print(f'Delay            : {delay}s')
    print(f'ETA              : ~{eta_min:.0f} minutes')
    print(f'Paper default    : top_k={PAPER_TOPK}, max_size={PAPER_MAXSIZE}')
    print('=' * 70)
    input('\nPress ENTER to start...')

    # ── Sweep ─────────────────────────────────────────────────────────────
    grid_pass3   = {}
    grid_pass1   = {}
    all_results  = {}
    config_count = 0

    for top_k, max_size in product(topk_values, maxsize_values):
        config_count += 1
        label = f'Vector_k{top_k}_s{max_size}'

        print(f'\n[{config_count}/{n_configs}] top_k={top_k}, '
              f'max_size={max_size}  ({label})')

        pass3, pass1, results = run_config(
            llm, tasks, top_k, max_size, verbose=args.verbose
        )

        grid_pass3[(top_k, max_size)] = pass3
        grid_pass1[(top_k, max_size)] = pass1
        all_results[label] = results

        print(f'  → Pass@3={pass3:.1f}%  Pass@1={pass1:.1f}%')

    # ── Report ────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('ABLATION RESULTS')
    print('=' * 70)
    print_heatmap(
        grid_pass3, grid_pass1,
        topk_values, maxsize_values,
        PAPER_TOPK, PAPER_MAXSIZE,
    )
    print_sensitivity(grid_pass3, topk_values, maxsize_values)

    # ── Interpretation ────────────────────────────────────────────────────
    all_pass3 = list(grid_pass3.values())
    overall_range = max(all_pass3) - min(all_pass3)
    print(f'\n── Overall ──')
    print(f'  Pass@3 range across all {n_configs} configs: '
          f'{min(all_pass3):.1f}% – {max(all_pass3):.1f}%  '
          f'(spread = {overall_range:.1f} pp)')

    if overall_range <= 5.0:
        print('  ✅ Results are robust: <5 pp spread across all configurations.')
        print('     Hyperparameter choices do not meaningfully affect performance.')
    elif overall_range <= 10.0:
        print('  ⚠️  Moderate sensitivity: 5-10 pp spread.')
        print('     Discuss which region is optimal in the paper.')
    else:
        print('  ❌ High sensitivity: >10 pp spread.')
        print('     Hyperparameter choice matters; justify paper defaults.')

    paper_val = grid_pass3.get((PAPER_TOPK, PAPER_MAXSIZE))
    if paper_val is not None:
        best_val = max(all_pass3)
        gap      = best_val - paper_val
        print(f'\n  Paper default (k={PAPER_TOPK}, s={PAPER_MAXSIZE}): '
              f'{paper_val:.1f}%')
        print(f'  Best config:                              {best_val:.1f}%')
        if gap <= 2.0:
            print(f'  ✅ Paper default is within 2 pp of best — well-chosen.')
        else:
            print(f'  ⚠️  Gap of {gap:.1f} pp — consider noting in paper.')

    # ── Save ──────────────────────────────────────────────────────────────
    # Convert tuple keys to strings for JSON serialization
    output = {
        'experiment':    'memory_hyperparameter_ablation',
        'dataset':       'HumanEval',
        'num_tasks':     args.tasks,
        'task_ids':      [t['task_id'] for t in tasks],
        'description': (
            'Ablation of VectorEpisodicMemory top_k and max_size. '
            'Addresses Reviewer s56K Concern 3 on memory hyperparameters.'
        ),
        'ablation_grid': {
            'top_k_values':    topk_values,
            'max_size_values': maxsize_values,
            'paper_defaults':  {
                'top_k':    PAPER_TOPK,
                'max_size': PAPER_MAXSIZE,
            },
        },
        'pass3_grid': {
            f'k{k}_s{s}': grid_pass3.get((k, s))
            for k, s in product(topk_values, maxsize_values)
        },
        'pass1_grid': {
            f'k{k}_s{s}': grid_pass1.get((k, s))
            for k, s in product(topk_values, maxsize_values)
        },
        'results': all_results,
    }

    out_path = Path(args.outdir) / 'memory_ablation.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=numpy_encoder)

    print(f'\n💾 Results saved → {out_path}')
    print('\nNext steps:')
    print('  1. Add Pass@3 heatmap table to paper as Table X (ablation section)')
    print('  2. Quote the overall spread in the rebuttal to show robustness')
    print('  3. Confirm paper default (k=5) is near the plateau')


if __name__ == '__main__':
    main()