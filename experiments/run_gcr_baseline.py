"""
run_gcr_baseline.py — Compute-Matched Single-Agent GCR Baseline

PURPOSE
-------
Addresses Reviewer s56K's critical concern:
  "MultiAgentReflexion uses three LLM calls per trial. Without a
   compute-matched baseline it is unclear whether the gain comes from
   role separation or simply from more inference."

This script implements Single-Agent Generate→Critique→Revise (GCR):
  - Exactly 3 LLM calls per trial  (same as MultiAgentReflexion)
  - One model does all three steps  (no role separation)
  - No shared memory pool           (no cross-agent learning)
  - Same temporal memory as ModularBaseline

The three-way comparison this produces:
  Agent               Calls/trial  Role sep?  Shared mem?
  ─────────────────── ────────────  ─────────  ───────────
  ModularBaseline          1          No          No
  SingleAgentGCR           3          No          No   ← new
  MultiAgentReflexion      3          Yes         Yes

If MultiAgentReflexion > SingleAgentGCR → gain is from role separation.
If MultiAgentReflexion ≈ SingleAgentGCR → gain is from call count alone.

USAGE
-----
  # Quick smoke test (20 tasks, ~30 min at 0.5s delay)
  python run_gcr_baseline.py --tasks 20

  # Full run matching paper (164 tasks, ~4-5 hrs at 0.5s delay)
  python run_gcr_baseline.py --tasks 164

  # Skip ModularBaseline if you already have those results
  python run_gcr_baseline.py --tasks 164 --skip-baseline

OUTPUT
------
  ../results/gcr_baseline.json   — full results for all three agents
  Console table                  — Pass@3, Pass@1, stats vs both baselines

PLACE THIS FILE AT
------------------
  experiments/run_gcr_baseline.py
  (same directory as run_comparison.py)
"""

import logging
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, '..')

from reflexion.config import SecureConfigLoader
from reflexion.llm import BaseLLMModel
from reflexion.benchmarks import HumanEvalLoader
from reflexion.agents import ReflexionAgent
from reflexion.evaluators import ObjectiveCodeEvaluator
from reflexion.memory import TemporalMemory

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# numpy JSON encoder (same as run_comparison.py)
# ─────────────────────────────────────────────────────────────────────────────

def numpy_encoder(obj):
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    raise TypeError(f'Object {type(obj)} not serializable')


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-AGENT GCR AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SingleAgentGCR:
    """
    Compute-matched Generate→Critique→Revise baseline.

    Uses exactly 3 LLM calls per trial — the same budget as
    MultiAgentReflexion — but with a single model performing all
    three roles without architectural role separation:

      Call 1 — Generate:  LLM writes candidate code from task prompt
      Call 2 — Critique:  Same LLM reviews its own code (no code-write
                           restriction — it can still rationalize errors)
      Call 3 — Revise:    Same LLM rewrites code using its own critique

    Key differences from MultiAgentReflexion:
      • No role constraints  (the critic can generate; the generator
        can review — no architectural separation of concerns)
      • No shared memory pool (reflections stay in a single temporal
        buffer, not tagged by role)
      • Single model perspective throughout (confirmation bias is not
        structurally prevented)

    Memory: TemporalMemory (same as ModularBaseline), reset per task.
    Max trials: 3 (same as all other agents).
    """

    AGENT_TYPE = 'SingleAgentGCR'

    def __init__(self, llm, max_trials: int = 3, memory_k: int = 3):
        self.llm        = llm
        self.max_trials = max_trials
        self.memory_k   = memory_k
        self.evaluator  = ObjectiveCodeEvaluator(timeout=10)
        self.memory     = TemporalMemory(max_size=10)

    # ── Prompt builders ───────────────────────────────────────────────────

    def _generate_prompt(self, task: dict, mem_ctx: str, trial: int) -> str:
        return (
            f"You are an expert Python programmer. "
            f"Complete this function (attempt {trial}/{self.max_trials}).\n\n"
            f"Task:\n{task['prompt']}\n\n"
            f"Past reflections (learn from these failures):\n{mem_ctx}\n\n"
            f"Output ONLY the complete Python function. "
            f"No markdown, no explanation:\n"
        )

    def _critique_prompt(self, task: dict, code: str) -> str:
        return (
            f"Review the following Python code for correctness.\n\n"
            f"Task:\n{task['prompt'][:400]}\n\n"
            f"Code to review:\n{code[:600]}\n\n"
            f"Identify ALL of the following if present:\n"
            f"  1. Logical errors (wrong algorithm or condition)\n"
            f"  2. Missing edge cases (empty input, None, negatives, "
            f"single element, zero)\n"
            f"  3. Type or boundary errors (off-by-one, wrong return type)\n\n"
            f"Be specific. Name the exact line or condition that is wrong. "
            f"Do NOT rewrite the code — only identify problems.\n"
        )

    def _revise_prompt(self, task: dict, code: str, critique: str) -> str:
        return (
            f"Rewrite the following Python function to fix ALL issues "
            f"identified in the critique.\n\n"
            f"Task:\n{task['prompt'][:400]}\n\n"
            f"Original code:\n{code[:500]}\n\n"
            f"Critique:\n{critique[:400]}\n\n"
            f"Output ONLY the corrected Python function. "
            f"No markdown, no explanation:\n"
        )

    # ── Code cleaner ──────────────────────────────────────────────────────

    def _clean(self, raw) -> str:
        """Strip markdown fences from LLM output."""
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

    # ── Core solve loop ───────────────────────────────────────────────────

    def solve_task(self, task: dict, verbose: bool = False) -> dict:
        """
        3-call-per-trial solve loop:
          Call 1: Generate code from task + memories
          Call 2: Self-critique the generated code
          Call 3: Revise code using own critique
          Evaluate revised code → pass/fail
          On failure: store reflection, increment trial
        """
        task_id = task['task_id']

        for trial in range(1, self.max_trials + 1):

            # Retrieve past reflections (temporal, same as baseline)
            memories = self.memory.get_relevant_memories(k=self.memory_k)
            mem_ctx  = '\n'.join(f'- {m}' for m in memories) \
                       if memories else 'None (first attempt)'

            try:
                # ── Call 1: Generate ──────────────────────────────────
                logger.info(f'  [GCR] Trial {trial}/{self.max_trials} '
                            f'— {task_id} — Call 1: Generate')
                raw_code = self.llm.call_llm(
                    self._generate_prompt(task, mem_ctx, trial),
                    max_tokens=2048,
                )
                code = self._clean(raw_code)

                # ── Call 2: Self-critique ─────────────────────────────
                logger.info(f'  [GCR] Trial {trial}/{self.max_trials} '
                            f'— {task_id} — Call 2: Critique')
                critique = self.llm.call_llm(
                    self._critique_prompt(task, code),
                    max_tokens=512,
                )
                if isinstance(critique, list):
                    critique = ''.join(
                        p.get('text', '') if isinstance(p, dict) else str(p)
                        for p in critique
                    )
                critique = str(critique).strip()

                # ── Call 3: Revise ────────────────────────────────────
                logger.info(f'  [GCR] Trial {trial}/{self.max_trials} '
                            f'— {task_id} — Call 3: Revise')
                raw_revised = self.llm.call_llm(
                    self._revise_prompt(task, code, critique),
                    max_tokens=2048,
                )
                revised = self._clean(raw_revised)

                # ── Evaluate ──────────────────────────────────────────
                result = self.evaluator.evaluate(
                    revised, task['entry_point'], task['test']
                )

                if result['passed']:
                    if verbose:
                        logger.info(f'  ✅ {task_id} solved in {trial} trial(s)')
                    return {
                        'task_id':    task_id,
                        'success':    True,
                        'trials':     trial,
                        'code':       revised,
                        'agent_type': self.AGENT_TYPE,
                    }

                # ── Reflection on failure ─────────────────────────────
                # Store a brief reflection (same format as ModularBaseline)
                error_snippet = (result.get('error') or 'tests failed')[:150]
                reflection = (
                    f"Trial {trial} failed: {error_snippet}. "
                    f"Critique noted: {critique[:100]}"
                )
                self.memory.add_reflection(reflection)

                if verbose:
                    logger.info(f'  ❌ {task_id} trial {trial} failed: '
                                f'{error_snippet[:60]}')

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.memory.add_reflection(
                    f"Trial {trial} exception: {str(exc)[:100]}"
                )
                logger.error(f'  [GCR] Exception on {task_id} trial {trial}: {exc}')

        return {
            'task_id':    task_id,
            'success':    False,
            'trials':     self.max_trials,
            'agent_type': self.AGENT_TYPE,
        }

    def reset(self):
        """Clear memory between tasks."""
        self.memory.clear()


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(agent, tasks, name, llm=None):
    """Run agent on all tasks, reset between each, return results list.

    Instrumentation (api_calls, latency_s) is recorded when the llm object
    has a track_task() context manager (instrumented BaseLLMModel).
    Falls back silently when the older un-instrumented version is installed.
    """
    logger.info(f'\n{"="*70}\n Running: {name}\n{"="*70}')

    # Detect whether this llm supports instrumentation
    use_tracking = llm is not None and hasattr(llm, 'track_task')

    results = []

    for i, task in enumerate(tasks):
        logger.info(f'\nTask {i+1}/{len(tasks)}: {task["task_id"]}')

        if use_tracking:
            with llm.track_task() as task_stats:
                result = agent.solve_task(task, verbose=True)
            result['api_calls'] = task_stats['api_calls']
            result['latency_s'] = task_stats['latency_s']
        else:
            result = agent.solve_task(task, verbose=True)

        result['agent_type'] = name
        if 'trials' not in result:
            result['trials'] = result.get('used_trials', 0)

        results.append(result)

        # Always reset between tasks
        if hasattr(agent, 'reset'):
            agent.reset()
        elif hasattr(agent, 'memory'):
            agent.memory.clear()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ci(n_success, n_total, z=1.96):
    if n_total == 0:
        return 0.0, 0.0
    p = n_success / n_total
    d = 1 + z**2 / n_total
    c = (p + z**2 / (2 * n_total)) / d
    m = z * (p * (1 - p) / n_total + z**2 / (4 * n_total**2)) ** 0.5 / d
    return max(0.0, c - m) * 100, min(1.0, c + m) * 100


def compute_metrics(results):
    n      = len(results)
    passed = sum(1 for r in results if r.get('success'))
    p1     = sum(1 for r in results
                 if r.get('success') and r.get('trials', 0) == 1)
    failed_t1 = [r for r in results
                 if not (r.get('success') and r.get('trials', 0) == 1)]
    recovered = sum(1 for r in results
                    if r.get('success') and r.get('trials', 0) > 1)
    lo, hi = wilson_ci(passed, n)
    calls  = [r['api_calls'] for r in results if 'api_calls' in r]
    return {
        'n':           n,
        'pass3':       passed / n * 100 if n else 0.0,
        'pass3_raw':   f'{passed}/{n}',
        'pass1':       p1 / n * 100 if n else 0.0,
        'recovery':    recovered / len(failed_t1) * 100 if failed_t1 else 100.0,
        'ci_lo':       lo,
        'ci_hi':       hi,
        'avg_calls':   float(np.mean(calls)) if calls else None,
    }


def paired_stats(results_a, results_b):
    """Paired t-test and Cohen's d comparing b vs a."""
    a_bin = [1 if r.get('success') else 0 for r in results_a]
    b_bin = [1 if r.get('success') else 0 for r in results_b]
    n = min(len(a_bin), len(b_bin))
    a = np.array(a_bin[:n], dtype=float)
    b = np.array(b_bin[:n], dtype=float)
    t_stat, p_val = scipy_stats.ttest_rel(b, a)
    diff = b - a
    d = float(diff.mean() / (diff.std() + 1e-9))
    delta = float((b.mean() - a.mean()) * 100)
    return {
        'delta_pass3':   round(delta, 2),
        't_statistic':   round(float(t_stat), 4),
        'p_value':       round(float(p_val), 4),
        'cohens_d':      round(d, 4),
        'significant':   bool(p_val < 0.05),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(baseline_r, gcr_r, multi_r):
    """
    Print the three-way comparison table the reviewer asked for.

    Columns: Agent | Calls/trial | Pass@3 | Pass@1 | Rec. | 95% CI
    Statistical rows: GCR vs Baseline, Multi vs Baseline, Multi vs GCR
    """
    bm = compute_metrics(baseline_r)
    gm = compute_metrics(gcr_r)
    mm = compute_metrics(multi_r)

    print('\n' + '=' * 78)
    print('COMPUTE-MATCHED COMPARISON — REVIEWER s56K TABLE')
    print('=' * 78)
    print(f'{"Agent":<26} {"Calls":>6} {"Pass@3":>8} {"Pass@1":>8} '
          f'{"Rec%":>7} {"95% CI":>20}')
    print('-' * 78)

    rows = [
        ('ModularBaseline',      1, bm),
        ('SingleAgentGCR',       3, gm),
        ('MultiAgentReflexion',  3, mm),
    ]
    for name, calls, m in rows:
        ci = f'[{m["ci_lo"]:.1f}%, {m["ci_hi"]:.1f}%]'
        print(f'{name:<26} {calls:>6} {m["pass3"]:>7.1f}% '
              f'{m["pass1"]:>7.1f}% {m["recovery"]:>6.1f}% {ci:>20}')

    print('\n── Statistical Validation ──')
    print(f'{"Comparison":<38} {"ΔPass@3":>9} {"t":>7} {"p":>8} '
          f'{"d":>7} {"Sig?":>5}')
    print('-' * 78)

    comparisons = [
        ('GCR vs ModularBaseline',   baseline_r, gcr_r),
        ('MultiAgent vs Baseline',   baseline_r, multi_r),
        ('MultiAgent vs GCR',        gcr_r,      multi_r),
    ]
    for label, ref, ext in comparisons:
        s = paired_stats(ref, ext)
        sig = '✅' if s['significant'] else '❌'
        print(f'{label:<38} {s["delta_pass3"]:>+8.1f}pp '
              f'{s["t_statistic"]:>7.3f} {s["p_value"]:>8.4f} '
              f'{s["cohens_d"]:>7.3f} {sig:>5}')

    print('\n── Interpretation ──')
    delta_multi_gcr = compute_metrics(multi_r)['pass3'] - compute_metrics(gcr_r)['pass3']
    s_mg = paired_stats(gcr_r, multi_r)
    if s_mg['significant'] and delta_multi_gcr > 0:
        print(f'✅ MultiAgentReflexion outperforms compute-matched GCR by '
              f'{delta_multi_gcr:+.1f} pp (p={s_mg["p_value"]:.4f}).')
        print('   Gain is attributable to ROLE SEPARATION, not call count.')
    elif delta_multi_gcr > 0:
        print(f'⚠️  MultiAgent leads GCR by {delta_multi_gcr:+.1f} pp but '
              f'p={s_mg["p_value"]:.4f} — not significant at n={len(gcr_r)}.')
        print('   Run full 164-task set for definitive result.')
    else:
        print(f'❌ MultiAgent does NOT outperform GCR ({delta_multi_gcr:+.1f} pp).')
        print('   Gain over baseline may be attributable to call count, '
              'not role separation.')
        print('   Revise paper claims accordingly.')

    if gm['avg_calls'] and mm['avg_calls']:
        print(f'\n── Cost Summary ──')
        print(f'   GCR avg calls/task:        {gm["avg_calls"]:.2f}')
        print(f'   MultiAgent avg calls/task:  {mm["avg_calls"]:.2f}')

    print('=' * 78)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute-matched GCR baseline vs MultiAgentReflexion'
    )
    parser.add_argument(
        '--tasks', type=int, default=20,
        help='Number of HumanEval tasks (default 20 for smoke test; '
             'use 164 for full paper run)',
    )
    parser.add_argument(
        '--outdir', default='../results',
        help='Directory to write gcr_baseline.json',
    )
    parser.add_argument(
        '--skip-baseline', action='store_true',
        help='Skip ModularBaseline run (use if you already have results)',
    )
    parser.add_argument(
        '--skip-multi', action='store_true',
        help='Skip MultiAgentReflexion run (use if you already have results)',
    )
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────
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

    # ── ETA estimate ──────────────────────────────────────────────────────
    delay      = config['rate_limit_delay']
    n          = len(tasks)
    agents_run = (0 if args.skip_baseline else 1) + 1 + (0 if args.skip_multi else 1)
    # GCR uses 3 calls per trial, Baseline uses 1, Multi uses 3 (approx)
    calls_map  = {'baseline': 1, 'gcr': 3, 'multi': 3}
    eta_min    = n * 3 * delay * (
        (0 if args.skip_baseline else calls_map['baseline'])
        + calls_map['gcr']
        + (0 if args.skip_multi else calls_map['multi'])
    ) / 60

    print('\n' + '=' * 70)
    print('COMPUTE-MATCHED GCR BASELINE EXPERIMENT')
    print('=' * 70)
    print(f'Tasks:        {n}')
    print(f'Delay:        {delay}s between calls')
    print(f'ETA:          ~{eta_min:.0f} minutes')
    print(f'Skip baseline: {args.skip_baseline}')
    print(f'Skip multi:    {args.skip_multi}')
    print('=' * 70)
    input('\nPress ENTER to start...')

    all_results = {}

    # ── 1. ModularBaseline (1 call/trial) ─────────────────────────────────
    if not args.skip_baseline:
        logger.info('\n🔵 Running ModularBaseline (1 call/trial)...')
        baseline_agent   = ReflexionAgent(llm, memory_mode='temporal', max_trials=3)
        baseline_results = run_agent(baseline_agent, tasks, 'Modular_Baseline', llm)
        all_results['modular_baseline'] = baseline_results
    else:
        logger.info('\n⏭️  Skipping ModularBaseline')
        baseline_results = []

    # ── 2. SingleAgentGCR (3 calls/trial, no role sep) ───────────────────
    logger.info('\n🟡 Running SingleAgentGCR (3 calls/trial, no role sep)...')
    gcr_agent   = SingleAgentGCR(llm, max_trials=3)
    gcr_results = run_agent(gcr_agent, tasks, 'SingleAgentGCR', llm)
    all_results['single_agent_gcr'] = gcr_results

    # ── 3. MultiAgentReflexion (3 calls/trial, role sep) ─────────────────
    if not args.skip_multi:
        logger.info('\n🔴 Running MultiAgentReflexion (3 calls/trial, role sep)...')
        # Import here so the script still runs if only GCR is needed
        from reflexion.agents import MultiAgentReflexion
        multi_agent   = MultiAgentReflexion(llm, max_trials=3)
        multi_results = run_agent(multi_agent, tasks, 'MultiAgentReflexion', llm)
        all_results['multiagent_reflexion'] = multi_results
    else:
        logger.info('\n⏭️  Skipping MultiAgentReflexion')
        multi_results = []

    # ── Report ────────────────────────────────────────────────────────────
    if baseline_results and gcr_results and multi_results:
        print_report(baseline_results, gcr_results, multi_results)
    else:
        # Partial report — just print what we have
        print('\n── Partial Results ──')
        for name, results in all_results.items():
            if not results:
                continue
            m = compute_metrics(results)
            print(f'{name}: Pass@3={m["pass3"]:.1f}%  Pass@1={m["pass1"]:.1f}%  '
                  f'n={m["n"]}')

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        'experiment':    'compute_matched_gcr_baseline',
        'dataset':       'HumanEval',
        'num_tasks':     len(tasks),
        'task_ids':      [t['task_id'] for t in tasks],
        'description': (
            'Three-way comparison: ModularBaseline (1 call/trial), '
            'SingleAgentGCR (3 calls/trial, no role sep), '
            'MultiAgentReflexion (3 calls/trial, role sep). '
            'Addresses Reviewer s56K compute-matched baseline concern.'
        ),
        'agent_design': {
            'ModularBaseline': {
                'calls_per_trial': 1,
                'role_separation': False,
                'shared_memory':   False,
            },
            'SingleAgentGCR': {
                'calls_per_trial':  3,
                'role_separation':  False,
                'shared_memory':    False,
                'steps':           ['generate', 'self_critique', 'revise'],
                'note': (
                    'Same model for all three steps. No restriction on '
                    'what the critic can do — it can generate code too. '
                    'Self-Refine-style (Madaan et al. 2023).'
                ),
            },
            'MultiAgentReflexion': {
                'calls_per_trial': 3,
                'role_separation': True,
                'shared_memory':   True,
                'roles':          ['Generator', 'Critic', 'Verifier'],
                'note': (
                    'Critic explicitly prohibited from generating code. '
                    'Shared vector memory pool across all three agents.'
                ),
            },
        },
        'metrics': {
            name: compute_metrics(res)
            for name, res in all_results.items() if res
        },
        'statistics': {
            'gcr_vs_baseline':   paired_stats(baseline_results, gcr_results)
                                 if baseline_results and gcr_results else None,
            'multi_vs_baseline': paired_stats(baseline_results, multi_results)
                                 if baseline_results and multi_results else None,
            'multi_vs_gcr':      paired_stats(gcr_results, multi_results)
                                 if gcr_results and multi_results else None,
        },
        'results': all_results,
    }

    out_path = Path(args.outdir) / 'gcr_baseline.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=numpy_encoder)

    print(f'\n💾 Results saved → {out_path}')
    print('\nNext steps:')
    print('  1. If Multi > GCR (significant): role separation is the cause → '
          'rebuttal claim holds')
    print('  2. If Multi ≈ GCR: revise claims, gain may be call-count only')
    print('  3. Add gcr_baseline.json results to paper Table 4 and rebuttal')


if __name__ == '__main__':
    main()

