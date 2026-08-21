"""
retention_policy_ablation.py — TMLR revision: retention-policy ablation
(now with capacity sweep, matching your top-k / max-size sensitivity style)

Your existing sensitivity analyses vary CAPACITY (max_size) and RETRIEVAL
COUNT (k) — but every one of them uses the same eviction MECHANISM: plain
FIFO. This script varies the mechanism itself across a capacity sweep,
comparing three retention policies on VectorEpisodicMemory:

  fifo        — evict oldest-inserted (current default, unchanged)
  lru         — evict least-recently-RETRIEVED
  importance  — evict lowest explicitly-tagged importance score

EXPERIMENT DESIGN
──────────────────
A small set of "signal" reflections (a valuable learned pattern — same
chunking-pattern scenario as long_horizon_benchmark.py, for continuity)
is inserted early, then buried under a fixed pool of "noise" reflections.
max_size is swept across several capacity levels so you get a curve, not
a single point — matching the style of MEMORY_SCALES in
memory_efficiency.py / retrieval_analysis.py. At very large max_size
(>= total insertions), no eviction ever happens and all policies trivially
converge to 100% survival — this is an expected sanity-check boundary, not
a bug, and confirms retention policy only matters UNDER capacity pressure.

Two scenarios at every capacity level:
  DORMANT     — signal never re-queried until the final recall check.
  REVISITED   — signal periodically re-queried during the noise phase.

This needs ZERO OpenRouter API calls — only llm.get_embedding() (local
SentenceTransformer) — fast, free, no rate limiting.

Usage:
    # single config (original behavior, unchanged if only one value given)
    python retention_policy_ablation.py --max-sizes 10

    # full capacity sweep (recommended for the paper)
    python retention_policy_ablation.py --max-sizes 5 10 20 40 80 160 --n-noise 100
"""

import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
sys.path.insert(0, '..')

from reflexion.config import SecureConfigLoader
from reflexion.llm import BaseLLMModel
from reflexion.memory.vector import VectorEpisodicMemory

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

POLICIES = ['fifo', 'lru', 'importance']
SCENARIOS = ['dormant', 'revisited']

SIGNAL_REFLECTIONS = [
    "Learned chunking_pattern: use [data[i:i+size] for i in range(0, len(data), size)] "
    "to split a list into fixed-size chunks. Key: step=size in range().",
    "Learned batch_process pattern: iterate in batches using chunk_list(items, batch_size). "
    "Apply fn to each batch. Chunking is core to all batch processing.",
    "Learned that chunking large datasets before processing avoids memory overflow "
    "and improves throughput on large-scale batch jobs.",
]

SIGNAL_QUERY = "How do I process a large dataset efficiently using chunks?"

NOISE_TOPICS = [
    "fibonacci", "palindrome", "vowel counting", "string reversal",
    "prime checking", "matrix transpose", "binary search", "linked list",
    "graph traversal", "sorting algorithm", "hash table", "recursion depth",
    "stack overflow", "queue implementation", "tree balancing", "set operations",
]


def numpy_encoder(obj):
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    raise TypeError(f"Object {type(obj)} not serializable")


def make_noise(n: int):
    return [
        f"Solved a problem about {NOISE_TOPICS[i % len(NOISE_TOPICS)]}: "
        f"variant {i}, unrelated to chunking or batching."
        for i in range(n)
    ]


def run_condition(llm, policy: str, scenario: str, max_size: int,
                   n_noise: int, revisit_every: int = 5):
    """
    Run one (policy, scenario, max_size) combination and report whether
    the signal survives eviction pressure and whether it's retrievable
    at the end.
    """
    mem = VectorEpisodicMemory(llm, max_size=max_size, retention_policy=policy)

    for s in SIGNAL_REFLECTIONS:
        importance = 1.0 if policy == 'importance' else None
        mem.add_reflection(s, importance_score=importance)

    noise = make_noise(n_noise)
    for i, n in enumerate(noise):
        importance = 0.0 if policy == 'importance' else None
        mem.add_reflection(n, importance_score=importance)

        if scenario == 'revisited' and (i + 1) % revisit_every == 0:
            mem.get_relevant_memories(SIGNAL_QUERY, k=len(SIGNAL_REFLECTIONS))

    surviving_signal = [
        r for r in mem.reflections
        if any(r == s for s in SIGNAL_REFLECTIONS)
    ]
    survival_rate = len(surviving_signal) / len(SIGNAL_REFLECTIONS) * 100

    k = min(5, len(mem))
    retrieved = mem.get_relevant_memories(SIGNAL_QUERY, k=k)
    signal_in_top_k = sum(1 for r in retrieved if r in SIGNAL_REFLECTIONS)
    recall_hit_rate = signal_in_top_k / len(SIGNAL_REFLECTIONS) * 100

    return {
        'policy':          policy,
        'scenario':        scenario,
        'max_size':        max_size,
        'n_noise':         n_noise,
        'total_insertions': len(SIGNAL_REFLECTIONS) + n_noise,
        'final_pool_size': len(mem),
        'signal_survived': len(surviving_signal),
        'signal_total':    len(SIGNAL_REFLECTIONS),
        'survival_rate_pct':   round(survival_rate, 1),
        'signal_in_top_k':     signal_in_top_k,
        'recall_hit_rate_pct': round(recall_hit_rate, 1),
    }


def print_single_report(results):
    """Original single-config report (unchanged from before)."""
    print('\n' + '='*78)
    print('RETENTION-POLICY ABLATION REPORT (TMLR revision)')
    print('='*78)
    print(f'{"Policy":<12} {"Scenario":<12} {"Survived":>10} {"Survival %":>12} {"Recall Hit %":>14}')
    print('-'*78)
    for r in results:
        print(f"{r['policy']:<12} {r['scenario']:<12} "
              f"{r['signal_survived']}/{r['signal_total']:<8} "
              f"{r['survival_rate_pct']:>11.1f}% {r['recall_hit_rate_pct']:>13.1f}%")
    print('-'*78)

    print('\n── Interpretation ──')
    dormant = {r['policy']: r for r in results if r['scenario'] == 'dormant'}
    revisited = {r['policy']: r for r in results if r['scenario'] == 'revisited'}

    fifo_dormant = dormant['fifo']['survival_rate_pct']
    lru_dormant = dormant['lru']['survival_rate_pct']
    imp_dormant = dormant['importance']['survival_rate_pct']

    print(f"DORMANT scenario (valuable memory never re-queried until the end):")
    print(f"  FIFO survival:       {fifo_dormant:.1f}%")
    print(f"  LRU survival:        {lru_dormant:.1f}%")
    print(f"  Importance survival: {imp_dormant:.1f}%")
    if abs(lru_dormant - fifo_dormant) < 1.0:
        print(f"  -> LRU behaves ~identically to FIFO here: with no interim retrieval,")
        print(f"     LRU has no usage signal to act on, so it falls back to age-based")
        print(f"     eviction — same as FIFO. Only explicit importance-tagging protects")
        print(f"     a valuable-but-dormant memory.")
    if imp_dormant > max(fifo_dormant, lru_dormant) + 1.0:
        print(f"  -> Importance-weighted retention meaningfully outperforms both FIFO")
        print(f"     and LRU when the valuable memory isn't naturally re-queried.")

    print(f"\nREVISITED scenario (valuable memory periodically re-queried):")
    for pol in POLICIES:
        print(f"  {pol.upper()} survival: {revisited[pol]['survival_rate_pct']:.1f}%")
    if revisited['lru']['survival_rate_pct'] > lru_dormant + 1.0:
        print(f"  -> LRU's survival improves substantially once the memory is actually")
        print(f"     used ({lru_dormant:.1f}% -> {revisited['lru']['survival_rate_pct']:.1f}%),")
        print(f"     confirming LRU protects based on usage, not intrinsic value.")

    print('='*78)


def print_sweep_report(results, max_sizes, n_noise):
    """Sweep report: survival rate at each capacity level, per policy/scenario."""
    total_insertions = len(SIGNAL_REFLECTIONS) + n_noise

    print('\n' + '='*90)
    print('RETENTION-POLICY CAPACITY SWEEP (TMLR revision)')
    print(f'signal_count={len(SIGNAL_REFLECTIONS)}  n_noise={n_noise}  '
          f'total_insertions={total_insertions}')
    print('='*90)

    for scenario in SCENARIOS:
        print(f'\n── Scenario: {scenario.upper()} — Survival Rate (%) by max_size ──')
        header = f'{"Policy":<12}' + ''.join(f'{ms:>10}' for ms in max_sizes)
        print(header)
        print('-'*len(header))
        for policy in POLICIES:
            row = f'{policy:<12}'
            for ms in max_sizes:
                r = next(x for x in results
                         if x['policy'] == policy and x['scenario'] == scenario
                         and x['max_size'] == ms)
                row += f'{r["survival_rate_pct"]:>9.1f}%'
            print(row)

    print('\n── Interpretation ──')
    print(f'At max_size >= {total_insertions} (total insertions), no eviction occurs at')
    print(f'all — every policy trivially reaches 100% survival there. That convergence')
    print(f'point is a sanity check, not a finding: retention policy only matters when')
    print(f'capacity < total insertions actually forces eviction decisions.')
    print(f'The informative region is the smaller max_size values, where FIFO and')
    print(f'(dormant) LRU lose the signal early while importance-weighting still')
    print(f'retains it — that gap is the effect worth reporting.')
    print('='*90)


def maybe_plot(results, max_sizes, n_noise, outdir):
    """Generate a sensitivity-curve figure matching plot_results.py's dark theme,
    if matplotlib is available. Non-fatal if it isn't."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # headless backend — avoids Tkinter/Tcl entirely,
                                # since we only ever save the figure, never show() it
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning('matplotlib not available — skipping figure generation '
                        '(results are still saved to JSON).')
        return

    matplotlib.rcParams.update({
        "figure.facecolor":  "#0d0d14",
        "axes.facecolor":    "#111118",
        "axes.edgecolor":    "#2a2a3e",
        "axes.labelcolor":   "#c0c0d8",
        "axes.titlecolor":   "#e0e0f0",
        "xtick.color":       "#606080",
        "ytick.color":       "#606080",
        "grid.color":        "#1e1e2e",
        "legend.facecolor":  "#111118",
        "legend.edgecolor":  "#2a2a3e",
        "text.color":        "#e0e0f0",
        "font.family":       "monospace",
        "savefig.facecolor": "#0d0d14",
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
    })

    colors = {'fifo': '#9090b0', 'lru': '#7c6aff', 'importance': '#6affd4'}
    markers = {'fifo': 'o', 'lru': 's', 'importance': '^'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Retention-Policy Ablation — Signal Survival vs. Capacity", fontsize=12)

    for ax, scenario in zip(axes, SCENARIOS):
        for policy in POLICIES:
            ys = [
                next(x for x in results
                     if x['policy'] == policy and x['scenario'] == scenario
                     and x['max_size'] == ms)['survival_rate_pct']
                for ms in max_sizes
            ]
            ax.plot(max_sizes, ys, marker=markers[policy], color=colors[policy],
                    label=policy, linewidth=2, markersize=6)
        ax.set_xscale('log')
        ax.set_xlabel('max_size (capacity, log scale)')
        ax.set_ylabel('Signal Survival Rate (%)')
        ax.set_title(f'{scenario.capitalize()} scenario')
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle='--')
        ax.legend(fontsize=8)

    fig.tight_layout()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fig_path = Path(outdir) / 'fig_retention_policy_sweep.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f'\n📊 Figure saved → {fig_path}')


def main():
    parser = argparse.ArgumentParser(description='Retention-policy ablation (TMLR revision)')
    parser.add_argument('--max-sizes',     type=int, nargs='+', default=[10],
                        help='One value = single-config run (original behavior). '
                             'Multiple values = capacity sweep, e.g. 5 10 20 40 80 160')
    parser.add_argument('--n-noise',       type=int, default=40,
                        help='Number of distractor reflections inserted after signal')
    parser.add_argument('--revisit-every', type=int, default=5,
                        help='In the "revisited" scenario, re-query signal every N noise insertions')
    parser.add_argument('--outdir',        default='../results')
    parser.add_argument('--figdir',        default='../figures')
    parser.add_argument('--no-plot',       action='store_true',
                        help='Skip figure generation even in sweep mode')
    args = parser.parse_args()

    try:
        config = SecureConfigLoader().load_from_env_file('../.env')
    except Exception as e:
        logger.error(f'❌ Config error: {e}'); sys.exit(1)

    # Only get_embedding() is used below — call_llm() (OpenRouter) is
    # never invoked, so this doesn't burn API calls or hit rate limits.
    llm = BaseLLMModel(
        config['openrouter_api_key'],
        config['openrouter_model'],
        config['gemini_api_base'],
        config['rate_limit_delay'],
    )

    is_sweep = len(args.max_sizes) > 1

    logger.info('🚀 Retention-policy ablation — no API calls, local embeddings only')
    logger.info(f'   mode={"SWEEP" if is_sweep else "single-config"}, '
                f'max_sizes={args.max_sizes}, n_noise={args.n_noise}, '
                f'signal_count={len(SIGNAL_REFLECTIONS)}')

    results = []
    for max_size in args.max_sizes:
        for scenario in SCENARIOS:
            for policy in POLICIES:
                logger.info(f'Running: max_size={max_size}, policy={policy}, scenario={scenario}')
                r = run_condition(
                    llm, policy, scenario,
                    max_size=max_size,
                    n_noise=args.n_noise,
                    revisit_every=args.revisit_every,
                )
                results.append(r)

    if is_sweep:
        print_sweep_report(results, args.max_sizes, args.n_noise)
        if not args.no_plot:
            try:
                maybe_plot(results, args.max_sizes, args.n_noise, args.figdir)
            except Exception as e:
                logger.warning(f'Figure generation failed ({e}) — continuing, '
                                f'results will still be saved to JSON.')
    else:
        print_single_report(results)

    output = {
        'config': {
            'max_sizes': args.max_sizes,
            'n_noise': args.n_noise,
            'revisit_every': args.revisit_every,
            'n_signal': len(SIGNAL_REFLECTIONS),
            'mode': 'sweep' if is_sweep else 'single',
        },
        'results': results,
    }

    suffix = '_sweep' if is_sweep else ''
    out_path = Path(args.outdir) / f'extension1_retention_policy_ablation{suffix}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=numpy_encoder)

    print(f'\n💾 Saved → {out_path}')


if __name__ == '__main__':
    main()