"""
role_conditioned_memory_ablation.py — TMLR revision: role-conditioned MEMORY ablation

This is the direct answer to the reviewer's remaining point on
"role-conditioned vs. non-role-conditioned memory" — distinct from the
prompt-level role_conditioning_ablation.py you already have.

Compares:
  MultiAgentReflexion()                        — non-role-conditioned memory
      (plain cosine-similarity retrieval; only agent-id self-exclusion;
       this is your existing published method, unchanged)
  MultiAgentReflexionRoleConditionedMemory()   — role-conditioned memory
      (identical architecture + identical prompt-level role text; ONLY
       difference is retrieval now prefers same-role-tagged reflections
       over cross-role ones)

Both are run on the SAME tasks in the SAME session, so this isolates the
marginal contribution of role-conditioned memory RETRIEVAL specifically —
prompt-level role-conditioning is held constant at True in both arms.

Usage:
    python role_conditioned_memory_ablation.py
    python role_conditioned_memory_ablation.py --tasks 50   # quick smoke test
    python role_conditioned_memory_ablation.py --tasks 164  # full run for paper
"""

import logging
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy import stats
sys.path.insert(0, '..')

from reflexion.config import SecureConfigLoader
from reflexion.llm import BaseLLMModel
from reflexion.benchmarks import HumanEvalLoader
from reflexion.agents import MultiAgentReflexion, MultiAgentReflexionRoleConditionedMemory

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def numpy_encoder(obj):
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    raise TypeError(f"Object {type(obj)} not serializable")


def wilson_ci(n_s, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = n_s / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / d
    return max(0, c-m)*100, min(1, c+m)*100


def run_agent(agent, tasks, name, llm):
    """Runs one agent across all tasks, resetting between tasks for
    task-isolated memory (matches your existing run_agent convention).

    IMPORTANT: MultiAgentReflexion.reset() clears agent.task_results and
    agent.communication_log (by design — task isolation). If we reset
    after every task, the agent's own get_communication_analysis() would
    see nothing by the time we call it at the end. So we snapshot both
    lists into our own running totals BEFORE each reset, and compute the
    communication analysis from that accumulated copy afterward — see
    compute_communication_analysis() below.
    """
    logger.info(f'\n{"="*60}\n Running: {name}\n{"="*60}')
    results = []
    all_task_results: List[Dict] = []
    all_comm_log: List[Dict] = []
    for i, task in enumerate(tasks):
        logger.info(f'Task {i+1}/{len(tasks)}: {task["task_id"]}')
        with llm.track_task() as task_stats:
            result = agent.solve_task(task, verbose=False)
        result['api_calls'] = task_stats['api_calls']
        result['latency_s'] = task_stats['latency_s']
        if 'trials' not in result:
            result['trials'] = result.get('used_trials', 0)
        results.append(result)

        # Snapshot BEFORE reset wipes agent.task_results / communication_log
        all_task_results.extend(agent.task_results)
        all_comm_log.extend(agent.communication_log)

        agent.reset()
    return results, all_task_results, all_comm_log


def compute_communication_analysis(task_results: List[Dict], comm_log: List[Dict],
                                    use_role_conditioning: bool,
                                    role_conditioned_memory: bool) -> Dict:
    """Reimplements MultiAgentReflexion.get_communication_analysis(), but
    operating on accumulated lists collected across resets instead of the
    agent's (post-reset) live state. Same fields, same logic."""
    if not task_results:
        return {"message": "No tasks run yet"}

    total_tasks = len(task_results)
    collab_tasks = sum(1 for r in task_results if r["agents_solved"] > 1)
    supervisor_wins = sum(
        1 for r in task_results if r.get("winning_agent") == "Supervisor"
    )

    agent_stats: Dict[str, Dict] = {}
    for entry in comm_log:
        aid = entry["agent_id"]
        if aid not in agent_stats:
            agent_stats[aid] = {"attempts": 0, "successes": 0, "memories_received": 0}
        agent_stats[aid]["attempts"] += 1
        agent_stats[aid]["successes"] += int(entry["success"])
        agent_stats[aid]["memories_received"] += entry["memories_received"]

    return {
        "total_tasks": total_tasks,
        "collaboration_rate_pct": round(collab_tasks / total_tasks * 100, 1),
        "supervisor_interventions": supervisor_wins,
        "avg_shared_reflections_per_task": round(
            float(np.mean([r["shared_reflections_used"] for r in task_results])), 2
        ),
        "per_agent_stats": agent_stats,
        "protocol": "Round-robin → Debate → Weighted vote",
        "use_role_conditioning": use_role_conditioning,
        "role_conditioned_memory": role_conditioned_memory,
    }


def metrics(results):
    n      = len(results)
    passed = sum(1 for r in results if r['success'])
    p1     = sum(1 for r in results if r['success'] and r.get('trials', 0) == 1)
    failed = [r for r in results if not (r['success'] and r.get('trials', 0) == 1)]
    recov  = sum(1 for r in results if r['success'] and r.get('trials', 0) > 1)
    lo, hi = wilson_ci(passed, n)
    return {
        'n':          n,
        'pass3':      passed / n * 100,
        'pass3_raw':  f'{passed}/{n}',
        'pass1':      p1 / n * 100,
        'recovery':   recov / len(failed) * 100 if failed else 100.0,
        'avg_trials': np.mean([r['trials'] for r in results
                               if r['success'] and r.get('trials', 0) > 0]) or 0,
        'avg_calls':  np.mean([r['api_calls'] for r in results if 'api_calls' in r]),
        'ci_lo':      lo,
        'ci_hi':      hi,
    }


def avg_memories_received(analysis):
    """Average memories_received per agent from get_communication_analysis()."""
    stats_by_agent = analysis.get('per_agent_stats', {})
    if not stats_by_agent:
        return 0.0
    totals = [s['memories_received'] / max(s['attempts'], 1) for s in stats_by_agent.values()]
    return float(np.mean(totals)) if totals else 0.0


def report(plain_r, rc_r, plain_analysis, rc_analysis):
    """Print the role-conditioned-memory ablation report."""
    pm = metrics(plain_r)
    rm = metrics(rc_r)

    agree     = sum(1 for p, r in zip(plain_r, rc_r) if p['success'] == r['success'])
    agree_pct = agree / len(plain_r) * 100

    p_bin = [1 if r['success'] else 0 for r in plain_r]
    r_bin = [1 if r['success'] else 0 for r in rc_r]
    t_stat, p_val = stats.ttest_rel(r_bin, p_bin)

    diff     = np.array(r_bin) - np.array(p_bin)
    cohens_d = diff.mean() / (diff.std() + 1e-9)

    delta_p3 = rm['pass3'] - pm['pass3']

    print('\n' + '='*76)
    print('ROLE-CONDITIONED MEMORY ABLATION REPORT (TMLR revision)')
    print('Prompt-level role-conditioning held constant at True in BOTH arms.')
    print('='*76)
    print(f'\n{"Metric":<30} {"Non-Role-Cond. Mem":>20} {"Role-Cond. Mem":>16} {"Δ":>8}')
    print('-'*76)
    print(f'{"Pass@3":<30} {pm["pass3"]:>19.1f}% {rm["pass3"]:>15.1f}% {delta_p3:>+7.1f}pp')
    print(f'{"Pass@1":<30} {pm["pass1"]:>19.1f}% {rm["pass1"]:>15.1f}% {rm["pass1"]-pm["pass1"]:>+7.1f}pp')
    print(f'{"Recovery Rate":<30} {pm["recovery"]:>19.1f}% {rm["recovery"]:>15.1f}%')
    print(f'{"Avg Trials":<30} {pm["avg_trials"]:>20.2f} {rm["avg_trials"]:>16.2f}')
    print(f'{"Avg API Calls/Task":<30} {pm["avg_calls"]:>20.2f} {rm["avg_calls"]:>16.2f}')
    print(f'{"Avg Memories Received/Agent":<30} {avg_memories_received(plain_analysis):>20.2f} '
          f'{avg_memories_received(rc_analysis):>16.2f}')
    print(f'{"Collaboration Rate":<30} {plain_analysis["collaboration_rate_pct"]:>19.1f}% '
          f'{rc_analysis["collaboration_rate_pct"]:>15.1f}%')
    print(f'{"95% CI (Wilson)":<30} [{pm["ci_lo"]:.1f}-{pm["ci_hi"]:.1f}%] '
          f'[{rm["ci_lo"]:.1f}-{rm["ci_hi"]:.1f}%]')
    print('-'*76)
    print(f'\nPer-task agreement:  {agree}/{len(plain_r)} tasks ({agree_pct:.1f}%)')
    print(f'Paired t-test:       t={t_stat:.3f}, p={p_val:.4f}')
    print(f"Cohen's d:           {cohens_d:.3f}")
    print(f'Pass@3 delta:        {delta_p3:+.1f}pp (role-cond. memory − non-role-cond. memory)')

    print('\n── Interpretation ──')
    if p_val < 0.05:
        direction = 'HELPS' if delta_p3 > 0 else 'HURTS'
        print(f'✅ Statistically significant difference (p = {p_val:.4f}) — '
              f'role-conditioned memory retrieval {direction} performance')
        print(f'   Δ Pass@3 = {delta_p3:+.1f}pp, Cohen\'s d = {cohens_d:.3f}')
    else:
        print(f'⚠️  No statistically significant difference (p = {p_val:.4f})')
        print(f'   Biasing retrieval toward same-role reflections does NOT show a')
        print(f'   significant effect at n={pm["n"]} beyond the existing plain')
        print(f'   semantic retrieval already used in the published method.')
        print(f'   → Report as: shared-memory retrieval quality is already largely')
        print(f'     saturated by semantic similarity; explicit role-tag biasing')
        print(f'     does not add distinguishable lift at current sample size.')

    print('='*76)

    return {
        'non_role_conditioned_memory': pm,
        'role_conditioned_memory':     rm,
        'pairwise_agreement_pct': agree_pct,
        't_statistic': t_stat,
        'p_value':     p_val,
        'cohens_d':    cohens_d,
        'delta_pass3': delta_p3,
        'significant': bool(p_val < 0.05),
        'avg_memories_received_per_agent': {
            'non_role_conditioned': avg_memories_received(plain_analysis),
            'role_conditioned':     avg_memories_received(rc_analysis),
        },
        'communication_analysis': {
            'non_role_conditioned_memory': plain_analysis,
            'role_conditioned_memory':     rc_analysis,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Role-conditioned memory ablation (TMLR revision)')
    parser.add_argument('--tasks',  type=int, default=164)
    parser.add_argument('--outdir', default='../results')
    args = parser.parse_args()

    try:
        config = SecureConfigLoader().load_from_env_file('../.env')
    except Exception as e:
        logger.error(f'❌ Config error: {e}'); sys.exit(1)

    llm = BaseLLMModel(
        config['openrouter_api_key'],
        config['openrouter_model'],
        config['gemini_api_base'],
        config['rate_limit_delay'],
    )

    logger.info('📚 Loading HumanEval tasks...')
    try:
        tasks = HumanEvalLoader.load_from_file(
            '../HumanEval.jsonl.gz', num_samples=args.tasks)
        logger.info(f'✓ Loaded {len(tasks)} tasks')
    except FileNotFoundError:
        logger.error('❌ HumanEval.jsonl.gz not found!'); sys.exit(1)

    eta = len(tasks) * 2 * 9 * config['rate_limit_delay'] / 60
    logger.info(f'⚠️  ETA: ~{eta:.0f} minutes (both conditions)')
    input('\nPress ENTER to start role-conditioned memory ablation...')

    logger.info('\n⚪ Running MultiAgentReflexion (non-role-conditioned memory — published method)...')
    plain_agent = MultiAgentReflexion(llm, max_trials=3)
    plain_results, plain_task_results, plain_comm_log = run_agent(
        plain_agent, tasks, 'MultiAgentReflexion', llm)
    plain_analysis = compute_communication_analysis(
        plain_task_results, plain_comm_log,
        use_role_conditioning=plain_agent.use_role_conditioning,
        role_conditioned_memory=plain_agent.role_conditioned_memory,
    )

    logger.info('\n🟢 Running MultiAgentReflexionRoleConditionedMemory (ablation)...')
    rc_agent = MultiAgentReflexionRoleConditionedMemory(llm, max_trials=3)
    rc_results, rc_task_results, rc_comm_log = run_agent(
        rc_agent, tasks, 'MultiAgentReflexionRoleConditionedMemory', llm)
    rc_analysis = compute_communication_analysis(
        rc_task_results, rc_comm_log,
        use_role_conditioning=rc_agent.use_role_conditioning,
        role_conditioned_memory=rc_agent.role_conditioned_memory,
    )

    summary = report(plain_results, rc_results, plain_analysis, rc_analysis)

    output = {
        'dataset':   'HumanEval',
        'num_tasks': len(tasks),
        'task_ids':  [t['task_id'] for t in tasks],
        'summary':   summary,
        'results': {
            'multiagentreflexion':                     plain_results,
            'multiagentreflexionroleconditionedmemory': rc_results,
        },
    }

    out_path = Path(args.outdir) / 'extension2_role_conditioned_memory_ablation.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=numpy_encoder)

    print(f'\n💾 Saved → {out_path}')
    print('   This directly addresses the reviewer\'s "role-conditioned vs.')
    print('   non-role-conditioned memory" point.')


if __name__ == '__main__':
    main()

