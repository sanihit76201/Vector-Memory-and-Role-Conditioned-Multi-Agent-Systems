"""
🔬 EXTENSIONS 1-2: COMPLETE COMPARISON BENCHMARK WITH QUANTIFIED METRICS
Tests Extensions 1-2: Semantic Vector Memory + Multi-Agent Collaboration
"""

import logging
import sys
import json
import numpy as np
import argparse
sys.path.insert(0, '..')

from reflexion.config import SecureConfigLoader
from reflexion.llm import BaseLLMModel
from reflexion.benchmarks import HumanEvalLoader
from reflexion.agents import (
    ReflexionAgent,
    OriginalReflexionAgent,
    VectorReflexionAgent,
    MultiAgentReflexion,
)
from reflexion.memory.vector import VectorEpisodicMemory

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_agent(agent, tasks, name):
    """Run agent on tasks and return results."""
    logger.info(f'\n{"="*80}\n🧠 Running: {name}\n{"="*80}')
    results = []

    for i, task in enumerate(tasks):
        logger.info(f'\n📋 Task {i+1}/{len(tasks)}: {task["task_id"]}')
        logger.info(f'Prompt preview: {task["prompt"][:80]}...')
        result = agent.solve_task(task, verbose=True)
        result['agent_type'] = name
        results.append(result)

        if not isinstance(getattr(agent, 'memory', None), VectorEpisodicMemory):
            if hasattr(agent, 'reset'):
                agent.reset()
            elif hasattr(agent, 'memory'):
                agent.memory.clear()

    return results

def print_quantified_metrics(baseline_results, original_results, extension_results, extension_name, total_tasks):
    """Print comprehensive quantified metrics table with statistical validation."""
    modes = {
        'Modular_Baseline': baseline_results,
        'Original_Working': original_results,
        extension_name: extension_results
    }

    print('='*80)
    print('📈 QUANTIFIED PERFORMANCE METRICS')
    print('='*80)

    # 1. PRIMARY METRICS
    print('\n1. PRIMARY METRICS')
    print('-'*50)

    for mode_name, results in modes.items():
        passed = sum(1 for r in results if r['success'])
        total = len(results)
        pass_rate = passed/total*100 if total > 0 else 0

        pass_at_1 = sum(1 for r in results if r['success'] and r['trials'] == 1)
        pass1_rate = pass_at_1/total*100 if total > 0 else 0

        successful_trials = [r['trials'] for r in results if r['success']]
        avg_trials = np.mean(successful_trials) if successful_trials else 0

        print(f'\n{mode_name.upper()}:')
        print(f'  Pass@3: {passed}/{total} ({pass_rate:.1f}%)')
        print(f'  Pass@1: {pass_at_1}/{total} ({pass1_rate:.1f}%)')
        print(f'  Avg Trials: {avg_trials:.2f}')

    # 2. COMPARATIVE ANALYSIS
    print('\n2. COMPARATIVE ANALYSIS')
    print('-'*50)

    baseline_passed = sum(1 for r in baseline_results if r['success'])
    baseline_total = len(baseline_results)
    ext_passed = sum(1 for r in extension_results if r['success'])
    ext_total = len(extension_results)

    baseline_rate = baseline_passed/baseline_total*100 if baseline_total > 0 else 0
    ext_rate = ext_passed/ext_total*100 if ext_total > 0 else 0

    abs_improvement = ext_rate - baseline_rate
    rel_improvement = (abs_improvement/baseline_rate*100) if baseline_rate > 0 else 0

    print(f'Extension vs Baseline:')
    print(f'  Absolute: {abs_improvement:+.1f}%')
    print(f'  Relative: {rel_improvement:+.1f}%')

    # 3. LEARNING EFFICIENCY
    print('\n3. LEARNING EFFICIENCY')
    print('-'*50)

    for mode_name, results in modes.items():
        failed_t1 = len([r for r in results if not (r['success'] and r['trials'] == 1)])
        recovered = sum(1 for r in results if r['success'] and r['trials'] > 1)
        recovery_rate = recovered/failed_t1*100 if failed_t1 > 0 else 100
        print(f'{mode_name}: {recovered}/{failed_t1} recovery ({recovery_rate:.1f}%)')

    # 4. STATISTICAL VALIDATION
    print('\n4. STATISTICAL VALIDATION')
    print('-'*50)

    baseline_binary = [1 if r['success'] else 0 for r in baseline_results]
    ext_binary = [1 if r['success'] else 0 for r in extension_results]

    if len(baseline_binary) == len(ext_binary) and len(baseline_binary) > 1:
        from scipy import stats

        t_stat, p_value = stats.ttest_rel(ext_binary, baseline_binary)

        print(f'Paired t-test (Extension vs Baseline):')
        print(f'  t-statistic: {t_stat:.3f}')
        print(f'  p-value: {p_value:.3f}')

        if p_value < 0.05:
            print(f'  ✅ Statistically significant (p < 0.05)')
        elif p_value < 0.10:
            print(f'  ⚠️  Marginally significant (0.05 < p < 0.10)')
        else:
            print(f'  ❌ Not significant (need more samples)')

        def cohens_d(group1, group2):
            diff = np.mean(group1) - np.mean(group2)
            n1, n2 = len(group1), len(group2)
            var1 = np.var(group1, ddof=1) if n1 > 1 else 0
            var2 = np.var(group2, ddof=1) if n2 > 1 else 0
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)) if (n1+n2) > 2 else 1
            return diff / pooled_std if pooled_std > 0 else 0

        effect = cohens_d(ext_binary, baseline_binary)
        print(f'\nEffect Size (Cohen\'s d): {effect:.3f}')

        if abs(effect) < 0.2:
            print(f'  → Negligible effect')
        elif abs(effect) < 0.5:
            print(f'  → Small effect')
        elif abs(effect) < 0.8:
            print(f'  → Medium effect')
        else:
            print(f'  → Large effect')

        def ci_95(data):
            n = len(data)
            mean = np.mean(data)
            std_err = stats.sem(data)
            margin = std_err * stats.t.ppf(0.975, n - 1)
            return mean, mean - margin, mean + margin

        baseline_mean, baseline_lower, baseline_upper = ci_95(baseline_binary)
        ext_mean, ext_lower, ext_upper = ci_95(ext_binary)

        print(f'\n95% Confidence Intervals:')
        print(f'  Baseline:  {baseline_mean*100:.1f}% [{baseline_lower*100:.1f}% - {baseline_upper*100:.1f}%]')
        print(f'  Extension: {ext_mean*100:.1f}% [{ext_lower*100:.1f}% - {ext_upper*100:.1f}%]')

    print('='*80)


def main():
    """Run configurable comparison with quantified metrics for Extensions 1 and 2."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--extension', choices=['vector', 'multiagent'], default='vector',
                        help='vector = Extension 1 (Semantic Vector Memory), multiagent = Extension 2 (Multi-Agent)')
    parser.add_argument('--tasks', type=int, default=50)
    args = parser.parse_args()

    print("\n" + "="*80)
    if args.extension == 'multiagent':
        print("🔬 EXTENSION 2: MULTI-AGENT COLLABORATION")
        extension_num = 2
    else:
        print("🔬 EXTENSION 1: VECTOR MEMORY (Semantic)")
        extension_num = 1
    print("="*80 + "\n")

    # Load config
    try:
        config = SecureConfigLoader().load_from_env_file('../.env')
        logger.info(f'Model: {config["openrouter_model"]}')
        logger.info(f'Rate limit: {config["rate_limit_delay"]}s')
    except Exception as e:
        logger.error(f'❌ Config error: {e}')
        sys.exit(1)

    # Initialize LLM
    llm = BaseLLMModel(
        config['openrouter_api_key'],
        config['openrouter_model'],
        config['gemini_api_base'],
        config['rate_limit_delay']
    )

    # Load tasks
    logger.info('\n📚 Loading HumanEval tasks...')
    try:
        tasks = HumanEvalLoader.load_from_file('../HumanEval.jsonl.gz', num_samples=args.tasks)
        logger.info(f'✓ Loaded {len(tasks)} tasks')
    except FileNotFoundError:
        logger.error('❌ HumanEval.jsonl.gz not found!')
        sys.exit(1)

    # ETA estimate
    multiplier = 12 if args.extension == 'multiagent' else 4
    eta_minutes = len(tasks) * 3 * config['rate_limit_delay'] * multiplier / 60
    logger.info(f'⚠️  ETA: ~{eta_minutes:.1f} minutes')
    input('\nPress ENTER to continue...')

    # 1. MODULAR BASELINE
    logger.info('\n🔵 Testing MODULAR BASELINE...')
    baseline_agent = ReflexionAgent(llm, memory_mode='temporal', max_trials=3)
    baseline_results = run_agent(baseline_agent, tasks, 'Modular_Baseline')

    # 2. ORIGINAL
    logger.info('\n🟡 Testing ORIGINAL...')
    original_agent = OriginalReflexionAgent(llm, memory_mode='temporal', max_trials=3)
    original_results = run_agent(original_agent, tasks, 'Original_Working')

    # 3. EXTENSION AGENT
    if args.extension == 'multiagent':
        print('\n🔴 Testing MULTI-AGENT (Extension 2)...')
        ext_agent = MultiAgentReflexion(llm, max_trials=3)
        extension_results = run_agent(ext_agent, tasks, 'MultiAgentReflexion')
        extension_name = 'MultiAgentReflexion'
    else:
        print('\n🟣 Testing VECTOR REFLEXION (Extension 1)...')
        ext_agent = VectorReflexionAgent(llm, max_trials=3)
        extension_results = run_agent(ext_agent, tasks, 'VectorReflexion')
        extension_name = 'VectorReflexion'

    # QUANTIFIED METRICS
    print_quantified_metrics(baseline_results, original_results, extension_results, extension_name, len(tasks))

    # SIMPLE SUMMARY
    print("\n" + "="*50 + " SIMPLE SUMMARY " + "="*50)
    agent_types = ['Modular_Baseline', 'Original_Working', extension_name]
    all_results = baseline_results + original_results + extension_results

    for agent_type in agent_types:
        results = [r for r in all_results if r['agent_type'] == agent_type]
        passed = sum(1 for r in results if r['success'])
        total = len(results)
        pass_rate = passed/total*100

        pass_at_1 = sum(1 for r in results if r['success'] and r['trials'] == 1)
        pass1_rate = pass_at_1/total*100

        successful_trials = [r['trials'] for r in results if r['success']]
        avg_trials = np.mean(successful_trials) if successful_trials else float('nan')

        print(f"🏃 {agent_type}:")
        print(f"   Pass@3:  {passed}/{total} ({pass_rate:.1f}%)")
        print(f"   Pass@1:  {pass_at_1}/{total} ({pass1_rate:.1f}%)")
        print(f"   Avg trials: {avg_trials:.2f}" if not np.isnan(avg_trials) else "   Avg trials: N/A")
        print()

    # Save results
    def numpy_encoder(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object {type(obj)} not serializable")

    output = {
        'dataset': 'HumanEval',
        'num_tasks': len(tasks),
        'task_ids': [t['task_id'] for t in tasks],
        'extension_tested': args.extension,
        'results': {
            'modular_baseline': baseline_results,
            'original_working': original_results,
            extension_name.lower(): extension_results
        }
    }

    filename = f'../results/seed_runs/extension{extension_num}_{args.extension}_agent.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=numpy_encoder)

    print(f"💾 Results saved: {filename}")
    print("\n🎉 EXTENSION COMPLETE WITH QUANTIFIED METRICS!")
    logger.info('✅ PAPER READY: Full ablation study complete!')


if __name__ == '__main__':
    main()