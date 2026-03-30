"""
Run HumanEval benchmark with Reflexion agents.
Compare temporal vs vector memory architectures.
"""

import logging
import sys
sys.path.insert(0, '..')  # Add parent directory to path

from reflexion.config import SecureConfigLoader
from reflexion.llm import BaseLLMModel
from reflexion.benchmarks import HumanEvalLoader
from analysis.ablation_study import AblationStudy

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run HumanEval benchmark."""
    print("\n" + "="*80)
    print("🧠 REFLEXION FRAMEWORK - HUMANEVAL BENCHMARK")
    print("="*80 + "\n")
    
    # Load configuration
    try:
        config = SecureConfigLoader().load_from_env_file('../.env')
    except Exception as e:
        logger.error(f'❌ Config error: {e}')
        return
    
    logger.info(f'Model: {config["openrouter_model"]}')
    logger.info(f'Rate limit delay: {config["rate_limit_delay"]}s')
    
    # Initialize LLM
    llm = BaseLLMModel(
        config['openrouter_api_key'],
        config['openrouter_model'],
        config['gemini_api_base'],
        config['rate_limit_delay']
    )
    
    # Load HumanEval tasks
    logger.info('\n📚 Loading HumanEval dataset...')
    try:
        tasks = HumanEvalLoader.load_from_file('../HumanEval.jsonl.gz', num_samples=20)
    except FileNotFoundError:
        logger.error('❌ HumanEval.jsonl.gz not found in parent directory')
        return
    
    # Estimate time
    eta_minutes = (len(tasks) * 2 * config['rate_limit_delay'] * 3) / 60
    logger.info(f'\n⚠️  Estimated time: ~{eta_minutes:.1f} minutes')
    logger.info(f'   ({len(tasks)} tasks × 2 modes × 3 max trials × {config["rate_limit_delay"]}s delay)\n')
    
    input('Press ENTER to start benchmark...')
    
    # Run ablation study
    study = AblationStudy(llm, config)
    
    try:
        study.run_benchmark(tasks, modes=['temporal', 'vector'])
    except KeyboardInterrupt:
        logger.error('\n\n❌ Benchmark interrupted by user')
        if study.results:
            logger.info('Saving partial results...')
    
    # Display and save results
    print('\n' + study.summary())
    
    study.save_results('../results/humaneval_results.json')
    
    logger.info('\n🎉 Benchmark complete!')


if __name__ == '__main__':
    main()
