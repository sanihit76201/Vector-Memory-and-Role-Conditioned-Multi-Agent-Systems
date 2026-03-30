"""Ablation study with statistical analysis."""

import time
import json
import logging
import numpy as np
from scipy import stats
from scipy.stats import sem, t as t_dist
from typing import List, Dict
from reflexion.agents import ReflexionAgent

logger = logging.getLogger(__name__)


class AblationStudy:
    """Run ablation studies comparing different memory architectures."""
    
    def __init__(self, llm, config: Dict):
        """
        Initialize ablation study.
        
        Args:
            llm: BaseLLMModel instance
            config: Configuration dictionary
        """
        self.llm = llm
        self.config = config
        self.results = []
    
    def run_benchmark(self, tasks: List[Dict], modes: List[str] = None) -> List[Dict]:
        """
        Run benchmark across different memory modes.
        
        Args:
            tasks: List of task dictionaries
            modes: List of memory modes to test (default: ['temporal', 'vector'])
            
        Returns:
            List of result dictionaries
        """
        if modes is None:
            modes = ['temporal', 'vector']
        
        self.results = []
        
        for mode in modes:
            logger.info(f'\n{"="*60}\n🧠 Mode: {mode.upper()}\n{"="*60}')
            agent = ReflexionAgent(self.llm, mode, max_trials=3)
            
            for i, task in enumerate(tasks):
                logger.info(f'\n📋 Task {i+1}/{len(tasks)}: {task["task_id"]}')
                logger.info(f'Prompt preview: {task["prompt"][:80]}...')
                
                result = agent.solve_task(task, verbose=True)
                result['mode'] = mode
                self.results.append(result)
                
                if i < len(tasks) - 1:
                    logger.info(f'⏳ Cooldown {self.config["rate_limit_delay"]}s')
                    time.sleep(self.config['rate_limit_delay'])
            
            agent.reset()
        
        return self.results
    
    def summary(self) -> str:
        """
        Generate comprehensive summary with statistics.
        
        Returns:
            Formatted summary string
        """
        modes = {}
        for r in self.results:
            modes.setdefault(r['mode'], []).append(r)
        
        # Basic task table
        table = '\n' + '='*80 + '\n📊 REFLEXION RESULTS - REAL HUMANEVAL\n' + '='*80 + '\n'
        table += f'{"Task":<20} | {"Mode":<10} | {"Success":<10} | {"Trials":<10}\n' + '-'*80 + '\n'
        
        for task_id in sorted(set(r['task_id'] for r in self.results)):
            for mode in sorted(modes.keys()):
                rs = [r for r in self.results if r['task_id']==task_id and r['mode']==mode]
                if rs:
                    r = rs[0]
                    table += f"{task_id:<20} | {mode:<10} | {'✅' if r['success'] else '❌':<10} | {r['trials']:<10}\n"
        
        table += '-'*80 + '\n\n'
        table += '='*80 + '\n📈 QUANTIFIED PERFORMANCE METRICS\n' + '='*80 + '\n\n'
        table += '1. PRIMARY METRICS\n' + '-'*80 + '\n'
        
        for mode in sorted(modes.keys()):
            passed = sum(1 for r in modes[mode] if r['success'])
            total = len(modes[mode])
            pass_rate = (passed/total*100) if total else 0
            
            pass_at_1 = sum(1 for r in modes[mode] if r['success'] and r['trials'] == 1)
            pass_at_1_rate = (pass_at_1/total*100) if total else 0
            
            successful_trials = [r['trials'] for r in modes[mode] if r['success']]
            avg_trials = np.mean(successful_trials) if successful_trials else 0
            
            table += f'\n{mode.upper()}:\n'
            table += f'  Pass@3 (Overall):   {passed}/{total} = {pass_rate:.1f}%\n'
            table += f'  Pass@1 (Trial 1):   {pass_at_1}/{total} = {pass_at_1_rate:.1f}%\n'
            table += f'  Avg Trials:         {avg_trials:.2f}\n'
        
        if 'temporal' in modes and 'vector' in modes:
            table += '\n2. COMPARATIVE ANALYSIS\n' + '-'*80 + '\n'
            
            temp_passed = sum(1 for r in modes['temporal'] if r['success'])
            temp_total = len(modes['temporal'])
            vec_passed = sum(1 for r in modes['vector'] if r['success'])
            vec_total = len(modes['vector'])
            
            temp_rate = (temp_passed/temp_total*100) if temp_total else 0
            vec_rate = (vec_passed/vec_total*100) if vec_total else 0
            
            abs_improvement = vec_rate - temp_rate
            rel_improvement = (abs_improvement / temp_rate * 100) if temp_rate > 0 else 0
            
            table += f'Absolute Improvement: {abs_improvement:+.1f} percentage points\n'
            table += f'Relative Improvement: {rel_improvement:+.1f}%\n'
        
        table += '\n' + '='*80 + '\n'
        return table
    
    def save_results(self, filename: str = 'results/experiment_results.json'):
        """Save results to JSON file."""
        output_data = {
            'dataset': 'HumanEval',
            'tasks_count': len(set(r['task_id'] for r in self.results)),
            'results': self.results,
            'config': {k: v for k, v in self.config.items() if 'key' not in k.lower()}
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f'\n✓ Results saved to {filename}')
