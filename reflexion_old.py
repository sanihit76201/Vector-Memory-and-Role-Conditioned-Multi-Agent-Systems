# Reflexion Framework - REAL HumanEval Implementation
# Download HumanEval: wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz

import os
import json
import gzip
import time
import random
import logging
import tempfile
import subprocess
import sys
import ast
from typing import Dict, List, Tuple, Optional, Any
from functools import wraps
from collections import deque
import numpy as np
from scipy import stats
from scipy.stats import sem, t as t_dist
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================
class SecureConfigLoader:
    def __init__(self):
        self.config = {}
    
    def load_from_env_file(self, env_path: str = '.env') -> Dict[str, str]:
        if not os.path.exists(env_path):
            logger.warning(f'{env_path} not found. Creating template...')
            template = '''OPENROUTER_API_KEY=sk-or-v1-YOUR-KEY-HERE
OPENROUTER_MODEL=google/gemini-2.0-flash-exp:free
GEMINI_API_BASE=https://openrouter.ai/api/v1/
RATE_LIMIT_DELAY=30.0
'''
            with open(env_path, 'w') as f:
                f.write(template)
            raise FileNotFoundError(f'Fill {env_path} with API key and run again.')
        
        load_dotenv(env_path)
        self.config = {
            'openrouter_api_key': os.getenv('OPENROUTER_API_KEY'),
            'openrouter_model': os.getenv('OPENROUTER_MODEL', 'google/gemini-2.0-flash-exp:free'),
            'gemini_api_base': os.getenv('GEMINI_API_BASE', 'https://openrouter.ai/api/v1/'),
            'rate_limit_delay': float(os.getenv('RATE_LIMIT_DELAY', '30.0')),
        }
        
        if not self.config['openrouter_api_key'] or 'YOUR-KEY' in self.config['openrouter_api_key']:
            raise ValueError('Invalid OPENROUTER_API_KEY in .env')
        
        logger.info('✓ Config loaded')
        return self.config

def exponential_backoff(max_retries=5, initial_delay=5.0):
    """Enhanced retry with connection error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.SSLError, 
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout) as e:
                    if attempt < max_retries:
                        wait = delay + random.uniform(0, delay * 0.2)
                        logger.warning(f'⚠ Connection error. Retry in {wait:.1f}s ({attempt+1}/{max_retries})')
                        time.sleep(wait)
                        delay *= 2.5
                    else:
                        raise
                except Exception as e:
                    if attempt < max_retries and ('429' in str(e) or 'rate' in str(e).lower()):
                        wait = delay + random.uniform(0, delay * 0.2)
                        logger.warning(f'⚠ Rate limited. Retry in {wait:.1f}s ({attempt+1}/{max_retries})')
                        time.sleep(wait)
                        delay *= 2.5
                    else:
                        raise
        return wrapper
    return decorator

# ============================================================================
# LLM WITH ROBUST CONNECTION HANDLING
# ============================================================================
class BaseLLMModel:
    def __init__(self, api_key, model='google/gemini-2.0-flash-exp:free', 
                 api_base='https://openrouter.ai/api/v1/', rate_limit_delay=30.0):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.rate_limit_delay = rate_limit_delay
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'HTTP-Referer': 'https://github.com/reflexion',
            'X-Title': 'Reflexion Study'
        }
        self.last_call_time = 0
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _wait(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.rate_limit_delay:
            wait = self.rate_limit_delay - elapsed
            logger.info(f'⏳ Rate protection: {wait:.1f}s')
            time.sleep(wait)
        self.last_call_time = time.time()

    @exponential_backoff(max_retries=5, initial_delay=5.0)
    def call_llm(self, prompt: str, max_tokens=1024) -> str:
        self._wait()
        response = self.session.post(
            f"{self.api_base}chat/completions",
            headers=self.headers,
            json={'model': self.model, 'messages': [{'role': 'user', 'content': prompt}], 
                  'max_tokens': max_tokens, 'temperature': 0.7},
            timeout=120  # Increased timeout
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def get_embedding(self, text: str) -> np.ndarray:
    # REAL semantic embeddings (not hash)
        from sentence_transformers import SentenceTransformer
    
    # Initialize model once
        if not hasattr(self, '_embed_model'):
            logger.info('📦 Loading embedding model (one-time, ~90MB)...')
            self._embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info('✓ Embedding model ready')
    
    # Return real 384-dim semantic vector
        return self._embed_model.encode(text, show_progress_bar=False)


# ============================================================================
# MEMORY
# ============================================================================
class TemporalMemory:
    def __init__(self, max_size=10):
        self.reflections = deque(maxlen=max_size)
    
    def add_reflection(self, refl: str):
        self.reflections.append(refl)
    
    def get_relevant_memories(self, query="", k=3) -> List[str]:
        return list(self.reflections)[-k:]
    
    def clear(self):
        self.reflections.clear()

class VectorEpisodicMemory:
    def __init__(self, llm, max_size=100):
        self.llm = llm
        self.reflections = deque(maxlen=max_size)
        self.embeddings = deque(maxlen=max_size)
    
    def add_reflection(self, refl: str):
        emb = self.llm.get_embedding(refl)
        self.reflections.append(refl)
        self.embeddings.append(emb)
    
    def get_relevant_memories(self, query: str, k=3) -> List[str]:
        if not self.reflections:
            return []
        q_emb = self.llm.get_embedding(query)
        sims = cosine_similarity(q_emb.reshape(1, -1), np.array(list(self.embeddings)))[0]
        top_k = np.argsort(sims)[-k:]
        return [list(self.reflections)[i] for i in sorted(top_k)]
    
    def clear(self):
        self.reflections.clear()
        self.embeddings.clear()

# ============================================================================
# EVALUATOR - HUMANEVAL COMPATIBLE
# ============================================================================
class ObjectiveCodeEvaluator:
    def __init__(self, timeout=10):
        self.timeout = timeout
    
    def evaluate(self, code: str, entry_point: str, test_code: str) -> Dict:
        """Evaluate code using HumanEval's test format."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {'passed': False, 'error': f'SyntaxError: {e}'}
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write solution code
                f.write(code + '\n\n')
                # Write test code
                f.write(test_code + '\n\n')
                # Run check function
                f.write('check({})\n'.format(entry_point))
                f.write('print("PASS")\n')
                temp = f.name
            
            result = subprocess.run(
                [sys.executable, temp], 
                capture_output=True, 
                text=True, 
                timeout=self.timeout
            )
            
            os.unlink(temp)
            
            if 'PASS' in result.stdout and not result.stderr:
                return {'passed': True, 'error': None}
            else:
                error = result.stderr or result.stdout
                return {'passed': False, 'error': error[:200]}
                
        except subprocess.TimeoutExpired:
            try:
                os.unlink(temp)
            except:
                pass
            return {'passed': False, 'error': 'Timeout'}
        except Exception as e:
            return {'passed': False, 'error': str(e)[:200]}

# ============================================================================
# HUMANEVAL LOADER - REAL DATA
# ============================================================================
class HumanEvalLoader:
    @staticmethod
    def load_from_file(file_path: str = 'HumanEval.jsonl.gz', num_samples: int = 3) -> List[Dict]:
        """Load REAL HumanEval tasks."""
        if not os.path.exists(file_path):
            logger.error(f'\n❌ {file_path} not found!')
            logger.error('\n📥 Download it first:')
            logger.error('   wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz')
            logger.error('\n   Or run:')
            logger.error('   curl -L -o HumanEval.jsonl.gz https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz')
            raise FileNotFoundError(f'{file_path} not found. Please download it first.')
        
        tasks = []
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                task = json.loads(line)
                tasks.append({
                    'task_id': task['task_id'],
                    'prompt': task['prompt'],
                    'entry_point': task['entry_point'],
                    'canonical_solution': task['canonical_solution'],
                    'test': task['test']
                })
        
        logger.info(f'✓ Loaded {len(tasks)} REAL HumanEval tasks')
        return tasks

# ============================================================================
# AGENT
# ============================================================================
class ReflexionAgent:
    def __init__(self, llm, memory_mode='temporal', max_trials=3):
        self.llm = llm
        self.max_trials = max_trials
        self.evaluator = ObjectiveCodeEvaluator(timeout=10)
        self.memory = TemporalMemory() if memory_mode == 'temporal' else VectorEpisodicMemory(llm)
        self.memory_mode = memory_mode
    
    def solve_task(self, task: Dict, verbose=False) -> Dict:
        task_id = task['task_id']
        
        for trial in range(self.max_trials):
            memories = (self.memory.get_relevant_memories(k=3) if self.memory_mode == 'temporal' 
                       else self.memory.get_relevant_memories(task['prompt'], k=3))
            
            mem_ctx = '\n'.join(f'- {m}' for m in memories) if memories else 'None'
            
            prompt = f"""You are an expert Python programmer. Complete this function:

{task['prompt']}

Past reflections (learn from mistakes):
{mem_ctx}

Requirements:
1. Complete the function implementation
2. Handle all edge cases
3. Make sure all test cases pass
4. Output ONLY the Python code, no markdown, no explanations

Your code:"""
            
            try:
                logger.info(f'🔄 Trial {trial+1}/{self.max_trials} - {task_id}')
                code = self.llm.call_llm(prompt, max_tokens=2048)
                
                # Clean markdown
                if '```python' in code:
                    code = code.split('```python')[1].split('```')[0].strip()
                elif '```' in code:
                    code = code.split('```')[1].split('```')[0].strip()
                
                # Evaluate using HumanEval test
                results = self.evaluator.evaluate(code, task['entry_point'], task['test'])
                
                if results['passed']:
                    logger.info(f'✅ {task_id} solved in {trial+1} trials')
                    return {'task_id': task_id, 'success': True, 'trials': trial+1, 'code': code}
                
                refl = f"Trial {trial+1} failed: {results['error']}"
                self.memory.add_reflection(refl)
                logger.warning(f'❌ {refl[:100]}...')
                
            except KeyboardInterrupt:
                logger.error('\n⚠️  Interrupted by user')
                raise
            except Exception as e:
                refl = f"Trial {trial+1} error: {str(e)[:100]}"
                self.memory.add_reflection(refl)
                logger.error(f'❌ {refl}')
        
        logger.warning(f'❌ {task_id} failed after {self.max_trials} trials')
        return {'task_id': task_id, 'success': False, 'trials': self.max_trials}
    
    def reset(self):
        self.memory.clear()

# ============================================================================
# ABLATION STUDY
# ============================================================================
class AblationStudy:
    def __init__(self, config):
        self.config = config
        self.llm = BaseLLMModel(
            config['openrouter_api_key'], 
            config['openrouter_model'],
            config['gemini_api_base'], 
            config['rate_limit_delay']
        )
        self.results = []
    
    def run_benchmark(self, tasks, modes=['temporal', 'vector']):
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
    
    def summary(self):
        """Enhanced summary with statistical quantification."""

    
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
    
    # ========================================================================
    # QUANTIFIED METRICS (WHAT YOUR MENTOR WANTS)
    # =======================================================================
        table += '='*80 + '\n📈 QUANTIFIED PERFORMANCE METRICS\n' + '='*80 + '\n\n'
    
    # 1. PRIMARY METRICS
        table += '1. PRIMARY METRICS\n' + '-'*80 + '\n'
    
        for mode in sorted(modes.keys()):
            passed = sum(1 for r in modes[mode] if r['success'])
            total = len(modes[mode])
            pass_rate = (passed/total*100) if total else 0
        
        # Pass@1 (solved in first trial)
            pass_at_1 = sum(1 for r in modes[mode] if r['success'] and r['trials'] == 1)
            pass_at_1_rate = (pass_at_1/total*100) if total else 0
        
        # Average trials for successful tasks
            successful_trials = [r['trials'] for r in modes[mode] if r['success']]
            avg_trials = np.mean(successful_trials) if successful_trials else 0
        
            table += f'\n{mode.upper()}:\n'
            table += f'  Pass@3 (Overall):   {passed}/{total} = {pass_rate:.1f}%\n'
            table += f'  Pass@1 (Trial 1):   {pass_at_1}/{total} = {pass_at_1_rate:.1f}%\n'
            table += f'  Avg Trials:         {avg_trials:.2f}\n'
    
    # 2. COMPARATIVE ANALYSIS
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
        
        # Trials comparison
            temp_trials = [r['trials'] for r in modes['temporal'] if r['success']]
            vec_trials = [r['trials'] for r in modes['vector'] if r['success']]
        
            if temp_trials and vec_trials:
                temp_avg = np.mean(temp_trials)
                vec_avg = np.mean(vec_trials)
                trial_improvement = temp_avg - vec_avg
                trial_pct = (trial_improvement / temp_avg * 100) if temp_avg > 0 else 0
            
                table += f'Trial Reduction: {trial_improvement:+.2f} trials ({trial_pct:+.1f}%)\n'
    
    # 3. LEARNING EFFICIENCY
        table += '\n3. LEARNING EFFICIENCY\n' + '-'*80 + '\n'
    
        for mode in sorted(modes.keys()):
        # Trial 1 failures
            failed_trial_1 = len([r for r in modes[mode] if not (r['success'] and r['trials'] == 1)])
        
        # Recovery in Trial 2+
            recovered = sum(1 for r in modes[mode] if r['success'] and r['trials'] > 1)
        
            recovery_rate = (recovered/failed_trial_1*100) if failed_trial_1 > 0 else 0
        
            table += f'{mode.upper()} Recovery Rate: {recovered}/{failed_trial_1} = {recovery_rate:.1f}%\n'
    
    # 4. STATISTICAL VALIDATION
        if 'temporal' in modes and 'vector' in modes:
            table += '\n4. STATISTICAL VALIDATION\n' + '-'*80 + '\n'
        
        # Binary success arrays
            temp_binary = [1 if r['success'] else 0 for r in modes['temporal']]
            vec_binary = [1 if r['success'] else 0 for r in modes['vector']]
        
            if len(temp_binary) == len(vec_binary) and len(temp_binary) > 1:
            # Paired t-test
                t_stat, p_value = stats.ttest_rel(vec_binary, temp_binary)
            
                table += f'Paired t-test:\n'
                table += f'  t-statistic: {t_stat:.3f}\n'
                table += f'  p-value: {p_value:.3f}\n'
            
                if p_value < 0.05:
                    table += f'  ✅ Statistically significant (p < 0.05)\n'
                elif p_value < 0.10:
                    table += f'  ⚠️  Marginally significant (0.05 < p < 0.10)\n'
                else:
                    table += f'  ❌ Not significant (need more samples)\n'
            
            # Effect size (Cohen's d)
                def cohens_d(group1, group2):
                    diff = np.mean(group1) - np.mean(group2)
                    n1, n2 = len(group1), len(group2)
                    var1 = np.var(group1, ddof=1) if n1 > 1 else 0
                    var2 = np.var(group2, ddof=1) if n2 > 1 else 0
                    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)) if (n1+n2) > 2 else 1
                    return diff / pooled_std if pooled_std > 0 else 0
            
                effect = cohens_d(vec_binary, temp_binary)
                table += f'\nEffect Size (Cohen\'s d): {effect:.3f}\n'
            
                if abs(effect) < 0.2:
                    table += f'  → Negligible effect\n'
                elif abs(effect) < 0.5:
                    table += f'  → Small effect\n'
                elif abs(effect) < 0.8:
                    table += f'  → Medium effect\n'
                else:
                    table += f'  → Large effect\n'
            
            # Confidence intervals
                if len(temp_binary) > 1:
                    def ci_95(data):
                        n = len(data)
                        mean = np.mean(data)
                        std_err = sem(data)
                        margin = std_err * t_dist.ppf(0.975, n - 1)
                        return mean, mean - margin, mean + margin
                
                    temp_mean, temp_lower, temp_upper = ci_95(temp_binary)
                    vec_mean, vec_lower, vec_upper = ci_95(vec_binary)
                
                    table += f'\n95% Confidence Intervals:\n'
                    table += f'  Temporal: {temp_mean*100:.1f}% [{temp_lower*100:.1f}% - {temp_upper*100:.1f}%]\n'
                    table += f'  Vector:   {vec_mean*100:.1f}% [{vec_lower*100:.1f}% - {vec_upper*100:.1f}%]\n'
    
    # 5. TRIAL DISTRIBUTION
        table += '\n5. TRIAL DISTRIBUTION\n' + '-'*80 + '\n'
    
        for mode in sorted(modes.keys()):
            trial_counts = {}
            for r in modes[mode]:
                if r['success']:
                    trial_counts[r['trials']] = trial_counts.get(r['trials'], 0) + 1
        
            table += f'{mode.upper()}:\n'
            for trial in sorted(trial_counts.keys()):
                count = trial_counts[trial]
                pct = count / len(modes[mode]) * 100
                table += f'  Trial {trial}: {count} tasks ({pct:.1f}%)\n'
    
        table += '='*80 + '\n'
        return table

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*80)
    print("🧠 REFLEXION FRAMEWORK - REAL HUMANEVAL BENCHMARK")
    print("="*80 + "\n")
    
    # Load config
    try:
        config = SecureConfigLoader().load_from_env_file('.env')
    except Exception as e:
        logger.error(f'❌ {e}')
        return
    
    logger.info(f'Model: {config["openrouter_model"]}')
    logger.info(f'Rate limit delay: {config["rate_limit_delay"]}s')
    
    # Load REAL HumanEval
    logger.info('\n📚 Loading REAL HumanEval dataset...')
    try:
        tasks = HumanEvalLoader.load_from_file('HumanEval.jsonl.gz', num_samples=50)
    except FileNotFoundError:
        return
    
    eta_minutes = (len(tasks) * 2 * config['rate_limit_delay'] * 3) / 60
    logger.info(f'\n⚠️  Estimated time: ~{eta_minutes:.1f} minutes')
    logger.info('   (3 tasks × 2 modes × 3 max trials × 30s delay)\n')
    
    input('Press ENTER to start benchmark...')
    
    # Run study
    study = AblationStudy(config)
    
    try:
        study.run_benchmark(tasks)
    except KeyboardInterrupt:
        logger.error('\n\n❌ Benchmark interrupted by user')
        if study.results:
            logger.info('Saving partial results...')
    
    # Show results
    print('\n' + study.summary())
    
    # Save
    output_file = 'reflexion_humaneval_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': 'HumanEval (real)',
            'tasks_count': len(tasks),
            'results': study.results,
            'config': {k: v for k, v in config.items() if 'key' not in k.lower()}
        }, f, indent=2)
    
    logger.info(f'\n✓ Results saved to {output_file}')
    logger.info('\n🎉 Benchmark complete!')

if __name__ == '__main__':
    main()