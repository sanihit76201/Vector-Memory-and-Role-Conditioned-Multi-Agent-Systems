"""EXACT replica of your original working single-file Reflexion agent."""

import logging
from typing import Dict, List
from ..evaluators import ObjectiveCodeEvaluator
from ..memory import TemporalMemory, VectorEpisodicMemory

logger = logging.getLogger(__name__)


class OriginalReflexionAgent:
    """Matches your original reflexion_old.py behavior exactly."""
    
    def __init__(self, llm, memory_mode: str = 'temporal', max_trials: int = 3):
        self.llm = llm
        self.max_trials = max_trials
        self.evaluator = ObjectiveCodeEvaluator(timeout=10)
        
        # Original memory sizes
        if memory_mode == 'temporal':
            self.memory = TemporalMemory(max_size=10)
        else:
            self.memory = VectorEpisodicMemory(llm, max_size=100)
        
        self.memory_mode = memory_mode
    
    def solve_task(self, task: Dict, verbose: bool = False) -> Dict:
        """Exact original solve_task logic - NO changes."""
        task_id = task['task_id']
        
        for trial in range(self.max_trials):
            # Original memory retrieval
            if self.memory_mode == 'temporal':
                memories = self.memory.get_relevant_memories(k=3)
            else:
                memories = self.memory.get_relevant_memories(task['prompt'], k=3)
            
            # EXACT original prompt wording
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
                # Original token limit
                code = self.llm.call_llm(prompt, max_tokens=2048)
                
                # Original code cleaning
                if '```python' in code:
                    code = code.split('```python')[1].split('```')[0].strip()
                elif '```' in code:
                    code = code.split('```')[1].split('```')[0].strip()
                
                # Evaluate
                results = self.evaluator.evaluate(code, task['entry_point'], task['test'])
                
                if results['passed']:
                    if verbose:
                        logger.info(f'✅ {task_id} solved in {trial+1} trials')
                    return {
                        'task_id': task_id,
                        'success': True,
                        'trials': trial+1,
                        'agent_type': 'Original'
                    }
                
                # EXACT original reflection
                reflection = f"Trial {trial+1} failed: {results['error']}"
                self.memory.add_reflection(reflection)
                
            except Exception as e:
                reflection = f"Trial {trial+1} error: {str(e)[:100]}"
                self.memory.add_reflection(reflection)
        
        return {
            'task_id': task_id,
            'success': False,
            'trials': self.max_trials,
            'agent_type': 'Original'
        }
