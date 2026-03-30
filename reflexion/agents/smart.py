"""Smart Reflexion Agent - Extension 1: Task-Isolated Memory (BUG-FIXED)."""

import logging
from typing import Dict
from ..evaluators import ObjectiveCodeEvaluator
from ..memory import TemporalMemory

logger = logging.getLogger(__name__)


class SmartReflexionAgent:
    def __init__(self, llm, max_trials: int = 3):
        self.llm = llm
        self.max_trials = max_trials
        self.evaluator = ObjectiveCodeEvaluator(timeout=10)
        self.memory = TemporalMemory(max_size=3)
        self.current_task = None
    
    def solve_task(self, task: Dict, verbose: bool = False) -> Dict:
        task_id = task['task_id']
        
        if self.current_task != task_id:
            self.memory.clear()
            self.current_task = task_id
            if verbose:
                logger.info(f'🧹 Memory cleared for {task_id}')
        
        for trial in range(self.max_trials):
            memories = self.memory.get_relevant_memories(k=2)
            mem_ctx = '\n'.join([m[:120] for m in memories]) if memories else 'First attempt'
            
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
            
            code = self.llm.call_llm(prompt, max_tokens=2048)

            if isinstance(code, list):
                code = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in code
            )
            
            # 🔥 BULLETPROOF CODE CLEANING (handles LIST + STRING)
            code = self._clean_code(code)
            
            result = self.evaluator.evaluate(code, task['entry_point'], task['test'])
            
            if result['passed']:
                if verbose:
                    logger.info(f'✅ {task_id} solved in {trial+1} trials')
                return {'task_id': task_id, 'success': True, 'trials': trial+1, 'agent_type': 'Smart'}
            
            self.memory.add_reflection(f"Trial {trial+1}: {result['error'][:100]}")
        
        return {'task_id': task_id, 'success': False, 'trials': self.max_trials, 'agent_type': 'Smart'}
    
    def _clean_code(self, raw_code) -> str:
        """Bulletproof code cleaning - handles LIST responses from Gemini."""
        if isinstance(raw_code, list):
            raw_code = raw_code if raw_code else ""
        
        raw_code = str(raw_code).strip()
        
        # Standard markdown cleaning
        if '```python' in raw_code:
            code = raw_code.split('```python')[1].split('```')[0].strip()
        elif '```' in raw_code:
            parts = raw_code.split('```')
            code = parts[1] if len(parts) > 1 else raw_code
            code = code.split('\n')[0].strip() if '\n' in code else code.strip()
        else:
            # Raw code block
            lines = raw_code.split('\n')
            code = '\n'.join(line.strip() for line in lines if line.strip())
        
        return code.strip()
