"""Reflexion agent implementation."""

import logging
from typing import Dict
from ..evaluators import ObjectiveCodeEvaluator
from ..memory import TemporalMemory, VectorEpisodicMemory

logger = logging.getLogger(__name__)


class ReflexionAgent:
    """
    Reflexion agent for iterative code generation with self-reflection.
    
    The agent attempts to solve tasks through multiple trials,
    learning from failures via verbal reflections stored in memory.
    """
    
    def __init__(self, llm, memory_mode: str = 'temporal', max_trials: int = 3):
        """
        Initialize Reflexion agent.
        
        Args:
            llm: BaseLLMModel instance for code generation
            memory_mode: 'temporal' or 'vector' for memory type
            max_trials: Maximum number of attempts per task
        """
        self.llm = llm
        self.max_trials = max_trials
        self.evaluator = ObjectiveCodeEvaluator(timeout=10)
        
        # Initialize memory based on mode
        if memory_mode == 'temporal':
            self.memory = TemporalMemory()
        else:
            self.memory = VectorEpisodicMemory(llm)
        
        self.memory_mode = memory_mode
    
    def solve_task(self, task: Dict, verbose: bool = False) -> Dict:
        """
        Attempt to solve a task using iterative refinement.
        
        Args:
            task: Dictionary with keys: task_id, prompt, entry_point, test
            verbose: Whether to log detailed information
            
        Returns:
            Dictionary with keys:
            - task_id: Task identifier
            - success: Whether task was solved
            - trials: Number of trials used
            - code: Final generated code (if successful)
        """
        task_id = task['task_id']
        
        for trial in range(self.max_trials):
            # Retrieve relevant memories
            if self.memory_mode == 'temporal':
                memories = self.memory.get_relevant_memories(k=3)
            else:
                memories = self.memory.get_relevant_memories(task['prompt'], k=3)
            
            # Build context from memories
            mem_ctx = '\n'.join(f'- {m}' for m in memories) if memories else 'None'
            
            # Create prompt for LLM
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
                
                # Generate code
                code = self.llm.call_llm(prompt, max_tokens=2048)

                if isinstance(code, list):          # handle list first
                    code = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in code
                    )
                
                # Clean markdown formatting if present
                if '```python' in code:
                    code = code.split('```python')[1].split('```')[0].strip()
                elif '```' in code:
                    code = code.split('```')[1].split('```')[0].strip()

                if isinstance(code, list):
                    code = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in code
                )
                
                # Evaluate code
                results = self.evaluator.evaluate(code, task['entry_point'], task['test'])
                
                if results['passed']:
                    logger.info(f'✅ {task_id} solved in {trial+1} trials')
                    return {
                        'task_id': task_id,
                        'success': True,
                        'trials': trial+1,
                        'code': code
                    }
                
                # Generate reflection for failure
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
        return {
            'task_id': task_id,
            'success': False,
            'trials': self.max_trials
        }
    
    def reset(self):
        """Clear agent memory."""
        self.memory.clear()