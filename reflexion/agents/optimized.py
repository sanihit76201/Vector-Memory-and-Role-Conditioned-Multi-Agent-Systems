"""Enhanced Reflexion agent with optimized reflections."""

import logging
from typing import Dict
from ..evaluators import ObjectiveCodeEvaluator
from ..memory import TemporalMemory, VectorEpisodicMemory
from ..reflection.optimizer import ReflectionOptimizer

logger = logging.getLogger(__name__)


class OptimizedReflexionAgent:
    """
    Enhanced Reflexion agent with optimized reflection generation.
    
    Key improvements over base agent:
    1. Task-isolated memory (cleared between tasks)
    2. Structured reflection generation
    3. Reflection quality filtering
    4. Better prompt engineering
    """
    
    def __init__(self, llm, memory_mode: str = 'temporal', 
                 max_trials: int = 3, optimize_reflections: bool = True):
        """
        Initialize optimized Reflexion agent.
        
        Args:
            llm: BaseLLMModel instance
            memory_mode: 'temporal' or 'vector'
            max_trials: Maximum attempts per task
            optimize_reflections: Whether to use reflection optimizer
        """
        self.llm = llm
        self.max_trials = max_trials
        self.evaluator = ObjectiveCodeEvaluator(timeout=10)
        self.optimize_reflections = optimize_reflections
        
        # Initialize memory
        if memory_mode == 'temporal':
            self.memory = TemporalMemory(max_size=5)
        else:
            self.memory = VectorEpisodicMemory(llm, max_size=10)
        
        self.memory_mode = memory_mode
        
        # Initialize reflection optimizer
        if optimize_reflections:
            self.optimizer = ReflectionOptimizer(llm, min_score=0.6)
        else:
            self.optimizer = None
    
    def solve_task(self, task: Dict, verbose: bool = False) -> Dict:
        """
        Solve task with optimized reflections.
        
        Key difference: Memory is CLEARED at start of each task
        to prevent cross-task contamination.
        
        Args:
            task: Task dictionary
            verbose: Enable detailed logging
            
        Returns:
            Result dictionary with success status
        """
        task_id = task['task_id']
        
        # 🔑 KEY IMPROVEMENT: Clear memory for each new task
        self.memory.clear()
        logger.info(f'🧹 Memory cleared for {task_id}')
        
        for trial in range(self.max_trials):
            # Retrieve relevant memories (task-specific now!)
            if self.memory_mode == 'temporal':
                memories = self.memory.get_relevant_memories(k=3)
            else:
                memories = self.memory.get_relevant_memories(task['prompt'], k=3)
            
            # Build prompt with filtered memories
            if memories and self.optimizer:
                # Filter reflections by quality
                memories = self.optimizer.filter_reflections(memories, task['prompt'])
            
            mem_ctx = '\n'.join(f'- {m}' for m in memories) if memories else 'None (first attempt)'
            
            # Enhanced prompt with better structure
            prompt = f"""You are an expert Python programmer. Complete this function carefully.

TASK:
{task['prompt']}

PREVIOUS ATTEMPTS & LESSONS:
{mem_ctx}

REQUIREMENTS:
1. Study the previous failures to avoid repeating mistakes
2. Handle ALL edge cases (empty input, None, negative numbers, etc.)
3. Ensure correct return types
4. Test logic carefully before submitting

OUTPUT ONLY PYTHON CODE (no markdown, no explanations):"""
            
            try:
                logger.info(f'🔄 Trial {trial+1}/{self.max_trials} - {task_id}')
                
                # Generate code
                code = self.llm.call_llm(prompt, max_tokens=2048)
                
                # Clean formatting
                if '```python' in code:
                    code = code.split('```python').split('```').strip()[1]
                elif '```' in code:
                    code = code.split('```')[1].split('```')[0].strip()
                
                # Evaluate
                results = self.evaluator.evaluate(code, task['entry_point'], task['test'])
                
                if results['passed']:
                    logger.info(f'✅ {task_id} solved in {trial+1} trials')
                    return {
                        'task_id': task_id,
                        'success': True,
                        'trials': trial+1,
                        'code': code,
                        'optimized': self.optimize_reflections
                    }
                
                # Generate optimized reflection
                if self.optimizer:
                    reflection = self.optimizer.generate_reflection(
                        task['prompt'],
                        code,
                        results['error'],
                        trial+1
                    )
                    
                    # Score and only add if high quality
                    score = self.optimizer.score_reflection(reflection, task['prompt'])
                    if score >= self.optimizer.min_score:
                        self.memory.add_reflection(reflection)
                        logger.info(f'📝 Reflection added (score: {score:.2f})')
                    else:
                        logger.warning(f'⚠️  Reflection rejected (score: {score:.2f})')
                else:
                    # Simple reflection
                    reflection = f"Trial {trial+1} failed: {results['error'][:100]}"
                    self.memory.add_reflection(reflection)
                
                logger.warning(f'❌ Trial {trial+1} failed: {results["error"][:100]}...')
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f'❌ Trial {trial+1} error: {str(e)[:100]}')
                if self.optimizer:
                    refl = self.optimizer.generate_reflection(
                        task['prompt'], '', str(e), trial+1
                    )
                    self.memory.add_reflection(refl)
        
        logger.warning(f'❌ {task_id} failed after {self.max_trials} trials')
        return {
            'task_id': task_id,
            'success': False,
            'trials': self.max_trials,
            'optimized': self.optimize_reflections
        }
    
    def reset(self):
        """Clear agent memory."""
        self.memory.clear()
