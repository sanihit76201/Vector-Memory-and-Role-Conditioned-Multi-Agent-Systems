"""HumanEval benchmark loader."""

import os
import json
import gzip
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class HumanEvalLoader:
    """Load tasks from HumanEval dataset."""
    
    @staticmethod
    def load_from_file(file_path: str = 'HumanEval.jsonl.gz', 
                       num_samples: int = 3) -> List[Dict]:
        """
        Load REAL HumanEval tasks.
        
        Args:
            file_path: Path to HumanEval.jsonl.gz file
            num_samples: Number of tasks to load
            
        Returns:
            List of task dictionaries with keys:
            - task_id: Task identifier (e.g., 'HumanEval/0')
            - prompt: Function signature and docstring
            - entry_point: Function name to test
            - canonical_solution: Reference solution
            - test: Test code to run
            
        Raises:
            FileNotFoundError: If HumanEval file not found
        """
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