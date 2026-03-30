"""Code evaluation using subprocess execution."""

import os
import sys
import ast
import tempfile
import subprocess
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ObjectiveCodeEvaluator:
    """
    Evaluate Python code using HumanEval test format.
    
    Executes code in isolated subprocess with timeout protection.
    """
    
    def __init__(self, timeout: int = 10):
        """
        Initialize evaluator.
        
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
    
    def evaluate(self, code: str, entry_point: str, test_code: str) -> Dict:
        """
        Evaluate code using HumanEval's test format.
        
        Args:
            code: Solution code to evaluate
            entry_point: Function name to test
            test_code: Test code from HumanEval dataset
            
        Returns:
            Dictionary with 'passed' (bool) and 'error' (str or None)
        """
        # Check syntax first
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {'passed': False, 'error': f'SyntaxError: {e}'}
        
        # Execute in subprocess
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
