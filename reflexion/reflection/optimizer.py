"""Optimized reflection generation and scoring."""

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ReflectionOptimizer:
    """
    Generate high-quality, structured reflections from failures.
    
    Filters and scores reflections to ensure only useful feedback
    is stored in memory.
    """
    
    def __init__(self, llm, min_score: float = 0.6):
        """
        Initialize reflection optimizer.
        
        Args:
            llm: BaseLLMModel instance
            min_score: Minimum quality score to keep reflection (0-1)
        """
        self.llm = llm
        self.min_score = min_score
    
    def generate_reflection(self, task_prompt: str, failed_code: str, 
                          error_msg: str, trial_num: int) -> str:
        """
        Generate structured reflection from failure.
        
        Args:
            task_prompt: Original task description
            failed_code: Code that failed
            error_msg: Error message from evaluation
            trial_num: Current trial number
            
        Returns:
            Structured reflection string
        """
        reflection_prompt = f"""You are analyzing a failed code solution. Generate a concise, actionable reflection.

**Task:**
{task_prompt[:300]}...

**Failed Code:**
{failed_code[:500]}...

**Error:**
{error_msg[:200]}

**Instructions:**
1. Identify the specific mistake (1 sentence)
2. Explain why it failed (1 sentence)
3. Suggest specific fix (1 sentence)

Format your response as:
MISTAKE: [what went wrong]
REASON: [why it failed]
FIX: [specific solution]

Your reflection:"""
        
        try:
            reflection = self.llm.call_llm(reflection_prompt, max_tokens=200)
            
            # Structure and clean reflection
            reflection = self._structure_reflection(reflection, trial_num, error_msg)
            
            return reflection
            
        except Exception as e:
            logger.warning(f'Reflection generation failed: {e}')
            # Fallback to simple reflection
            return f"Trial {trial_num}: {error_msg[:100]}"
    
    def _structure_reflection(self, raw_reflection: str, trial_num: int, 
                             error_msg: str) -> str:
        """Structure reflection into consistent format."""
        # Extract structured components if present
        mistake = self._extract_field(raw_reflection, 'MISTAKE')
        reason = self._extract_field(raw_reflection, 'REASON')
        fix = self._extract_field(raw_reflection, 'FIX')
        
        if mistake and reason and fix:
            reflection = f"""Trial {trial_num} Analysis:
- Mistake: {mistake}
- Reason: {reason}
- Fix: {fix}"""
        else:
            # Fallback: use first 3 sentences
            sentences = [s.strip() for s in raw_reflection.split('.') if s.strip()]
            if len(sentences) >= 2:
                reflection = f"Trial {trial_num}: " + '. '.join(sentences[:2]) + '.'
            else:
                reflection = f"Trial {trial_num}: {error_msg[:100]}"
        
        return reflection
    
    def _extract_field(self, text: str, field: str) -> str:
        """Extract field value from structured reflection."""
        pattern = rf'{field}:\s*(.+?)(?:\n|$)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
    
    def score_reflection(self, reflection: str, task_prompt: str) -> float:
        """
        Score reflection quality and relevance.
        
        Args:
            reflection: Reflection text to score
            task_prompt: Original task prompt
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        
        # Check 1: Length (not too short, not too long)
        length = len(reflection.split())
        if 15 <= length <= 100:
            score += 0.3
        elif 10 <= length <= 150:
            score += 0.15
        
        # Check 2: Contains actionable keywords
        actionable_keywords = [
            'fix', 'change', 'add', 'remove', 'modify', 'use', 'check',
            'handle', 'return', 'ensure', 'should', 'need', 'must'
        ]
        found_keywords = sum(1 for kw in actionable_keywords if kw in reflection.lower())
        score += min(found_keywords * 0.1, 0.3)
        
        # Check 3: Contains code-related terms
        code_terms = [
            'function', 'variable', 'loop', 'condition', 'return', 'parameter',
            'edge case', 'test', 'error', 'exception', 'index', 'value'
        ]
        found_terms = sum(1 for term in code_terms if term in reflection.lower())
        score += min(found_terms * 0.08, 0.2)
        
        # Check 4: Structured format
        if 'Trial' in reflection and any(sep in reflection for sep in [':', '-', '•']):
            score += 0.2
        
        return min(score, 1.0)
    
    def filter_reflections(self, reflections: List[str], 
                          task_prompt: str) -> List[str]:
        """
        Filter reflections by quality score.
        
        Args:
            reflections: List of reflection strings
            task_prompt: Task prompt for relevance checking
            
        Returns:
            Filtered list of high-quality reflections
        """
        scored = [(r, self.score_reflection(r, task_prompt)) for r in reflections]
        filtered = [r for r, score in scored if score >= self.min_score]
        
        logger.info(f'Filtered reflections: {len(filtered)}/{len(reflections)} passed (min_score={self.min_score})')
        
        return filtered
