"""LLM interface with retry logic and embedding support."""

import time
import random
import logging
import numpy as np
from functools import wraps
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def exponential_backoff(max_retries=5, initial_delay=5.0):
    """
    Decorator for exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
    """
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


class BaseLLMModel:
    """
    Base LLM interface with retry logic and embedding support.
    
    Attributes:
        api_key: OpenRouter API key
        model: Model identifier
        api_base: API base URL
        rate_limit_delay: Delay between API calls in seconds
    """
    
    def __init__(self, api_key: str, model: str = 'google/gemini-2.0-flash-exp:free', 
                 api_base: str = 'https://openrouter.ai/api/v1/', 
                 rate_limit_delay: float = 30.0):
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
        
        # Embedding model (lazy loaded)
        self._embed_model: Optional[SentenceTransformer] = None
    
    def _wait(self):
        """Implement rate limiting between API calls."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.rate_limit_delay:
            wait = self.rate_limit_delay - elapsed
            logger.info(f'⏳ Rate protection: {wait:.1f}s')
            time.sleep(wait)
        self.last_call_time = time.time()
    
    @exponential_backoff(max_retries=5, initial_delay=5.0)
    def call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Call LLM API with retry logic.
        
        Args:
            prompt: Input prompt for the LLM
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text from LLM
        """
        self._wait()
        response = self.session.post(
            f"{self.api_base}chat/completions",
            headers=self.headers,
            json={
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': 0.7
            },
            timeout=120
        )
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        
        # Gemini via OpenRouter occasionally returns content as a list of parts
        # e.g. [{"type": "text", "text": "..."}] instead of a plain string
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        
        return content
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding for text using Sentence-BERT.
        
        Args:
            text: Input text to embed
            
        Returns:
            384-dimensional embedding vector
        """
        if self._embed_model is None:
            logger.info('📦 Loading embedding model (one-time, ~90MB)...')
            self._embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info('✓ Embedding model ready')
        
        return self._embed_model.encode(text, show_progress_bar=False)
