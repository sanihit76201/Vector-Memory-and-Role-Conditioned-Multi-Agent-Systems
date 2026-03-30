"""Configuration management for Reflexion framework."""

import os
import logging
from dotenv import load_dotenv
from typing import Dict

logger = logging.getLogger(__name__)


class SecureConfigLoader:
    """Load and validate configuration from .env file."""
    
    def __init__(self):
        self.config = {}
    
    def load_from_env_file(self, env_path: str = '.env') -> Dict[str, str]:
        """Load configuration from .env file."""
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
