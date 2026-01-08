"""
Model client implementations for OpenAI and Fireworks AI.
"""

import json
import time
import re
import openai
from fireworks import Fireworks
from dotenv import load_dotenv

from .config import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


# Rate limiting configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_RETRY_BACKOFF = 2.0  # exponential backoff multiplier
DEFAULT_REQUEST_DELAY = 0.5  # delay between requests (seconds)


class ModelClient:
    
    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES, 
                 retry_delay: float = DEFAULT_RETRY_DELAY,
                 retry_backoff: float = DEFAULT_RETRY_BACKOFF,
                 request_delay: float = DEFAULT_REQUEST_DELAY):
        load_dotenv()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.request_delay = request_delay
        self._last_request_time = 0
    
    def _extract_palette_text(self, data: dict) -> list:
        """
        Recursively extract palette_text from potentially nested JSON.
        """
        if isinstance(data, dict):
            if 'palette_text' in data:
                return data['palette_text']
            
            for key, value in data.items():
                if isinstance(value, dict):
                    result = self._extract_palette_text(value)
                    if result:
                        return result
        return []
    
    def _extract_hex_from_string(self, content: str) -> list:
        """
        Extract hex colors directly from string using regex.
        """
        hex_pattern = r'#[0-9a-fA-F]{6}'
        matches = re.findall(hex_pattern, content)
        # Return first 5 unique colors
        seen = []
        for m in matches:
            if m.lower() not in [s.lower() for s in seen]:
                seen.append(m)
            if len(seen) >= 5:
                break
        return seen
    
    def _wait_for_rate_limit(self):
        """Wait if needed to respect rate limits."""
        if self.request_delay > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()
    
    def _call_with_retry(self, api_call_func):
        """Execute API call with retry logic and exponential backoff."""
        last_exception = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                return api_call_func()
            except (openai.RateLimitError, openai.APITimeoutError) as e:
                last_exception = e
                print(f"Rate limit/timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= self.retry_backoff
            except openai.APIError as e:
                print(f"API Error: {e}")
                return {}
            except Exception as e:
                print(f"API Error: {e}")
                last_exception = e
                print(f"Rate limit/timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= self.retry_backoff
        
        print(f"Max retries ({self.max_retries}) exceeded. Last error: {last_exception}")
        return {}
    
    def generate_palette(self, query: str, examples: str) -> dict:
        user_message = USER_PROMPT_TEMPLATE.format(query=query, examples=examples)
        
        def api_call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            palette_hex = self._extract_hex_from_string(content)
            
            try:
                content_json = json.loads(content)
                palette_text = self._extract_palette_text(content_json)
            except (json.JSONDecodeError, TypeError):
                palette_text = []
                
            return palette_hex, palette_text, content
        
        return self._call_with_retry(api_call)


class OpenAIClient(ModelClient):
    """OpenAI API client."""
    
    def __init__(self, model: str = "gpt-4o-mini", **kwargs):
        super().__init__(**kwargs)
        self.client = openai.OpenAI()
        self.model = model


class FireworksClient(ModelClient):
    """Fireworks AI API client."""
    
    def __init__(self, model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct", **kwargs):
        super().__init__(**kwargs)
        self.client = Fireworks()
        self.model = model
    
    


def get_model_client(provider: str, model: str = None, **kwargs):
    """
    Factory function to get the appropriate model client.
    
    Args:
        provider: Model provider ('openai' or 'fireworks')
        model: Model name (optional, uses default for provider)
        **kwargs: Rate limiting options (max_retries, retry_delay, request_delay, etc.)
    """
    from .config import PROVIDER_OPENAI, PROVIDER_FIREWORKS, DEFAULT_MODELS
    
    if provider == PROVIDER_OPENAI:
        model = model or DEFAULT_MODELS[PROVIDER_OPENAI]
        return OpenAIClient(model=model, **kwargs)
    elif provider == PROVIDER_FIREWORKS:
        model = model or DEFAULT_MODELS[PROVIDER_FIREWORKS]
        return FireworksClient(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
