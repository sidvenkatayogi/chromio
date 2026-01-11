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


def extract_hsl_from_string(content: str) -> list:
    """
    Extract HSL colors from string using regex.
    Matches format: (H, S%, L%) where H is 0-360, S and L are 0-100%
    """
    # Pattern to match (H, S%, L%) format
    hsl_pattern = r'\(\s*(\d{1,3})\s*,\s*(\d{1,3})%\s*,\s*(\d{1,3})%\s*\)'
    matches = re.findall(hsl_pattern, content)
    
    hsl_colors = []
    for h, s, l in matches:
        h_int, s_int, l_int = int(h), int(s), int(l)
        # Validate ranges
        if 0 <= h_int < 360 and 0 <= s_int <= 100 and 0 <= l_int <= 100:
            hsl_colors.append(f"({h_int}, {s_int}%, {l_int}%)")
        if len(hsl_colors) >= 5:
            break
    
    return hsl_colors


def hsl_to_hex(hsl_str: str) -> str:
    """
    Convert HSL string format "(H, S%, L%)" to hex color "#RRGGBB".
    """
    # Parse HSL values
    match = re.match(r'\(\s*(\d{1,3})\s*,\s*(\d{1,3})%\s*,\s*(\d{1,3})%\s*\)', hsl_str)
    if not match:
        return "#000000"
    
    h, s, l = int(match.group(1)), int(match.group(2)), int(match.group(3))
    
    # Convert to 0-1 range
    h = h / 360.0
    s = s / 100.0
    l = l / 100.0
    
    # HSL to RGB conversion
    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1/6:
            return p + (q - p) * 6 * t
        if t < 1/2:
            return q
        if t < 2/3:
            return p + (q - p) * (2/3 - t) * 6
        return p
    
    if s == 0:
        r = g = b = l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
    
    # Convert to 0-255 range and format as hex
    r_int = int(round(r * 255))
    g_int = int(round(g * 255))
    b_int = int(round(b * 255))
    
    return f"#{r_int:02x}{g_int:02x}{b_int:02x}"


def extract_lab_from_string(content: str) -> list:
    """
    Extract CIELAB colors from string using regex.
    Matches format: L(L, a, b) where L is 0-100, a and b are -128 to 127
    """
    # Pattern to match L(L, a, b) format
    lab_pattern = r'L\(\s*(\d{1,3})\s*,\s*(-?\d{1,3})\s*,\s*(-?\d{1,3})\s*\)'
    matches = re.findall(lab_pattern, content)
    
    lab_colors = []
    for L, a, b in matches:
        L_int, a_int, b_int = int(L), int(a), int(b)
        # Validate ranges
        if 0 <= L_int <= 100 and -128 <= a_int <= 127 and -128 <= b_int <= 127:
            lab_colors.append(f"L({L_int}, {a_int}, {b_int})")
        if len(lab_colors) >= 5:
            break
    
    return lab_colors


def lab_to_hex(lab_str: str) -> str:
    """
    Convert CIELAB string format "L(L, a, b)" to hex color "#RRGGBB".
    """
    import numpy as np
    from skimage import color as sk_color
    
    # Parse LAB values
    match = re.match(r'L\(\s*(\d{1,3})\s*,\s*(-?\d{1,3})\s*,\s*(-?\d{1,3})\s*\)', lab_str)
    if not match:
        return "#000000"
    
    L, a, b = float(match.group(1)), float(match.group(2)), float(match.group(3))
    
    # Convert LAB to RGB
    lab_array = np.array([[[L, a, b]]])
    rgb_array = sk_color.lab2rgb(lab_array)[0][0]
    
    # Convert to 0-255 range and format as hex
    r_int = int(round(np.clip(rgb_array[0] * 255, 0, 255)))
    g_int = int(round(np.clip(rgb_array[1] * 255, 0, 255)))
    b_int = int(round(np.clip(rgb_array[2] * 255, 0, 255)))
    
    return f"#{r_int:02x}{g_int:02x}{b_int:02x}"


class ModelClient:
    
    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES, 
                 retry_delay: float = DEFAULT_RETRY_DELAY,
                 retry_backoff: float = DEFAULT_RETRY_BACKOFF,
                 request_delay: float = DEFAULT_REQUEST_DELAY,
                 color_format: str = 'hex'):
        load_dotenv()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.request_delay = request_delay
        self._last_request_time = 0
        self.color_format = color_format.lower()
    
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
            
            # Extract colors based on format
            if self.color_format == 'hsl':
                palette_hsl = extract_hsl_from_string(content)
                # Convert HSL to hex for evaluation
                palette_hex = [hsl_to_hex(hsl) for hsl in palette_hsl]
            elif self.color_format == 'cielab' or self.color_format == 'lab':
                palette_lab = extract_lab_from_string(content)
                # Convert LAB to hex for evaluation
                palette_hex = [lab_to_hex(lab) for lab in palette_lab]
            else:
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
