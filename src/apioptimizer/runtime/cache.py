import json
import hashlib
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import pickle

class ResponseCache:
    
    def __init__(self, ttl_minutes: int = 60):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _make_key(self, prompt: str, model: str, **kwargs) -> str:
        cache_data = {
            'prompt': prompt,
            'model': model,
            'temperature': kwargs.get('temperature', 1.0),
            'max_tokens': kwargs.get('max_tokens')
        }
        
        key_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, **kwargs) -> Optional[Any]:
        key = self._make_key(prompt, model, **kwargs)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check TTL
            if datetime.now() - entry['timestamp'] < self.ttl:
                return entry['response']
            else:
                # Expired
                del self.cache[key]
        
        return None
    
    def set(self, prompt: str, model: str, response: Any, **kwargs):
        key = self._make_key(prompt, model, **kwargs)
        
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.now(),
            'prompt': prompt,
            'model': model
        }
    
    def clear(self):
        self.cache.clear()
    
    def stats(self) -> Dict:
        total = len(self.cache)
        expired = sum(
            1 for entry in self.cache.values()
            if datetime.now() - entry['timestamp'] >= self.ttl
        )
        
        return {
            'total_entries': total,
            'valid_entries': total - expired,
            'expired_entries': expired
        }
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def load(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                self.cache = pickle.load(f)
        except FileNotFoundError:
            pass