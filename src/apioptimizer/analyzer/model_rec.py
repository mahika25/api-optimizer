from typing import List, Dict, Optional
import re

class ModelRecommender:
    SIMPLE_TASKS = [
        'sentiment', 'classify', 'category', 'tag', 'label',
        'yes or no', 'true or false', 'extract', 'parse',
        'summarize', 'title', 'headline', 'caption',
        'translate', 'detect language', 'count', 'list',
        'is this', 'does this', 'spam', 'valid'
    ]
    
    COMPLEX_TASKS = [
        'explain in detail', 'analyze deeply', 'reason about', 'prove',
        'write code', 'debug', 'refactor', 'design',
        'architect', 'optimize', 'implement', 'create algorithm',
        'comprehensive', 'detailed analysis', 'compare and contrast',
        'evaluate', 'critique', 'argue', 'philosophical'
    ]
    
    # Model pricing (per 1M tokens) - input/output separated
    PRICING = {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }
    
    def __init__(self):
        try:
            self.encoders = {
                "gpt-4": tiktoken.encoding_for_model("gpt-4"),
                "gpt-4-turbo": tiktoken.encoding_for_model("gpt-4-turbo"),
                "gpt-4o": tiktoken.encoding_for_model("gpt-4o"),
                "gpt-4o-mini": tiktoken.encoding_for_model("gpt-4o-mini"),
                "claude-3-opus": tiktoken.get_encoding("cl100k_base"),
                "claude-3-sonnet": tiktoken.get_encoding("cl100k_base"),
                "claude-3-5-sonnet": tiktoken.get_encoding("cl100k_base"),
                "claude-3-haiku": tiktoken.get_encoding("cl100k_base"),
            }
        except Exception:
            self.encoders = {}  # fallback to simple word estimate
    
    def estimate_tokens(self, text: str, model: str = 'gpt-3.5-turbo') -> int:
        openai_models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini']
        
        if model in openai_models:
            try:
                import tiktoken
                try:
                    enc = tiktoken.encoding_for_model(model)
                except KeyError:
                    enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text))
            except ImportError:
                # fallback if tiktoken isn't installed
                return int(len(text.split()) * 1.3)
        
        # Fallback for non-OpenAI models like Claude
        return int(len(text.split()) * 1.3)
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if model not in self.PRICING:
            return 0.0
        
        pricing = self.PRICING[model]
        return ((input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"])
    
    def estimate_response_tokens(self, prompt: str) -> int:
        prompt_tokens = self.estimate_tokens(prompt)
        prompt_lower = prompt.lower()
        
        if any(keyword in prompt_lower for keyword in ['yes or no', 'true or false', 'one word']):
            return 10
        
        if any(keyword in prompt_lower for keyword in ['extract', 'find', 'list']):
            return int(prompt_tokens * 0.5)
        
        if 'summarize' in prompt_lower or 'summary' in prompt_lower:
            return int(prompt_tokens * 0.3)
        
        if any(keyword in prompt_lower for keyword in ['classify', 'sentiment', 'category']):
            return 20
        
        if any(keyword in prompt_lower for keyword in ['write code', 'function', 'class', 'implement']):
            return max(200, prompt_tokens * 2)
        
        if any(keyword in prompt_lower for keyword in ['explain', 'analyze', 'describe']):
            return int(prompt_tokens * 1.5)
        
        return prompt_tokens
    
    
    def estimate_response_tokens(self, prompt: str) -> int:
        prompt_tokens = self.estimate_tokens(prompt)
        prompt_lower = prompt.lower()
        
        if any(keyword in prompt_lower for keyword in ['yes or no', 'true or false', 'one word']):
            return 10
        if any(keyword in prompt_lower for keyword in ['extract', 'find', 'list']):
            return int(prompt_tokens * 0.5)
        if 'summarize' in prompt_lower or 'summary' in prompt_lower:
            return int(prompt_tokens * 0.3)
        if any(keyword in prompt_lower for keyword in ['classify', 'sentiment', 'category']):
            return 20
        if any(keyword in prompt_lower for keyword in ['write code', 'function', 'class', 'implement']):
            return max(200, prompt_tokens * 2)
        if any(keyword in prompt_lower for keyword in ['explain', 'analyze', 'describe']):
            return int(prompt_tokens * 1.5)
        
        return prompt_tokens
    
    def analyze_task_complexity(self, prompt: str) -> str:
        keyword_result = self.keyword_analysis(prompt)
        return keyword_result['complexity']
    
    def keyword_analysis(self, prompt: str) -> Dict:
        prompt_lower = prompt.lower()
        simple_score = sum(1 for kw in self.SIMPLE_TASKS if kw in prompt_lower)
        complex_score = sum(1 for kw in self.COMPLEX_TASKS if kw in prompt_lower)
        word_count = len(prompt.split())
        
        has_code = '```' in prompt or 'def ' in prompt or 'class ' in prompt_lower
        asks_for_short = any(phrase in prompt_lower for phrase in ['one word', 'yes or no', 'true or false', 'in one sentence', 'briefly'])
        asks_for_long = any(phrase in prompt_lower for phrase in ['detailed', 'comprehensive', 'in depth', 'thoroughly', 'explain everything'])
        
        if simple_score >= 2 or (simple_score >= 1 and asks_for_short and word_count < 50):
            return {'complexity': 'simple', 'confidence': 0.9}
        if complex_score >= 2 or has_code or (complex_score >= 1 and word_count > 200):
            return {'complexity': 'complex', 'confidence': 0.9}
        if simple_score >= 1 and word_count < 50:
            return {'complexity': 'simple', 'confidence': 0.7}
        if complex_score >= 1 or asks_for_long:
            return {'complexity': 'complex', 'confidence': 0.7}
        if word_count < 15 and '?' in prompt:
            return {'complexity': 'simple', 'confidence': 0.5}
        if word_count > 300:
            return {'complexity': 'complex', 'confidence': 0.6}
        if word_count > 150:
            return {'complexity': 'medium', 'confidence': 0.6}
        if word_count < 50:
            return {'complexity': 'simple', 'confidence': 0.3}
        return {'complexity': 'medium', 'confidence': 0.3}
    
    def recommend_model(self, current_model: str, prompt: str, temperature: float = 1.0) -> Optional[Dict]:
        complexity = self.analyze_task_complexity(prompt)
        recommendations = {
            'gpt-4': {
                'simple': ('gpt-4o-mini', 0.90),
                'medium': ('gpt-4o', 0.70),
                'complex': (None, 0.0)
            },
            'gpt-4-turbo': {
                'simple': ('gpt-4o-mini', 0.85),
                'medium': ('gpt-4o', 0.75),
                'complex': (None, 0.0)
            },
            'gpt-4o': {
                'simple': ('gpt-4o-mini', 0.80),
                'medium': (None, 0.0),
                'complex': (None, 0.0)
            },
            'claude-3-opus': {
                'simple': ('claude-3-haiku', 0.90),
                'medium': ('claude-3-sonnet', 0.80),
                'complex': (None, 0.0)
            },
            'claude-3-sonnet': {
                'simple': ('claude-3-haiku', 0.85),
                'medium': (None, 0.0),
                'complex': (None, 0.0)
            },
            'claude-3-5-sonnet': {
                'simple': ('claude-3-haiku', 0.85),
                'medium': (None, 0.0),
                'complex': (None, 0.0)
            }
        }
        
        if current_model not in recommendations:
            return None
        
        recommended, confidence = recommendations[current_model].get(complexity, (None, 0.0))
        if not recommended or current_model not in self.PRICING or recommended not in self.PRICING:
            return None
        
        current_pricing = self.PRICING[current_model]
        recommended_pricing = self.PRICING[recommended]
        input_savings = current_pricing["input"] - recommended_pricing["input"]
        output_savings = current_pricing["output"] - recommended_pricing["output"]
        if input_savings <= 0 and output_savings <= 0:
            return None
        
        reasons = {
            'simple': f'Task appears simple ({self.extract_task_type(prompt)})',
            'medium': 'Task has medium complexity',
            'complex': 'Complex task requires powerful model'
        }
        
        return {
            'current_model': current_model,
            'recommended_model': recommended,
            'reason': reasons[complexity],
            'cost_savings_per_1m_tokens': {
                'input': input_savings,
                'output': output_savings
            },
            'confidence': confidence,
            'task_complexity': complexity
        }
    
    def extract_task_type(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        for task in self.SIMPLE_TASKS:
            if task in prompt_lower:
                return task
        return 'general task'
    
    def batch_analyze(self, calls: List[Dict]) -> Dict:
        recommendations = []
        total_current = 0.0
        total_recommended = 0.0
        
        for call in calls:
            model = call.get('model', 'unknown')
            prompt = call.get('prompt', '')
            
            rec = self.recommend_model(model, prompt)
            if rec:
                input_tokens = self.estimate_tokens(prompt)
                output_tokens = self.estimate_response_tokens(prompt)
                
                current_cost = self.estimate_cost(model, input_tokens, output_tokens)
                recommended_cost = self.estimate_cost(rec['recommended_model'], input_tokens, output_tokens)
                savings = current_cost - recommended_cost
                savings_percent = (savings / current_cost * 100) if current_cost > 0 else 0
                
                rec.update({
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'current_cost': current_cost,
                    'recommended_cost': recommended_cost,
                    'savings': savings,
                    'savings_percent': savings_percent
                })
                
                recommendations.append({'original': call, 'recommendation': rec})
                total_current += current_cost
                total_recommended += recommended_cost
        
        total_savings = total_current - total_recommended
        savings_percent = (total_savings / total_current * 100) if total_current > 0 else 0
        
        return {
            'recommendations': recommendations,
            'total_current_cost': total_current,
            'total_recommended_cost': total_recommended,
            'total_savings': total_savings,
            'savings_percent': savings_percent,
            'count': len(recommendations)
        }