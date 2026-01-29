import openai
from typing import Any, Dict, Optional
import json
import time
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer.semantic import SemanticAnalyzer
from analyzer.model_rec import ModelRecommender
from runtime.cache import ResponseCache


class APIOptimizerWrapper:
    
    def __init__(self, enable_caching: bool = True, enable_model_downgrade: bool = True, 
                 similarity_threshold: float = 0.90, auto_apply_optimizations: bool = False, 
                 verbose: bool = True, cache_ttl_minutes: int = 60):
    
        self.enable_caching = enable_caching
        self.enable_model_downgrade = enable_model_downgrade
        self.auto_apply = auto_apply_optimizations
        self.verbose = verbose
        self.similarity_threshold = similarity_threshold
        
        if enable_caching:
            self.cache = ResponseCache(ttl_minutes=cache_ttl_minutes)
            self.semantic_analyzer = SemanticAnalyzer(similarity_threshold=similarity_threshold)
        else:
            self.cache = None
            self.semantic_analyzer = None
            
        if enable_model_downgrade:
            self.model_recommender = ModelRecommender()
        else:
            self.model_recommender = None
        
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'model_downgrades': 0,
            'total_cost_saved': 0.0
        }
        
        self.original_openai_class = None
        self._is_wrapped = False
    
    def wrap(self):
        """Monkey-patch OpenAI to use our wrapper"""
        if self._is_wrapped:
            print("⚠️ Already wrapped!")
            return
        
        import openai
        original_openai_class = openai.OpenAI
        
        wrapper_self = self
        
        class WrappedOpenAI(original_openai_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                original_create = self.chat.completions.create
                
                def optimized_create(**call_kwargs):
                    return wrapper_self._optimized_call(original_create, **call_kwargs)
                
                self.chat.completions.create = optimized_create
        
        openai.OpenAI = WrappedOpenAI
        self.original_openai_class = original_openai_class
        self._is_wrapped = True
        
        print(" APIOptimizer wrapper activated!")
        print(f"   - Caching: {'ON' if self.enable_caching else 'OFF'}")
        print(f"   - Model optimization: {'ON' if self.enable_model_downgrade else 'OFF'}")
        print(f"   - Auto-apply: {'ON' if self.auto_apply else 'OFF'}")
    
    def unwrap(self):
        """Remove the monkey-patch"""
        if not self._is_wrapped:
            return
        
        import openai
        openai.OpenAI = self.original_openai_class
        self._is_wrapped = False
        print(" APIOptimizer wrapper deactivated")
    
    def _optimized_call(self, original_create, **kwargs) -> Any:
        """Main optimization logic"""
        self.stats['total_calls'] += 1

        messages = kwargs.get('messages', [])
        current_model = kwargs.get('model', 'gpt-3.5-turbo')

        prompt = self._extract_prompt(messages)

        cache_kwargs = dict(kwargs)
        cache_kwargs.pop('model', None)

        # 1. CHECK CACHE
        if self.enable_caching and self.cache and self.semantic_analyzer:
            cached = self._check_cache(prompt, current_model, **cache_kwargs)
            if cached:
                self.stats['cache_hits'] += 1
                if self.verbose:
                    cost_saved = self._estimate_call_cost(current_model, prompt)
                    print(f"💾 Cache hit! Saved ${cost_saved:.4f}")
                return cached

        # 2. CHECK MODEL OPTIMIZATION
        if self.enable_model_downgrade and self.model_recommender:
            recommendation = self.model_recommender.recommend_model(current_model, prompt)

            if recommendation:
                savings = recommendation['cost_savings_per_1m_tokens']
                confidence = recommendation['confidence']
                recommended_model = recommendation['recommended_model']

                if self.verbose:
                    print(f"\n💡 Model Optimization Available:")
                    print(f"   Current: {current_model}")
                    print(f"   Recommended: {recommended_model}")
                    print(f"   Reason: {recommendation['reason']}")
                    print(f"   Confidence: {confidence:.0%}")
                    print(f"   Savings: ${savings['input']:.2f} input / ${savings['output']:.2f} output per 1M tokens")

                if self.auto_apply and confidence >= 0.80:
                    if self.verbose:
                        print(f"   ✓ Auto-applying optimization...")
                    kwargs['model'] = recommended_model
                    self.stats['model_downgrades'] += 1

                    input_tokens = self.model_recommender.estimate_tokens(prompt, model=current_model)
                    output_tokens = kwargs.get('max_tokens', 150)  # rough estimate

                    old_cost = self.model_recommender.estimate_cost(current_model, input_tokens, output_tokens)
                    new_cost = self.model_recommender.estimate_cost(recommended_model, input_tokens, output_tokens)
                    self.stats['total_cost_saved'] += (old_cost - new_cost)
                else:
                    if self.verbose:
                        print(f"     ℹ️ Set auto_apply=True to use automatically")

        # 3. MAKE ACTUAL API CALL
        if self.verbose:
            print(f"\n🔄 Calling {kwargs.get('model', current_model)} API...")
        start_time = time.time()

        response = original_create(**kwargs)

        elapsed = time.time() - start_time
        if self.verbose:
            print(f" Response received in {elapsed:.2f}s")

        # 4. CACHE RESPONSE
        if self.enable_caching and self.cache:
            self._cache_response(prompt, response, current_model, **cache_kwargs)

        return response

       
    
    def _extract_prompt(self, messages: list) -> str:
        """Extract the user prompt from messages"""
        if not messages:
            return ""
        
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                return msg.get('content', '')
        
        return ""
    
    def _check_cache(self, prompt: str, model: str, **kwargs) -> Optional[Any]:
        """Check cache for exact or semantically similar prompts"""
        if not self.cache:
            return None
        
        cached = self.cache.get(prompt, model, **kwargs)
        if cached:
            if self.verbose:
                print("💾 Exact cache hit!")
            return cached
        
        if not self.semantic_analyzer:
            return None
        
        cached_prompts = [entry['prompt'] for entry in self.cache.cache.values()]
        
        if not cached_prompts:
            return None
        
        similar = self.semantic_analyzer.find_most_similar(
            prompt, 
            cached_prompts,
            threshold=self.similarity_threshold
        )
        
        if similar:
            similar_prompt, similarity = similar
            if self.verbose:
                print(f" Semantic cache hit! ({similarity:.0%} similar)")
                print(f"   Original: \"{similar_prompt[:60]}...\"")
                print(f"   Current:  \"{prompt[:60]}...\"")
            
            return self.cache.get(similar_prompt, model, **kwargs)
        
        return None
    
    def _cache_response(self, prompt: str, response: Any, model: str, **kwargs):
        if self.cache:
            self.cache.set(prompt, model, response, **kwargs)
    
    def _estimate_call_cost(self, model: str, prompt: str) -> float:
        if not self.model_recommender:
            return 0.0
        
        input_tokens = self.model_recommender.estimate_tokens(prompt, model=model)
        output_tokens = 150  # rough average
        return self.model_recommender.estimate_cost(model, input_tokens, output_tokens)
    
    def save_cache(self, filepath: str = "api_cache.pkl"):
        if self.cache:
            self.cache.save(filepath)
            if self.verbose:
                print(f"💾 Cache saved to {filepath}")
    
    def load_cache(self, filepath: str = "api_cache.pkl"):
        if self.cache:
            self.cache.load(filepath)
            if self.verbose:
                print(f"💾 Cache loaded from {filepath}")
    
    def clear_cache(self):
        if self.cache:
            self.cache.clear()
            if self.verbose:
                print("🗑️ Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        if self.cache:
            return self.cache.stats()
        return {'total_entries': 0, 'valid_entries': 0, 'expired_entries': 0}
    
    def print_stats(self):
        print("\n" + "="*60)
        print(" API OPTIMIZER STATISTICS")
        print("="*60)
        print(f"Total API calls: {self.stats['total_calls']}")
        print(f"Cache hits: {self.stats['cache_hits']}")
        
        if self.stats['total_calls'] > 0:
            hit_rate = (self.stats['cache_hits'] / self.stats['total_calls']) * 100
            print(f"Cache hit rate: {hit_rate:.1f}%")
        
        print(f"Model downgrades: {self.stats['model_downgrades']}")
        print(f"Estimated cost saved: ${self.stats['total_cost_saved']:.4f}")
        
        # Cache stats
        if self.cache:
            cache_stats = self.get_cache_stats()
            print(f"\nCache entries: {cache_stats['valid_entries']} valid, {cache_stats['expired_entries']} expired")
        
        print("="*60)


class APIOptimizer:
    """Convenience class for easy enable/disable"""
    _instance = None
    
    @staticmethod
    def enable(auto_apply: bool = False, similarity_threshold: float = 0.90, cache_ttl_minutes: int = 60):
        """Enable the API optimizer"""
        wrapper = APIOptimizerWrapper(
            enable_caching=True,
            enable_model_downgrade=True,
            similarity_threshold=similarity_threshold,
            auto_apply_optimizations=auto_apply,
            verbose=True,
            cache_ttl_minutes=cache_ttl_minutes
        )
        wrapper.wrap()
        APIOptimizer._instance = wrapper
        return wrapper
    
    @staticmethod
    def disable():
        """Disable the API optimizer"""
        if APIOptimizer._instance:
            APIOptimizer._instance.unwrap()
            APIOptimizer._instance = None
    
    @staticmethod
    def get_instance() -> Optional[APIOptimizerWrapper]:
        """Get the current wrapper instance"""
        return APIOptimizer._instance
