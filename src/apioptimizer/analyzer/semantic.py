from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class PromptCluster:
    representative: str
    prompts: List[str]
    similarity_scores: List[float] 
    count: int
    avg_similarity: float

class SemanticAnalyzer:

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded")
        
        self._embedding_cache = {}
    
    def get_embedding(self, text: str) -> np.ndarray:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash not in self._embedding_cache:
            self._embedding_cache[text_hash] = self.model.encode(text)
        
        return self._embedding_cache[text_hash]
    
    def find_similar_clusters(self, prompts: List[str]) -> List[PromptCluster]:
        if not prompts:
            return []
        
        print(f"Analyzing {len(prompts)} prompts...")
        embeddings = np.array([self.get_embedding(p) for p in prompts])
        
        clusters = []
        used_indices = set()
        
        for i, prompt in enumerate(prompts):
            if i in used_indices:
                continue
            
            cluster_prompts = [prompt]
            cluster_scores = [1.0]
            cluster_indices = {i}
            
            for j, other_prompt in enumerate(prompts):
                if j <= i or j in used_indices:
                    continue
                
                similarity = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
                
                if similarity >= self.threshold:
                    cluster_prompts.append(other_prompt)
                    cluster_scores.append(float(similarity))
                    cluster_indices.add(j)
            
            # Only create cluster if we found similar prompts
            if len(cluster_prompts) > 1:
                clusters.append(PromptCluster(
                    representative=prompt,
                    prompts=cluster_prompts,
                    similarity_scores=cluster_scores,
                    count=len(cluster_prompts),
                    avg_similarity=float(np.mean(cluster_scores))
                ))
                used_indices.update(cluster_indices)
        
        clusters.sort(key=lambda c: c.count, reverse=True)
        
        return clusters
    
    def calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        emb1 = self.get_embedding(prompt1)
        emb2 = self.get_embedding(prompt2)
        
        similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        
        return float(similarity)
    
    def find_most_similar(self, query_prompt: str, candidate_prompts: List[str], threshold: Optional[float] = None) -> Optional[Tuple[str, float]]:
        if not candidate_prompts:
            return None
        
        threshold = threshold or self.threshold
        query_emb = self.get_embedding(query_prompt)
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidate_prompts:
            candidate_emb = self.get_embedding(candidate)
            similarity = cosine_similarity(
                query_emb.reshape(1, -1),
                candidate_emb.reshape(1, -1)
            )[0][0]
            
            if similarity > best_score and similarity >= threshold:
                best_score = float(similarity)
                best_match = candidate
        
        return (best_match, best_score) if best_match else None
    
    def suggest_caching(self, clusters: List[PromptCluster], model: str = 'gpt-3.5-turbo', model_recommender=None) -> List[dict]:

        if model_recommender is None:
            from .model_rec import ModelRecommender
            model_recommender = ModelRecommender()
        
        suggestions = []
        
        for cluster in clusters:
            if cluster.count >= 3:
                representative = cluster.representative
                
                input_tokens = model_recommender.estimate_tokens(representative)
                output_tokens = model_recommender.estimate_response_tokens(representative)
                
                cost_per_call = model_recommender.estimate_cost(model, input_tokens, output_tokens)
                
                calls_saved = cluster.count - 1
                estimated_savings = calls_saved * cost_per_call
                
                suggestions.append({
                    'cluster': cluster,
                    'model': model,
                    'reason': f'Called {cluster.count} times with {cluster.avg_similarity:.0%} avg similarity',
                    'estimated_savings': estimated_savings,
                    'calls_saved': calls_saved,
                    'cost_per_call': cost_per_call,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                })
        
        suggestions.sort(key=lambda x: x['estimated_savings'], reverse=True)
        
        return suggestions