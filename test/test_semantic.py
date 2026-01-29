import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.analyzer.semantic import SemanticAnalyzer

def test_find_similar_clusters():
    
    analyzer = SemanticAnalyzer(similarity_threshold=0.85)
    
    prompts = [
        "Write a Python function to sort a list",
        "Create a Python function for sorting lists",
        "Make a sorting function in Python",
        "Calculate the fibonacci sequence",
        "Compute fibonacci numbers",
        "What is 2+2?",
        "Tell me about quantum physics",
    ]
    
    clusters = analyzer.find_similar_clusters(prompts)
    
    print(f"Found {len(clusters)} clusters")
    for i, cluster in enumerate(clusters, 1):
        print(f"\nCluster {i}: {cluster.count} prompts, avg similarity: {cluster.avg_similarity:.1%}")
        print(f"  Representative: \"{cluster.representative}\"")
    
    assert len(clusters) >= 2, f"Expected at least 2 clusters, got {len(clusters)}"
    assert clusters[0].count >= 2, "Largest cluster should have at least 2 prompts"
    
    sorting_cluster = None
    for cluster in clusters:
        if "sort" in cluster.representative.lower():
            sorting_cluster = cluster
            break
    
    assert sorting_cluster is not None, "Should find a sorting cluster"
    assert sorting_cluster.count == 3, f"Sorting cluster should have 3 prompts, got {sorting_cluster.count}"

def test_calculate_similarity():
    
    analyzer = SemanticAnalyzer(similarity_threshold=0.85)
    
    test_pairs = [
        ("Write a Python sort function", "Create a Python sorting function", 0.85),
        ("Analyze sentiment", "Determine sentiment", 0.80),
        ("What is 2+2?", "Explain quantum physics", 0.30),
    ]
    
    for prompt1, prompt2, expected_min in test_pairs:
        similarity = analyzer.calculate_similarity(prompt1, prompt2)
        print(f"\n'{prompt1}' vs '{prompt2}'")
        print(f"  Similarity: {similarity:.2%}")
        
        if expected_min > 0.5:
            assert similarity >= expected_min, \
                f"Expected similarity >= {expected_min:.0%}, got {similarity:.0%}"
        else:
            assert similarity <= 0.5, \
                f"Expected low similarity (<= 50%), got {similarity:.0%}"

def test_find_most_similar():
    
    analyzer = SemanticAnalyzer(similarity_threshold=0.80)
    
    candidates = [
        "Write a Python function to sort",
        "Calculate fibonacci numbers",
        "Analyze sentiment of text",
    ]
    
    query = "Create a sorting function in Python"
    
    result = analyzer.find_most_similar(query, candidates)
    
    assert result is not None, "Should find a match"
    match, score = result
    
    print(f"Query: '{query}'")
    print(f"Best match: '{match}'")
    print(f"Score: {score:.2%}")
    
    assert "sort" in match.lower(), "Best match should be about sorting"
    assert score >= 0.80, f"Score should be >= 80%, got {score:.0%}"


def test_threshold_variations():
    
    # Mix of very similar and somewhat similar prompts
    prompts = [
        "Write a Python function to sort a list",
        "Create a Python function for sorting lists", 
        "Make a sorting function in Python",
        "Calculate fibonacci sequence",
        "Compute fibonacci numbers",
    ]
    
    results = {}
    
    for threshold in [0.70, 0.85, 0.95]:
        analyzer = SemanticAnalyzer(similarity_threshold=threshold)
        clusters = analyzer.find_similar_clusters(prompts)
        results[threshold] = len(clusters)
        
        print(f"\nThreshold {threshold:.0%}: {len(clusters)} clusters")
    
    # lower thresholds should find same or more clusters
    assert results[0.70] >= results[0.85], \
        "70% threshold should find >= clusters than 85%"
    assert results[0.85] >= results[0.95], \
        "85% threshold should find >= clusters than 95%"
    
    assert results[0.70] >= 1, \
        f"Should find at least 1 cluster at 70%, found {results[0.70]}"

def test_embedding_cache():
    analyzer = SemanticAnalyzer()
    
    prompt = "Test prompt for caching"
    
    emb1 = analyzer.get_embedding(prompt)
    cache_size_1 = len(analyzer._embedding_cache)
    
    emb2 = analyzer.get_embedding(prompt)
    cache_size_2 = len(analyzer._embedding_cache)
    
    print(f"Cache size after first call: {cache_size_1}")
    print(f"Cache size after second call: {cache_size_2}")
    
    assert cache_size_1 == cache_size_2, "Cache size should not change on second call"
    assert (emb1 == emb2).all(), "Cached embeddings should be identical"
    
    print("All assertions passed!")

if __name__ == "__main__":
    
    test_find_similar_clusters()
    test_calculate_similarity()
    test_find_most_similar()
    test_threshold_variations()
    test_embedding_cache()
    
    print("\nALL SEMANTIC TESTS PASSED")
