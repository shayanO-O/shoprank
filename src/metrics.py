import numpy as np

# Compute Normalized Discounted Cumulative Gain
def ndcg_at_k(actual_items, recommended_items, k=10):
    if not actual_items:
        return 0.0
    actual_set = set(actual_items)
    top_recs = recommended_items[:k]
    dcg = 0.0
    for idx, item in enumerate(top_recs):
        if item in actual_set:
            dcg += 1.0 / np.log2(idx + 2)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(k, len(actual_set))))
    return float(dcg / idcg) if idcg > 0.0 else 0.0

# Compute Recall At Rank Cutoff
def recall_at_k(actual_items, recommended_items, k=50):
    if not actual_items:
        return 0.0
    hits = len(set(actual_items) & set(recommended_items[:k]))
    return float(hits / len(actual_items))

# Calculate Unique Recommendation Coverage
def catalog_coverage(all_top_recs, catalog_size):
    if catalog_size <= 0:
        return 0.0
    unique_items = set()
    for recs in all_top_recs:
        unique_items.update(recs[:10])
    return float(len(unique_items) / catalog_size)

# Compute Intra List Genre Diversity
def intra_list_diversity(all_top_recs, item_genre_dict):
    diversities = []
    for recs in all_top_recs:
        vecs = [item_genre_dict[item] for item in recs[:10] if item in item_genre_dict]
        if len(vecs) < 2:
            diversities.append(1.0)
            continue
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = arr / norms
        sim_matrix = np.dot(normed, normed.T)
        triu_indices = np.triu_indices(len(normed), k=1)
        mean_sim = np.mean(sim_matrix[triu_indices])
        diversities.append(float(1.0 - mean_sim))
    return float(np.mean(diversities)) if diversities else 1.0

# Calculate Request Latency Percentiles
def calculate_latency_percentiles(latencies):
    if not latencies:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95))
    }
