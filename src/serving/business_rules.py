from collections import defaultdict
from typing import Dict, List, Optional, Set
import numpy as np

def filter_seen_items(candidates: List[int], seen_set: Set[int]) -> List[int]:
    """
    Filters out items that the user has already seen/rated, preserving order.
    """
    if not seen_set:
        return candidates
    return [mid for mid in candidates if mid not in seen_set]

def filter_unavailable_items(candidates: List[int], unavailable_set: Set[int]) -> List[int]:
    """
    Filters out items that are marked as unavailable (out-of-stock/delisted).
    """
    if not unavailable_set:
        return candidates
    return [mid for mid in candidates if mid not in unavailable_set]

def enforce_diversity(ranked_candidates: List[int],
                      item_genre_dict: Dict[int, np.ndarray],
                      max_per_genre: int = 3,
                      top_k: int = 10) -> List[int]:
    """
    Intra-list diversity post-processing:
    Greedily selects items from ranked_candidates ensuring no dominant genre
    appears more than max_per_genre times.
    Surplus items from dominant genres are held in an overflow buffer and only
    used if insufficient diverse candidates exist to fulfill top_k.
    """
    if not ranked_candidates:
        return []
    
    genre_counts: Dict[int, int] = defaultdict(int)
    selected: List[int] = []
    overflow: List[int] = []
    
    for mid in ranked_candidates:
        g_vec = item_genre_dict.get(mid)
        # Dominant genre is the highest index/element in the one-hot or multi-hot vector
        dom_genre = int(np.argmax(g_vec)) if (g_vec is not None and np.any(g_vec > 0)) else -1
        
        if dom_genre >= 0 and genre_counts[dom_genre] >= max_per_genre:
            overflow.append(mid)
        else:
            selected.append(mid)
            if dom_genre >= 0:
                genre_counts[dom_genre] += 1
                
        if len(selected) == top_k:
            break
            
    # If selected < top_k, fill remaining slots from overflow to guarantee top_k recommendations
    if len(selected) < top_k and overflow:
        needed = top_k - len(selected)
        selected.extend(overflow[:needed])
        
    return selected

def trim_top_k(ranked_candidates: List[int], k: int = 10) -> List[int]:
    """
    Trims recommendation list to exact top-k items.
    """
    return ranked_candidates[:k]

def apply_business_rules(candidates: List[int],
                         seen_set: Optional[Set[int]] = None,
                         unavailable_set: Optional[Set[int]] = None,
                         item_genre_dict: Optional[Dict[int, np.ndarray]] = None,
                         max_per_genre: int = 3,
                         top_k: int = 10) -> List[int]:
    """
    Executes the full business rules post-processing pipeline:
    1. Seen items removal
    2. Catalog availability filtering
    3. Category diversity enforcement
    4. Top-K cutoff
    """
    items = candidates
    if seen_set:
        items = filter_seen_items(items, seen_set)
    if unavailable_set:
        items = filter_unavailable_items(items, unavailable_set)
    if item_genre_dict:
        items = enforce_diversity(items, item_genre_dict, max_per_genre=max_per_genre, top_k=top_k)
    return trim_top_k(items, k=top_k)
