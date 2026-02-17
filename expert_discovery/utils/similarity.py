import numpy as np


def cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return 0.0
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def jaccard_similarity(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def min_max_normalize(values):
    if not values:
        return {}
    min_val = min(values.values())
    max_val = max(values.values())
    if max_val == min_val:
        return {key: 0.0 for key in values}
    return {key: (val - min_val) / (max_val - min_val) for key, val in values.items()}


def weighted_score(features, weights):
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    score = 0.0
    for name, weight in weights.items():
        score += features.get(name, 0.0) * weight
    return score / total_weight
