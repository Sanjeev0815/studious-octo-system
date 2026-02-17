import math


def precision_at_k(relevances, k):
    if k == 0:
        return 0.0
    return sum(relevances[:k]) / k


def recall_at_k(relevances, k):
    total_relevant = sum(relevances)
    if total_relevant == 0:
        return 0.0
    return sum(relevances[:k]) / total_relevant


def mean_reciprocal_rank(relevances):
    for idx, rel in enumerate(relevances, start=1):
        if rel:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(relevances, k):
    def dcg(scores):
        return sum(score / math.log2(idx + 2) for idx, score in enumerate(scores))

    actual = dcg(relevances[:k])
    ideal = dcg(sorted(relevances, reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return actual / ideal
