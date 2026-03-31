"""
evaluation/metrics.py

IR Evaluation Metrics:
  - Precision@k  : fraction of top-k results that are relevant
  - Recall@k     : fraction of all relevant docs found in top-k
  - F1@k         : harmonic mean of Precision and Recall at k
  - AP           : Average Precision for one query
  - MAP          : Mean Average Precision across multiple queries
"""


def precision_at_k(retrieved, relevant, k):
    """
    P@k = |{retrieved[:k]} ∩ {relevant}| / k

    How many of the top-k retrieved documents are actually relevant?
    """
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits  = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved, relevant, k):
    """
    R@k = |{retrieved[:k]} ∩ {relevant}| / |relevant|

    What fraction of all relevant documents did we find in the top-k?
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits  = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def f1_score_at_k(retrieved, relevant, k):
    """
    F1@k = 2 * P@k * R@k / (P@k + R@k)

    Harmonic mean balances Precision and Recall.
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision(retrieved, relevant):
    """
    AP = (1/|R|) * sum_k [ P@k * rel(k) ]

    Where rel(k) = 1 if the k-th retrieved doc is relevant, else 0.
    AP rewards systems that rank relevant docs higher.
    """
    if not relevant:
        return 0.0

    hits       = 0
    sum_prec   = 0.0

    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits     += 1
            sum_prec += hits / rank   # precision at this rank position

    return sum_prec / len(relevant)


def mean_average_precision(results_per_query):
    """
    MAP = (1/|Q|) * sum over queries [ AP(q) ]

    results_per_query: list of (retrieved_list, relevant_set) tuples
    """
    if not results_per_query:
        return 0.0
    ap_sum = sum(average_precision(ret, rel) for ret, rel in results_per_query)
    return ap_sum / len(results_per_query)


def evaluate_query(query_label, retrieved_ids, relevant_ids, k=10):
    """
    Compute all metrics for a single query and return a dict.
    """
    relevant = set(relevant_ids)
    p  = precision_at_k(retrieved_ids, relevant, k)
    r  = recall_at_k(retrieved_ids, relevant, k)
    f1 = f1_score_at_k(retrieved_ids, relevant, k)
    ap = average_precision(retrieved_ids, relevant)
    return {
        "query":    query_label,
        f"P@{k}":   round(p,  4),
        f"R@{k}":   round(r,  4),
        f"F1@{k}":  round(f1, 4),
        "AP":       round(ap, 4),
    }
