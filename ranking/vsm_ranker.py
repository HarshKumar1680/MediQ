

import math
from collections import defaultdict


class VSMRanker:
    def __init__(self, index):
        self.index = index
        self._doc_norms = {}    # precomputed L2 norms for all docs
        self._build_doc_norms()

    def _build_doc_norms(self):
        """Precompute L2 norm of each document's TF-IDF vector (sublinear TF)."""
        doc_weights = defaultdict(float)

        for term, postings in self.index.index.items():
            idf = self.index.idf(term)
            for doc_id, tf in postings.items():
                w = (1 + math.log(tf)) * idf   # sublinear TF * IDF
                doc_weights[(doc_id, term)] = w

        # Group by doc and compute L2 norm
        norms = defaultdict(float)
        for (doc_id, term), w in doc_weights.items():
            norms[doc_id] += w * w
        self._doc_norms = {d: math.sqrt(v) for d, v in norms.items()}
        self._doc_weights = doc_weights

    def score(self, query_tokens, doc_id):
        """Cosine similarity between query vector and document vector."""
        doc_norm = self._doc_norms.get(doc_id, 0)
        if doc_norm == 0:
            return 0.0

        # Build query vector with sublinear TF
        query_tf = defaultdict(int)
        for t in query_tokens:
            query_tf[t] += 1

        dot = 0.0
        query_sq = 0.0
        for term, tf in query_tf.items():
            q_w = (1 + math.log(tf)) * self.index.idf(term)
            query_sq += q_w * q_w
            d_w = self._doc_weights.get((doc_id, term), 0.0)
            dot += q_w * d_w

        query_norm = math.sqrt(query_sq) if query_sq > 0 else 1.0
        cosine = dot / (query_norm * doc_norm)
        return round(cosine, 6)

    def rank(self, query_tokens, top_k=10):
        candidates = self.index.candidate_docs(query_tokens)
        scores = [(doc_id, self.score(query_tokens, doc_id)) for doc_id in candidates]
        scores = [(d, s) for d, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
