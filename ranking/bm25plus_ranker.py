"""
ranking/bm25plus_ranker.py

BM25+ (Lv & Zhai, 2011) — Latest improvement over BM25.

PROBLEM WITH BM25:
  BM25 assigns zero score to a document if a query term appears in ALL docs,
  because IDF becomes 0. Also, documents with very low TF are scored identically
  to documents with slightly higher TF due to saturation.

BM25+ FIX:
  Adds a lower-bound delta (δ) to the TF component, ensuring every document
  that contains a query term gets a non-zero positive contribution.

Formula:
  Score(q,d) = Σ IDF(t) * [ delta + tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl)) ]

Parameters:
  k1    = 1.5   (TF saturation)
  b     = 0.75  (length normalization)
  delta = 0.5   (lower-bound constant — the key BM25+ addition)

Academic Reference:
  Lv, Y., & Zhai, C. (2011). "Lower-bounding term frequency normalization."
  CIKM '11. doi:10.1145/2063576.2063584
"""

import math


class BM25PlusRanker:
    def __init__(self, index, k1=1.5, b=0.75, delta=0.5):
        self.index = index
        self.k1    = k1
        self.b     = b
        self.delta = delta          # THE key BM25+ parameter

    def score(self, query_tokens, doc_id):
        total   = 0.0
        doc_len = self.index.doc_lengths.get(doc_id, 1)
        avgdl   = self.index.avg_doc_length or 1

        for term in query_tokens:
            postings = self.index.get_postings(term)
            tf = postings.get(doc_id, 0)
            if tf == 0:
                continue

            idf = self.index.idf(term)

            # BM25+ TF component: standard BM25 + delta lower-bound
            norm_tf = tf * (self.k1 + 1) / (
                tf + self.k1 * (1 - self.b + self.b * (doc_len / avgdl))
            )
            bm25plus_tf = self.delta + norm_tf

            total += idf * bm25plus_tf

        return round(total, 6)

    def rank(self, query_tokens, top_k=10):
        candidates = self.index.candidate_docs(query_tokens)
        scores = [(doc_id, self.score(query_tokens, doc_id)) for doc_id in candidates]
        scores = [(d, s) for d, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
