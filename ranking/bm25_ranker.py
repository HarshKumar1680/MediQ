"""
ranking/bm25_ranker.py

BM25 (Best Match 25) Ranking — built from scratch.

Formula:
  Score(q,d) = sum over query terms t of:
    IDF(t) * [ tf(t,d) * (k1+1) ]
              / [ tf(t,d) + k1 * (1 - b + b * (|d|/avgdl)) ]

Parameters:
  k1 = 1.5   (term frequency saturation — controls how fast TF saturates)
  b  = 0.75  (document length normalization — 0 = no normalization, 1 = full)

Why BM25 is better than plain TF-IDF:
  1. TF SATURATION: BM25 diminishes returns for very high TF values.
     Mentioning "insulin" 20 times doesn't score 20x more than 2 times.
  2. LENGTH NORMALIZATION: Long documents are not unfairly rewarded.
     A 3000-word doc about "insulin" doesn't automatically beat a focused 300-word one.
  3. k1 & b are tunable hyperparameters proven effective across many test collections.

BM25 consistently outperforms plain TF-IDF on standard IR benchmarks (TREC, etc.)
"""

import math


class BM25Ranker:
    def __init__(self, index, k1=1.5, b=0.75):
        self.index  = index
        self.k1     = k1
        self.b      = b

    def score(self, query_tokens, doc_id):
        """
        Compute BM25 score for a single document.
        """
        total   = 0.0
        doc_len = self.index.doc_lengths.get(doc_id, 1)
        avgdl   = self.index.avg_doc_length or 1

        for term in query_tokens:
            postings = self.index.get_postings(term)
            tf = postings.get(doc_id, 0)
            if tf == 0:
                continue

            idf = self.index.idf(term)

            # BM25 TF component (saturating)
            numerator   = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avgdl))
            bm25_tf     = numerator / denominator

            total += idf * bm25_tf

        return round(total, 6)

    def rank(self, query_tokens, top_k=10):
        """
        Rank all candidate documents by BM25 score.
        Returns list of (doc_id, score) sorted descending.
        """
        candidates = self.index.candidate_docs(query_tokens)
        scores = []
        for doc_id in candidates:
            s = self.score(query_tokens, doc_id)
            if s > 0:
                scores.append((doc_id, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
