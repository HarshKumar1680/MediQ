

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
