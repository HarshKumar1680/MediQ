

import math


class TFIDFRanker:
    def __init__(self, index):
        self.index = index

    def score(self, query_tokens, doc_id):
        """
        Compute TF-IDF score for a single document given query tokens.
        """
        total = 0.0
        doc_len = self.index.doc_lengths.get(doc_id, 1)

        for term in query_tokens:
            postings = self.index.get_postings(term)
            tf_raw = postings.get(doc_id, 0)
            if tf_raw == 0:
                continue

            # Normalized TF: how often term appears relative to doc length
            tf = tf_raw / doc_len

            # IDF: log-smoothed, rewards rare terms
            idf = self.index.idf(term)

            total += tf * idf

        return round(total, 6)

    def rank(self, query_tokens, top_k=10):
        """
        Rank all candidate documents by TF-IDF score.
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
