

import math
from collections import defaultdict


class LanguageModelRanker:
    def __init__(self, index, lam=0.1):
        self.index = index
        self.lam   = lam        # Jelinek-Mercer smoothing parameter λ
        self._corpus_probs = {}
        self._total_tokens = 0
        self._build_corpus_model()

    def _build_corpus_model(self):
        """
        Compute collection language model: P(t|C) = cf(t) / total_corpus_tokens
        cf(t) = total occurrences of term t across all documents.
        """
        cf = defaultdict(int)
        total = 0
        for term, postings in self.index.index.items():
            count = sum(postings.values())
            cf[term] = count
            total   += count

        self._total_tokens = total or 1
        self._corpus_probs = {t: c / self._total_tokens for t, c in cf.items()}

    def score(self, query_tokens, doc_id):
        """
        Log-probability of generating query tokens from document d's LM.
        Score = Σ_t  log( (1-λ)*P_ml(t|d) + λ*P(t|C) )
        """
        doc_len = self.index.doc_lengths.get(doc_id, 1)
        log_score = 0.0

        for term in query_tokens:
            tf  = self.index.get_postings(term).get(doc_id, 0)
            p_ml  = tf / doc_len                              # doc probability
            p_col = self._corpus_probs.get(term, 1e-10)      # corpus probability

            p_smooth = (1 - self.lam) * p_ml + self.lam * p_col

            if p_smooth <= 0:
                log_score += math.log(1e-10)    # floor for unseen terms
            else:
                log_score += math.log(p_smooth)

        return round(log_score, 6)

    def rank(self, query_tokens, top_k=10):
        candidates = self.index.candidate_docs(query_tokens)
        scores = []
        for doc_id in candidates:
            s = self.score(query_tokens, doc_id)
            scores.append((doc_id, s))

        # Higher (less negative) log-prob = better match
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
