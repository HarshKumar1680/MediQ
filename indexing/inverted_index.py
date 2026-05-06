
import math
from collections import defaultdict


class InvertedIndex:
    def __init__(self):
        # Core index: term -> {doc_id: tf}
        self.index = defaultdict(lambda: defaultdict(int))
        self.doc_lengths   = {}   # {doc_id: num_tokens}
        self.doc_texts     = {}   # {doc_id: raw_text}
        self.doc_titles    = {}   # {doc_id: title}
        self.total_docs    = 0
        self.avg_doc_length = 0.0

    # ------------------------------------------------------------------
    def build(self, raw_docs, processed_docs):
        """
        Build the inverted index from preprocessed documents.

        raw_docs       : {doc_id: raw_text}
        processed_docs : {doc_id: [token, token, ...]}
        """
        print("[Index] Building inverted index...")
        self.doc_texts  = raw_docs
        self.total_docs = len(processed_docs)

        for doc_id, tokens in processed_docs.items():
            # Store raw title (first non-empty line)
            lines = raw_docs[doc_id].strip().splitlines()
            self.doc_titles[doc_id] = next(
                (l.replace("Title:","").strip() for l in lines if l.strip()), doc_id
            )

            self.doc_lengths[doc_id] = len(tokens)

            # Count term frequencies
            tf_map = defaultdict(int)
            for token in tokens:
                tf_map[token] += 1

            # Insert into index
            for term, freq in tf_map.items():
                self.index[term][doc_id] = freq

        self.avg_doc_length = (
            sum(self.doc_lengths.values()) / self.total_docs
            if self.total_docs else 0
        )

        print(f"[Index] Indexed {self.total_docs} docs | "
              f"{len(self.index)} unique terms | "
              f"avg doc length = {self.avg_doc_length:.1f} tokens\n")

    # ------------------------------------------------------------------
    def get_postings(self, term):
        """Return posting list for a term: {doc_id: tf}"""
        return dict(self.index.get(term, {}))

    def df(self, term):
        """Document frequency of a term."""
        return len(self.index.get(term, {}))

    def idf(self, term):
        """
        Inverse Document Frequency (smoothed):
          IDF(t) = log( (N + 1) / (df(t) + 1) ) + 1
        The +1 avoids zero IDF for terms appearing in all docs.
        """
        df = self.df(term)
        if df == 0:
            return 0.0
        return math.log((self.total_docs + 1) / (df + 1)) + 1

    def candidate_docs(self, query_tokens):
        """
        Return union of all docs containing at least one query term.
        This is the candidate set for ranking.
        """
        candidates = set()
        for token in query_tokens:
            candidates.update(self.index.get(token, {}).keys())
        return candidates

    def stats(self):
        return {
            "total_docs":       self.total_docs,
            "unique_terms":     len(self.index),
            "avg_doc_length":   round(self.avg_doc_length, 2),
        }
