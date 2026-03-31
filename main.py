"""
main.py
Loads corpus, builds index, initialises all 4 ranking algorithms,
and exposes query() + evaluate() + compare_all() functions.

Algorithms implemented:
  1. TF-IDF       — Classical normalized TF × IDF dot product
  2. BM25         — Probabilistic ranking with TF saturation (Robertson, 1994)
  3. BM25+        — BM25 with delta lower-bound fix (Lv & Zhai, 2011)
  4. VSM Cosine   — Sublinear TF-IDF with L2-normalized cosine similarity
  5. LM (JM)      — Query Likelihood Language Model, Jelinek-Mercer smoothing
"""

import os, glob, sys, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "documents")
sys.path.insert(0, BASE_DIR)

from preprocessing.preprocess  import preprocess_corpus, preprocess_query
from indexing.inverted_index   import InvertedIndex
from ranking.tfidf_ranker      import TFIDFRanker
from ranking.bm25_ranker       import BM25Ranker
from ranking.bm25plus_ranker   import BM25PlusRanker
from ranking.vsm_ranker        import VSMRanker
from ranking.lm_ranker         import LanguageModelRanker
from evaluation.metrics        import evaluate_query, mean_average_precision

# ── 1. Load documents ────────────────────────────────────────────────
def load_documents(folder):
    docs = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            docs[doc_id] = f.read()
    print(f"[Loader] {len(docs)} documents loaded.")
    return docs

# ── 2. Build everything ──────────────────────────────────────────────
print("\n[System] Initialising Medical IR System...")
t0 = time.perf_counter()

raw_docs       = load_documents(DATA_DIR)
processed_docs = preprocess_corpus(raw_docs)

index = InvertedIndex()
index.build(raw_docs, processed_docs)

# Initialise all rankers (VSM precomputes doc norms on init)
print("[System] Building rankers...")
t_rankers = time.perf_counter()
tfidf_ranker  = TFIDFRanker(index)
bm25_ranker   = BM25Ranker(index)
bm25p_ranker  = BM25PlusRanker(index)
vsm_ranker    = VSMRanker(index)       # precomputes doc norms here
lm_ranker     = LanguageModelRanker(index)
print(f"[System] All rankers ready in {time.perf_counter()-t_rankers:.3f}s")

RANKERS = {
    "tfidf":  tfidf_ranker,
    "bm25":   bm25_ranker,
    "bm25p":  bm25p_ranker,
    "vsm":    vsm_ranker,
    "lm":     lm_ranker,
}

ALGO_META = {
    "tfidf": {"label": "TF-IDF",          "year": 1972, "color": "#e8703e",
              "desc":  "Normalized term frequency × inverse document frequency. "
                       "Classic IR baseline (Salton & Yang, 1973)."},
    "bm25":  {"label": "BM25",            "year": 1994, "color": "#3b82f6",
              "desc":  "Best Match 25: probabilistic TF saturation + length "
                       "normalization (Robertson & Walker, 1994)."},
    "bm25p": {"label": "BM25+",           "year": 2011, "color": "#8b5cf6",
              "desc":  "BM25 with delta lower-bound fix ensuring non-zero scores. "
                       "Fixes BM25's IDF under-estimation (Lv & Zhai, 2011)."},
    "vsm":   {"label": "VSM Cosine",      "year": 2008, "color": "#10b981",
              "desc":  "Sublinear TF scaling + L2-normalized cosine similarity. "
                       "Modern standard vector space model (Manning et al., 2008)."},
    "lm":    {"label": "LM Jelinek-Mercer","year": 1998, "color": "#f59e0b",
              "desc":  "Query likelihood language model with JM smoothing. "
                       "Probabilistic generative framework (Ponte & Croft, 1998)."},
}

total_init = time.perf_counter() - t0
print(f"[System] Total init time: {total_init:.3f}s\n")

# ── 3. Query function ────────────────────────────────────────────────
def _parse_doc(doc_id):
    raw   = raw_docs.get(doc_id, "")
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    title   = next((l.replace("Title:","").strip()   for l in lines if l.startswith("Title:")),   doc_id)
    disease = next((l.replace("Disease:","").strip() for l in lines if l.startswith("Disease:")), "")
    source  = next((l.replace("Source:","").strip()  for l in lines if l.startswith("Source:")),  "")
    content = [l for l in lines if not l.startswith(("Title:","Source:","Disease:"))]
    snippet = " ".join(content)[:320] + "..."
    return title, disease, source, snippet

def query(q, method="bm25", top_k=10):
    """
    Run a timed query with one algorithm.
    Returns (results_list, tokens, time_ms).
    """
    tokens = preprocess_query(q)
    ranker = RANKERS.get(method, bm25_ranker)

    t_start = time.perf_counter()
    ranked  = ranker.rank(tokens, top_k=top_k)
    t_end   = time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000

    results = []
    for rank, (doc_id, score) in enumerate(ranked, 1):
        title, disease, source, snippet = _parse_doc(doc_id)
        results.append({
            "rank":           rank,
            "doc_id":         doc_id,
            "title":          title,
            "disease":        disease,
            "source":         source,
            "score":          score,
            "snippet":        snippet,
            "tokens_matched": [t for t in tokens
                                if t in index.index and doc_id in index.index[t]],
        })
    return results, tokens, round(elapsed_ms, 4)


def compare_all(q, top_k=10):
    """
    Run ALL algorithms on the same query and return timing + results for each.
    Returns dict keyed by method with timing and top results.
    """
    tokens = preprocess_query(q)
    comparison = {}

    for method, ranker in RANKERS.items():
        t_start = time.perf_counter()
        ranked  = ranker.rank(tokens, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        results = []
        for rank, (doc_id, score) in enumerate(ranked, 1):
            title, disease, source, snippet = _parse_doc(doc_id)
            results.append({
                "rank": rank, "doc_id": doc_id,
                "title": title, "disease": disease,
                "score": score, "snippet": snippet,
                "tokens_matched": [t for t in tokens
                                    if t in index.index and doc_id in index.index[t]],
            })

        comparison[method] = {
            "method":     method,
            "label":      ALGO_META[method]["label"],
            "year":       ALGO_META[method]["year"],
            "color":      ALGO_META[method]["color"],
            "desc":       ALGO_META[method]["desc"],
            "time_ms":    round(elapsed_ms, 4),
            "results":    results,
            "top1_title": results[0]["title"] if results else "—",
        }

    return comparison, tokens


# ── 4. Ground truth & evaluation ─────────────────────────────────────
GROUND_TRUTH = {
    "diabetes treatment":          ["doc_001_diabetes_type2","doc_031_diabetes_type1"],
    "asthma therapy":              ["doc_002_asthma"],
    "heart disease management":    ["doc_003_heart_disease","doc_035_heart_failure","doc_040_atrial_fibrillation"],
    "hypertension blood pressure": ["doc_004_hypertension"],
    "cancer chemotherapy":         ["doc_010_cancer_treatment","doc_028_breast_cancer","doc_036_lung_cancer"],
    "HIV antiretroviral":          ["doc_011_hiv_aids"],
    "depression antidepressant":   ["doc_009_depression"],
    "malaria artemisinin":         ["doc_007_malaria"],
    "tuberculosis antibiotic":     ["doc_005_tuberculosis"],
    "epilepsy seizure":            ["doc_014_epilepsy"],
}

def run_evaluation(method="bm25", k=10):
    all_results, per_query = [], []
    for q_text, relevant_ids in GROUND_TRUTH.items():
        results, _, _ = query(q_text, method=method, top_k=k)
        retrieved  = [r["doc_id"] for r in results]
        metrics    = evaluate_query(q_text, retrieved, relevant_ids, k=k)
        per_query.append(metrics)
        all_results.append((retrieved, set(relevant_ids)))
    map_score = mean_average_precision(all_results)
    return {"MAP": round(map_score, 4), "queries": per_query}

def get_stats():
    return {**index.stats(), "algorithms": list(ALGO_META.keys())}

def get_algo_meta():
    return ALGO_META


if __name__ == "__main__":
    print("\n=== Compare All Algorithms ===")
    comp, toks = compare_all("diabetes treatment insulin")
    print(f"Tokens: {toks}\n")
    print(f"{'Algorithm':<22} {'Time (ms)':>10}  {'Top Result'}")
    print("-"*70)
    for m, data in comp.items():
        print(f"{data['label']:<22} {data['time_ms']:>10.4f}  {data['top1_title'][:40]}")
