"""
preprocessing/preprocess.py  — with guaranteed NLTK download on import
"""
import re, sys, nltk

def _download_all():
    """Download all required NLTK data packages silently before anything else."""
    needed = [
        ("tokenizers/punkt",       "punkt"),
        ("tokenizers/punkt_tab",   "punkt_tab"),
        ("corpora/stopwords",      "stopwords"),
        ("corpora/wordnet",        "wordnet"),
        ("corpora/omw-1.4",        "omw-1.4"),
        ("taggers/averaged_perceptron_tagger",     "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  [NLTK] Downloading '{pkg}'...", flush=True)
            nltk.download(pkg, quiet=True)

print("[NLTK] Checking resources...", flush=True)
_download_all()
print("[NLTK] All resources ready.", flush=True)

from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords, wordnet
from nltk.stem     import WordNetLemmatizer
from nltk          import pos_tag

_lemmatizer = WordNetLemmatizer()
_STOP = set(stopwords.words("english")) | {
    "patient","patients","may","also","used","use","include","including",
    
}

def _wn_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOP and len(t) > 2]
    return [_lemmatizer.lemmatize(tok, _wn_pos(pos)) for tok, pos in pos_tag(tokens)]

def preprocess_corpus(documents):
    print(f"[Preprocessing] {len(documents)} documents...", flush=True)
    result = {}
    for doc_id, text in documents.items():
        result[doc_id] = preprocess_text(text)
        print(f"  ok {doc_id} -> {len(result[doc_id])} tokens", flush=True)
    print("[Preprocessing] Done.\n", flush=True)
    return result

def preprocess_query(query):
    tokens = preprocess_text(query)
    print(f"[Query] '{query}' -> {tokens}", flush=True)
    return tokens
