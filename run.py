"""
run.py — One-click launcher for MedIR.
Run:  python run.py
"""
import sys, os, nltk

# ── Step 0: Download ALL NLTK data first, before anything else ──
print("\n" + "="*55)
print("  MedIR — Medical IR System")
print("="*55)
print("\n[Step 0] Downloading NLTK resources (first run only)...")

packages = [
    "punkt", "punkt_tab", "stopwords", "wordnet",
    "omw-1.4", "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng"
]
for pkg in packages:
    try:
        nltk.download(pkg, quiet=True)
        print(f"  ✓ {pkg}")
    except Exception as e:
        print(f"  ! {pkg}: {e}")

print("[Step 0] NLTK ready.\n")

# ── Step 1: Now import and start Flask ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print("\n" + "="*55)
    print(f"  Open browser → http://127.0.0.1:{port}")
    print("  Press Ctrl+C to stop")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
