"""
app/app.py — Flask server with deployment-friendly configuration.
"""
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BASE)

# Keep NLTK assets in a deterministic folder for hosted environments.
NLTK_DATA_DIR = os.path.join(BASE, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
os.environ.setdefault("NLTK_DATA", NLTK_DATA_DIR)

from flask import Flask, request, jsonify, send_from_directory
import main as ir

app = Flask(__name__,
    static_folder  =os.path.join(BASE, "static"),
    template_folder=os.path.join(BASE, "templates"))

# Tell Flask not to sort JSON keys (keeps ordering)
app.config["JSON_SORT_KEYS"] = False

@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE, "templates"), "index.html")

@app.route("/search")
def search():
    q      = request.args.get("q","").strip()
    method = request.args.get("method","bm25").lower()
    try:
        top_k = min(int(request.args.get("top_k", 10)), 20)
    except ValueError:
        top_k = 10
    if not q:
        return jsonify({"error": "Empty query"}), 400
    try:
        results, tokens, elapsed = ir.query(q, method=method, top_k=top_k)
        return jsonify({
            "query": q, "method": method, "tokens": tokens,
            "time_ms": elapsed, "total_results": len(results), "results": results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compare")
def compare():
    q     = request.args.get("q","").strip()
    try:
        top_k = min(int(request.args.get("top_k", 10)), 20)
    except ValueError:
        top_k = 10
    if not q:
        return jsonify({"error": "Empty query"}), 400
    try:
        comp, tokens = ir.compare_all(q, top_k=top_k)
        return jsonify({"query": q, "tokens": tokens, "comparison": comp})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate")
def evaluate():
    method = request.args.get("method","bm25")
    try:
        return jsonify(ir.run_evaluation(method=method, k=10))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/algos")
def algos():
    return jsonify(ir.get_algo_meta())

@app.route("/stats")
def stats():
    return jsonify(ir.get_stats())

# Suppress 404 for favicon
@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(os.path.join(BASE,"static"), path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"\n[Flask] http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
