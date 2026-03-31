"""
app/app.py — Flask server with timeout-safe endpoints.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify, send_from_directory
import main as ir

BASE = os.path.join(os.path.dirname(__file__), "..")

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
    top_k  = min(int(request.args.get("top_k", 10)), 20)
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
    top_k = min(int(request.args.get("top_k", 10)), 20)
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
    print("\n[Flask] http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000, threaded=True)
