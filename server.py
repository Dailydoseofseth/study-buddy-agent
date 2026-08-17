"""
server.py — Study Buddy web backend.

Exposes the existing Study Buddy quiz agent (study_buddy_agent.py) over a
minimal Flask HTTP API instead of its terminal REPL. Mirrors the CLI's
__main__ block exactly: one genai.Client and one shared history list,
created once at process start, mutated in place across requests.

Single-session by design (matches the CLI's own single-global-history
model) — no per-browser-session state, cookies, or multi-user support.
"""

import os

from flask import Flask, request, jsonify
from google import genai

from study_buddy_agent import (
    run_agent_turn,
    get_score,
    get_current_hint,
    get_current_choices,
    get_last_answer_correct,
    get_current_visual,
)

app = Flask(__name__, static_folder="static", static_url_path="")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
history = []


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    history.append({"type": "user_input", "content": [{"type": "text", "text": user_message}]})

    try:
        reply = run_agent_turn(client, history)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    score = get_score()
    hint = get_current_hint()
    choices = get_current_choices()
    correct = get_last_answer_correct()
    visual = get_current_visual()
    return jsonify(
        {"reply": reply, "score": score, "hint": hint, "choices": choices, "correct": correct, "visual": visual}
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
