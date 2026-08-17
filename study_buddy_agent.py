"""
study_buddy_agent.py — Study Buddy Agent

Quizzes the user on flashcards and tracks their score across the conversation.
Follows the same Reason -> Act -> Observe loop as agent_loop.py in agent-workshop.
"""

import difflib
import json
import os
import random
import re
import time
from collections import deque

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = "gemini-3.6-flash"


# ---------------------------------------------------------------------------
# 1. TOOLS — the actions this agent is allowed to take.
# ---------------------------------------------------------------------------

FLASHCARDS = {
    "js syntax": [
        {"question": "What keyword declares a block-scoped variable that can be reassigned?", "answer": "let"},
        {"question": "What keyword declares a variable that cannot be reassigned?", "answer": "const"},
        {"question": "What array method creates a new array by transforming every element?", "answer": "map"},
        {"question": "What operator checks equality without type coercion?", "answer": "==="},
        {"question": "What symbol wraps a template literal string?", "answer": "`"},
    ],
    "bugs": [
        {"question": "How many legs does an insect have?", "answer": "6"},
        {"question": "What is the largest order of insects, containing beetles?", "answer": "Coleoptera"},
        {"question": "What do you call a butterfly in its larval stage?", "answer": "caterpillar"},
        {"question": "What is the scientific study of insects called?", "answer": "entomology"},
        {"question": "How many wings does a typical housefly have?", "answer": "2"},
    ],
    "debugs": [
        {"question": "What is the common term for pausing code execution at a specific line to inspect state?", "answer": "breakpoint"},
        {"question": "What tool lets you step through code line-by-line to inspect variables?", "answer": "debugger"},
        {"question": "What do you call an error that occurs while the program is running, not at compile time?", "answer": "runtime error"},
        {"question": "What is the process of finding and fixing bugs in code called?", "answer": "debugging"},
        {"question": "What browser feature lets you inspect the DOM and console errors?", "answer": "DevTools"},
    ],
}

# Stretch goal: score persists across the conversation for as long as this
# process keeps running (a plain global dict the tool functions update).
score = {"correct": 0, "incorrect": 0}

# Per-topic shuffled draw piles, so a quiz session works through every card
# in a topic before any repeat. Refilled and reshuffled once exhausted.
_draw_piles = {}

# The flashcard most recently handed out, tracked server-side so grading
# always checks against the question we actually asked — rather than
# trusting the model to echo the question text back verbatim, which it can
# paraphrase and silently break a text-based lookup. Also remembers its own
# topic so check_answer_and_next can draw the next card without the model
# needing to repeat the topic argument.
current_question = None

# How many questions the current quiz was asked for, and how far into it we
# are — tracked here (not left to the model's memory) so Python itself can
# render "Question 2 of 3" and detect quiz-complete, which is what lets the
# agent loop skip asking the model to phrase that text on every turn.
# start_correct/start_incorrect snapshot the lifetime score at quiz start, so
# quiz_complete can report this quiz's own tally (a delta) instead of the
# running lifetime total, which would include points from earlier quizzes.
quiz_state = {"num_questions": 0, "question_num": 0, "start_correct": 0, "start_incorrect": 0}


def _draw_next_card(topic_key: str) -> str:
    """Pop the next card off the topic's shuffled draw pile (reshuffling a
    fresh one if empty) and record it as the current question."""
    global current_question
    pile = _draw_piles.get(topic_key)
    if not pile:
        pile = list(FLASHCARDS[topic_key])
        random.shuffle(pile)
        _draw_piles[topic_key] = pile

    card = pile.pop()
    current_question = {**card, "topic": topic_key}
    return card["question"]


def get_flashcard(topic: str, num_questions: int = 3) -> dict:
    """Start a quiz on the given topic by returning its first question (no answer).
    Only call this to START a topic — after grading an answer, check_answer_and_next
    already returns the next question, so don't call this again mid-quiz."""
    topic_key = topic.strip().lower()
    if topic_key not in FLASHCARDS:
        return {"error": f"No flashcards for '{topic}'. Try: {', '.join(FLASHCARDS)}."}

    quiz_state["num_questions"] = max(1, num_questions)
    quiz_state["question_num"] = 1
    quiz_state["start_correct"] = score["correct"]
    quiz_state["start_incorrect"] = score["incorrect"]
    return {
        "question": _draw_next_card(topic_key),
        "question_num": quiz_state["question_num"],
        "num_questions": quiz_state["num_questions"],
    }


def _is_close_enough(user_answer: str, correct_answer: str) -> bool:
    """Accept near-misses on word-like answers (typos, partial words like
    'debug' for 'debugging'), but require an exact match for short or
    symbolic/numeric answers — leniency there could flip the meaning
    (e.g. '==' is a substring of '===', but they're different operators;
    '6' vs '60' would also wrongly match on containment)."""
    user = user_answer.strip().lower().rstrip(".!?")
    correct = correct_answer.strip().lower().rstrip(".!?")
    if not user:
        return False
    if user == correct:
        return True
    if not (correct.isalpha() and len(correct) >= 4):
        return False

    user_collapsed = user.replace(" ", "")
    correct_collapsed = correct.replace(" ", "")
    if user_collapsed in correct_collapsed or correct_collapsed in user_collapsed:
        return True
    return difflib.SequenceMatcher(None, user_collapsed, correct_collapsed).ratio() >= 0.75


def check_answer_and_next(user_answer: str) -> dict:
    """Grade the user's answer to the current flashcard, update the score, and
    immediately return the next question from the same topic — combined into
    one tool call (instead of a separate check_answer + get_flashcard) so a
    3-question quiz costs ~3-4 API round-trips instead of ~6-8, which matters
    a lot on the free tier's 5-requests/minute cap."""
    global current_question
    if current_question is None:
        return {"error": "No active question — call get_flashcard first."}

    correct_answer = current_question["answer"]
    topic_key = current_question["topic"]
    is_correct = _is_close_enough(user_answer, correct_answer)
    if is_correct:
        score["correct"] += 1
    else:
        score["incorrect"] += 1

    result = {
        "correct": is_correct,
        "correct_answer": correct_answer,
        "feedback": "Correct!" if is_correct else f"Not quite. The correct answer was '{correct_answer}'.",
    }

    quiz_state["question_num"] += 1
    if quiz_state["question_num"] > quiz_state["num_questions"]:
        # Quiz done — don't draw a card that will never be shown.
        # Report this quiz's own tally (a delta off the start-of-quiz
        # snapshot), not the lifetime session total from get_score().
        quiz_correct = score["correct"] - quiz_state["start_correct"]
        quiz_incorrect = score["incorrect"] - quiz_state["start_incorrect"]
        result["quiz_complete"] = True
        result["final_score"] = {
            "correct": quiz_correct,
            "incorrect": quiz_incorrect,
            "total": quiz_correct + quiz_incorrect,
        }
    else:
        result["next_question"] = _draw_next_card(topic_key)
        result["question_num"] = quiz_state["question_num"]
        result["num_questions"] = quiz_state["num_questions"]

    return result


def get_score() -> dict:
    """Report how many questions the user has gotten right vs wrong so far this session."""
    total = score["correct"] + score["incorrect"]
    return {"correct": score["correct"], "incorrect": score["incorrect"], "total": total}


# Map each tool's name (as Gemini will refer to it) to the function that runs it.
TOOL_FUNCTIONS = {
    "get_flashcard": get_flashcard,
    "check_answer_and_next": check_answer_and_next,
    "get_score": get_score,
}

# Tell Gemini what each tool does and what arguments it takes.
TOOL_DECLARATIONS = [
    {
        "type": "function",
        "name": "get_flashcard",
        "description": (
            "Starts a flashcard quiz on the given topic by returning its first question. "
            "Does not reveal the answer. Only call this once, to start a topic — after that, "
            "check_answer_and_next already returns each following question, so don't call "
            "get_flashcard again mid-quiz."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": f"Topic deck to pull from. One of: {', '.join(FLASHCARDS)}.",
                },
                "num_questions": {
                    "type": "integer",
                    "description": "How many questions the user asked for in this quiz (default 3 if they didn't say).",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "type": "function",
        "name": "check_answer_and_next",
        "description": (
            "Grades the user's answer to the current flashcard question, updates the running "
            "score, and returns the next question from the same topic in the same call. Use "
            "this (not get_flashcard) to advance the quiz after the user answers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_answer": {
                    "type": "string",
                    "description": "The answer the user gave.",
                },
            },
            "required": ["user_answer"],
        },
    },
    {
        "type": "function",
        "name": "get_score",
        "description": "Reports how many questions the user has answered correctly vs incorrectly so far in this session.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---------------------------------------------------------------------------
# 2. THE AGENT LOOP — same Reason -> Act -> Observe shape as agent_loop.py.
# ---------------------------------------------------------------------------

MAX_RATE_LIMIT_RETRIES = 5
RETRY_HINT_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)

# Free tier caps interactions.create() at 5 requests/minute. Rather than
# firing immediately and reactively waiting out a 429 after the fact (which
# is what caused repeated ~50s waits mid-quiz), track our own request
# timestamps and proactively pace calls to stay under that ceiling.
FREE_TIER_REQUESTS_PER_MINUTE = 5
_request_timestamps = deque()


def _throttle_before_request() -> None:
    now = time.time()
    while _request_timestamps and now - _request_timestamps[0] >= 60:
        _request_timestamps.popleft()
    if len(_request_timestamps) >= FREE_TIER_REQUESTS_PER_MINUTE:
        wait_time = 60 - (now - _request_timestamps[0]) + 0.5
        print(f"  pacing requests to stay under the free-tier rate limit — waiting {wait_time:.0f}s...")
        time.sleep(wait_time)
        now = time.time()
    _request_timestamps.append(now)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect a 429/quota-exceeded error from the google-genai SDK.

    The SDK actually raises from google.genai._gaos.lib.compat_errors
    (RateLimitError, etc.) for interactions.create(), an unrelated hierarchy
    from the public google.genai.errors classes — so rather than matching a
    class, check the `.status_code`/`.code` attribute both hierarchies set,
    falling back to sniffing the message text.
    """
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code == 429:
        return True
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


def _rate_limit_wait_seconds(exc: Exception, attempt: int) -> float:
    """Parse the "Please retry in N.NNs" hint out of the error message, adding
    a small buffer. Falls back to a fixed backoff schedule if no hint is found."""
    match = RETRY_HINT_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 1.0
    fallback_schedule = [5, 10, 20, 20, 20]
    return fallback_schedule[min(attempt - 1, len(fallback_schedule) - 1)]


def _format_tool_reply(call_name: str, result: dict):
    """Render a tool result as the user-facing reply directly, instead of
    spending a whole extra API call just to have the model restate the same
    data as prose. Returns None for anything not covered here, which tells
    run_agent_turn to fall back to letting the model phrase it (safe default
    for tools/cases this hasn't been taught to format)."""
    if call_name == "get_flashcard":
        if "error" in result:
            return result["error"]
        return f"**Question {result['question_num']} of {result['num_questions']}:**\n\n{result['question']}"

    if call_name == "check_answer_and_next":
        if "error" in result:
            return result["error"]
        if result.get("quiz_complete"):
            fs = result["final_score"]
            return (
                f"{result['feedback']}\n\n"
                f"That's the quiz! Final score: {fs['correct']} correct, {fs['incorrect']} incorrect "
                f"out of {fs['total']}."
            )
        return (
            f"{result['feedback']}\n\n"
            f"**Question {result['question_num']} of {result['num_questions']}:**\n\n{result['next_question']}"
        )

    if call_name == "get_score":
        return f"You've gotten {result['correct']} correct and {result['incorrect']} incorrect out of {result['total']} so far."

    return None


def run_agent_turn(client, history: list, max_steps: int = 12) -> str:
    """Run one Reason -> Act -> Observe turn against the shared history and
    return the model's reply text for this turn. `history` is mutated in
    place so the conversation (and the quiz) carries over to the next turn."""
    for step_num in range(1, max_steps + 1):
        # --- REASON ----------------------------------------------------
        for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
            _throttle_before_request()
            try:
                interaction = client.interactions.create(
                    model=MODEL,
                    store=False,
                    input=history,
                    tools=TOOL_DECLARATIONS,
                )
                break
            except Exception as exc:
                # Caught broadly because this SDK raises rate-limit errors from
                # an internal, undocumented class hierarchy that isn't a
                # subclass of anything public — see _is_rate_limit_error.
                # Anything that isn't actually a rate limit is re-raised
                # immediately below, so this doesn't swallow real bugs.
                if not _is_rate_limit_error(exc) or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                wait_time = _rate_limit_wait_seconds(exc, attempt)
                print(f"  [step {step_num}] rate limited — retrying in {wait_time:.0f}s...")
                time.sleep(wait_time)

        for step in interaction.steps:
            history.append(step.model_dump())

        tool_calls = [s for s in interaction.steps if s.type == "function_call"]

        if not tool_calls:
            return interaction.output_text

        # --- ACT + OBSERVE ----------------------------------------------
        results_this_step = []
        for call in tool_calls:
            print(f"  [step {step_num}] agent wants to call: {call.name}({call.arguments})")
            fn = TOOL_FUNCTIONS.get(call.name)
            result = fn(**call.arguments) if fn else {"error": f"Unknown tool: {call.name}"}
            print(f"  [step {step_num}] tool result: {result}")
            results_this_step.append((call.name, result))

            history.append({
                "type": "function_result",
                "name": call.name,
                "call_id": call.id,
                "result": [{"type": "text", "text": json.dumps(result)}],
            })

        # If exactly one tool was called and we know how to render its result
        # ourselves, skip the extra API call that would otherwise just ask
        # the model to restate this same data as prose. Anything else (no
        # formatter, or the model chained multiple tool calls) falls through
        # to the next loop iteration, which calls the model as before.
        if len(results_this_step) == 1:
            reply = _format_tool_reply(*results_this_step[0])
            if reply is not None:
                history.append({"type": "model_output", "content": [{"type": "text", "text": reply}]})
                return reply

    return "Agent stopped: hit max_steps without reaching a final answer."


if __name__ == "__main__":
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    history = []

    print("Study Buddy Agent — type 'quit' to exit.\n")
    while True:
        user_message = input("You: ")
        if user_message.strip().lower() in {"quit", "exit"}:
            break

        history.append({"type": "user_input", "content": [{"type": "text", "text": user_message}]})
        reply = run_agent_turn(client, history)
        print(f"\nAgent: {reply}\n")
