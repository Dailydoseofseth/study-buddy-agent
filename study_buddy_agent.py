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

# Tried in order. gemini-3.6-flash is preferred; gemini-3.1-flash-lite has its
# own separate free-tier quota bucket, so it's a real fallback (not just a
# retry) once the primary model's quota is actually exhausted for the day —
# not a transient per-minute rate limit, which the retry loop already handles.
MODEL_CHAIN = ["gemini-3.6-flash", "gemini-3.1-flash-lite"]
_active_model_index = 0


def _current_model() -> str:
    return MODEL_CHAIN[_active_model_index]


def _advance_to_next_model() -> bool:
    """Move to the next model in MODEL_CHAIN after the current one's quota is
    exhausted. Returns False if there's no fallback left to try. Clears the
    request-timestamp pacer since RPM limits are tracked per-model too — the
    fresh model hasn't made any requests yet."""
    global _active_model_index
    if _active_model_index + 1 >= len(MODEL_CHAIN):
        return False
    _active_model_index += 1
    _request_timestamps.clear()
    print(f"  {MODEL_CHAIN[_active_model_index - 1]}'s quota is exhausted — falling back to {MODEL_CHAIN[_active_model_index]}.")
    return True


# ---------------------------------------------------------------------------
# 1. TOOLS — the actions this agent is allowed to take.
# ---------------------------------------------------------------------------

# Each topic is split into easy/medium/hard tiers. get_flashcard defaults to
# "easy" when the user doesn't ask for a tier by name.
FLASHCARDS = {
    "js syntax": {
        "easy": [
            {"question": "What keyword declares a block-scoped variable that can be reassigned?", "answer": "let", "hint": "Reassignable, block-scoped — introduced in ES6 alongside const.", "emoji": "♻️"},
            {"question": "What keyword declares a variable that cannot be reassigned?", "answer": "const", "hint": "Block-scoped like let, but the binding can't be reassigned.", "emoji": "🔒"},
            {"question": "What operator returns a string naming a value's type?", "answer": "typeof", "hint": "Try it in a console: typeof 42 tells you the type as a string.", "emoji": "🏷️"},
        ],
        "medium": [
            {"question": "What array method creates a new array by transforming every element?", "answer": "map", "hint": "Returns a brand-new array — the original is left untouched.", "emoji": "🗺️"},
            {"question": "What operator checks equality without type coercion?", "answer": "===", "hint": "Compares both value and type — no coercion allowed.", "emoji": "⚖️"},
            {"question": "What array method returns a new array containing only elements that pass a test?", "answer": "filter", "hint": "Pairs naturally with map — this one removes, map transforms.", "emoji": "🧺"},
        ],
        "hard": [
            {"question": "What symbol wraps a template literal string?", "answer": "`", "hint": "Lets you embed ${expressions} directly inside a string.", "emoji": "🧵"},
            {"question": "What do you call a function that remembers variables from its enclosing scope even after that scope has finished executing?", "answer": "closure", "hint": "This is why a counter function can keep incrementing a private variable between calls.", "emoji": "🔐"},
            {"question": "What JavaScript behavior moves variable and function declarations to the top of their scope before code runs?", "answer": "hoisting", "hint": "This is why you can call a function before its declaration appears in the file.", "emoji": "🎈"},
        ],
    },
    "bugs": {
        "easy": [
            {"question": "How many legs does an insect have?", "answer": "6", "hint": "Count the legs on a housefly or an ant — that's the number.", "emoji": "🐜"},
            {"question": "How many wings does a typical housefly have?", "answer": "2", "hint": "Flies belong to the order Diptera — 'di-' means two.", "emoji": "🪰", "image_url": "/images/bugs/housefly.jpg"},
            {"question": "What do you call a butterfly in its larval stage?", "answer": "caterpillar", "hint": "This is what a butterfly looks like before it grows wings.", "emoji": "🐛"},
        ],
        "medium": [
            {"question": "What is the largest order of insects, containing beetles?", "answer": "Coleoptera", "hint": "This order includes ladybugs and fireflies — both beetles.", "emoji": "🐞"},
            {"question": "What is the term for the transformation an insect undergoes from larva to adult?", "answer": "metamorphosis", "hint": "Butterflies undergo a 'complete' one; grasshoppers undergo an 'incomplete' one.", "emoji": "🦋"},
            {"question": "What is the hard external covering that supports and protects an insect's body called?", "answer": "exoskeleton", "hint": "Insects wear their support structure on the outside, not the inside.", "emoji": "🪲"},
        ],
        "hard": [
            {"question": "What is the scientific study of insects called?", "answer": "entomology", "hint": "The prefix 'ento-' comes from Greek for 'insect'.", "emoji": "🔬", "image_url": "/images/bugs/entomology.jpg"},
            {"question": "What is the scientific order name for butterflies and moths?", "answer": "Lepidoptera", "hint": "This Greek-derived name literally means 'scale wing'.", "emoji": "🦋"},
            {"question": "What is the middle body segment of an insect called, the one bearing legs and wings?", "answer": "thorax", "hint": "An insect's body has 3 segments: head, this one, and the abdomen.", "emoji": "🐝"},
        ],
    },
    "debugs": {
        "easy": [
            {"question": "What is the common term for pausing code execution at a specific line to inspect state?", "answer": "breakpoint", "hint": "You set this in your IDE to pause execution mid-run.", "emoji": "⏸️"},
            {"question": "What tool lets you step through code line-by-line to inspect variables?", "answer": "debugger", "hint": "Chrome DevTools has a panel named exactly this.", "emoji": "🔧"},
            {"question": "What is the process of finding and fixing bugs in code called?", "answer": "debugging", "hint": "It's literally the '-ing' form of removing bugs from code.", "emoji": "🩹"},
        ],
        "medium": [
            {"question": "What do you call an error that occurs while the program is running, not at compile time?", "answer": "runtime error", "hint": "Contrast this with a 'syntax error', which happens before the code even runs.", "emoji": "💥"},
            {"question": "What browser feature lets you inspect the DOM and console errors?", "answer": "DevTools", "hint": "Right-click any webpage and choose 'Inspect' to open this.", "emoji": "🧰"},
            {"question": "What do you call the ordered list of function calls shown when an error is thrown, tracing back to where it started?", "answer": "stack trace", "hint": "This is what gets printed below an uncaught exception in the console.", "emoji": "📚"},
        ],
        "hard": [
            {"question": "What term describes a bug that disappears or changes behavior when you try to observe it, like adding a console.log makes it go away?", "answer": "heisenbug", "hint": "Named after the physicist behind the uncertainty principle — observing it changes it.", "emoji": "👻"},
            {"question": "What is the term for a bug caused by incorrect assumptions about the order of asynchronous operations?", "answer": "race condition", "hint": "Two async calls updating the same variable can produce different results depending on which finishes first.", "emoji": "🏁"},
            {"question": "What is the general term for a bug that only occurs intermittently, making it hard to reproduce consistently?", "answer": "flaky", "hint": "Testers use this word for a test that passes sometimes and fails other times with no code changes.", "emoji": "🎲"},
        ],
    },
}

DIFFICULTIES = ("easy", "medium", "hard")

# Special topic keys that draw from every real topic combined, instead of one deck.
MIXED_TOPIC_KEYS = ("mixed", "all")

# User-authored cards persist here across restarts, merged into FLASHCARDS
# at import time so they're indistinguishable from the built-in deck once loaded.
CUSTOM_CARDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_flashcards.json")


def _load_custom_cards() -> None:
    if not os.path.exists(CUSTOM_CARDS_PATH):
        return
    with open(CUSTOM_CARDS_PATH) as f:
        saved = json.load(f)
    for topic_key, tiers in saved.items():
        deck = FLASHCARDS.setdefault(topic_key, {tier: [] for tier in DIFFICULTIES})
        for tier, cards in tiers.items():
            deck.setdefault(tier, []).extend(cards)


_load_custom_cards()

# Stretch goal: score persists across the conversation for as long as this
# process keeps running (a plain global dict the tool functions update).
score = {"correct": 0, "incorrect": 0}

# Per-topic correct/incorrect counts, used to auto-pick a difficulty tier
# (see _suggest_difficulty) when the user starts a quiz without naming one.
topic_stats = {}

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

# Whether the most recently graded answer was correct — a one-shot flag
# consumed by get_last_answer_correct() so a sound only fires for the turn
# that actually graded an answer, not for unrelated turns (e.g. "what's my score").
last_answer_correct = None

# The question+answer most recently graded, kept around after current_question
# moves on to the next card, so explain_answer can still refer back to it if
# the user asks "why?" right after seeing their result.
last_graded_question = None

# How many questions the current quiz was asked for, and how far into it we
# are — tracked here (not left to the model's memory) so Python itself can
# render "Question 2 of 3" and detect quiz-complete, which is what lets the
# agent loop skip asking the model to phrase that text on every turn.
# start_correct/start_incorrect snapshot the lifetime score at quiz start, so
# quiz_complete can report this quiz's own tally (a delta) instead of the
# running lifetime total, which would include points from earlier quizzes.
quiz_state = {
    "num_questions": 0,
    "question_num": 0,
    "start_correct": 0,
    "start_incorrect": 0,
    "multiple_choice": False,
    "topic_key": "",
}


def _build_choices(topic_key: str, correct_answer: str) -> list:
    """Pick up to 3 wrong-answer distractors from elsewhere in the topic
    (any tier) plus the correct answer, shuffled into a 4-option list."""
    other_answers = {
        card["answer"]
        for tier_cards in FLASHCARDS[topic_key].values()
        for card in tier_cards
        if card["answer"] != correct_answer
    }
    distractors = random.sample(list(other_answers), k=min(3, len(other_answers)))
    choices = distractors + [correct_answer]
    random.shuffle(choices)
    return choices


def _draw_next_card(topic_key: str, difficulty: str, multiple_choice: bool = False) -> str:
    """Pop the next card off the topic+difficulty's shuffled draw pile
    (reshuffling a fresh one if empty) and record it as the current question.
    topic_key may be a real topic, or one of MIXED_TOPIC_KEYS to draw from
    every topic combined — each card in that pile keeps its own real topic
    (tagged at pile-build time) for scoring and distractor selection."""
    global current_question
    pile_key = f"{topic_key}:{difficulty}"
    pile = _draw_piles.get(pile_key)
    if not pile:
        if topic_key in MIXED_TOPIC_KEYS:
            pile = [
                {**c, "topic": t}
                for t, tiers in FLASHCARDS.items()
                for c in tiers.get(difficulty, [])
            ]
        else:
            pile = [{**c, "topic": topic_key} for c in FLASHCARDS[topic_key][difficulty]]
        random.shuffle(pile)
        _draw_piles[pile_key] = pile

    card = pile.pop()
    current_question = {**card, "difficulty": difficulty}
    if multiple_choice:
        current_question["choices"] = _build_choices(card["topic"], card["answer"])
    return card["question"]


def _suggest_difficulty(topic_key: str) -> str:
    """Auto-pick a tier from the user's accuracy in this topic so far: stay
    on easy until there's enough of a track record (3 attempts), then scale
    up for strong accuracy and back down for a weak one."""
    stats = topic_stats.get(topic_key, {"correct": 0, "incorrect": 0})
    attempts = stats["correct"] + stats["incorrect"]
    if attempts < 3:
        return "easy"
    accuracy = stats["correct"] / attempts
    if accuracy >= 0.8:
        return "hard"
    if accuracy >= 0.5:
        return "medium"
    return "easy"


def get_flashcard(topic: str, num_questions: int = 3, difficulty: str = None, multiple_choice: bool = False) -> dict:
    """Start a quiz on the given topic by returning its first question (no answer).
    Only call this to START a topic — after grading an answer, check_answer_and_next
    already returns the next question, so don't call this again mid-quiz."""
    topic_key = topic.strip().lower()
    if topic_key not in FLASHCARDS and topic_key not in MIXED_TOPIC_KEYS:
        return {"error": f"No flashcards for '{topic}'. Try: {', '.join(FLASHCARDS)}, or 'mixed' for all topics."}

    if difficulty:
        difficulty_key = difficulty.strip().lower()
        if difficulty_key not in DIFFICULTIES:
            return {"error": f"'{difficulty}' isn't a difficulty tier. Try: {', '.join(DIFFICULTIES)}."}
    else:
        difficulty_key = _suggest_difficulty(topic_key)

    quiz_state["num_questions"] = max(1, num_questions)
    quiz_state["question_num"] = 1
    quiz_state["start_correct"] = score["correct"]
    quiz_state["start_incorrect"] = score["incorrect"]
    quiz_state["multiple_choice"] = bool(multiple_choice)
    quiz_state["topic_key"] = topic_key
    result = {
        "question": _draw_next_card(topic_key, difficulty_key, quiz_state["multiple_choice"]),
        "question_num": quiz_state["question_num"],
        "num_questions": quiz_state["num_questions"],
        "difficulty": difficulty_key,
        "topic": current_question["topic"],
    }
    if quiz_state["multiple_choice"]:
        result["choices"] = current_question["choices"]
    return result


def add_flashcard(topic: str, question: str, answer: str, hint: str = "", difficulty: str = "easy") -> dict:
    """Add a new flashcard, creating the topic if it doesn't already exist.
    Persisted to disk immediately so it's still there next time the app starts."""
    topic_key = topic.strip().lower()
    if topic_key in MIXED_TOPIC_KEYS:
        return {"error": f"'{topic}' is a reserved keyword for quizzing across all topics, not a topic itself."}
    difficulty_key = (difficulty or "easy").strip().lower()
    if difficulty_key not in DIFFICULTIES:
        return {"error": f"'{difficulty}' isn't a difficulty tier. Try: {', '.join(DIFFICULTIES)}."}
    if not question.strip() or not answer.strip():
        return {"error": "Both question and answer are required."}

    card = {
        "question": question.strip(),
        "answer": answer.strip(),
        "hint": hint.strip() or "No hint available for this one.",
    }
    FLASHCARDS.setdefault(topic_key, {tier: [] for tier in DIFFICULTIES})[difficulty_key].append(card)
    _persist_custom_card(topic_key, difficulty_key, card)
    return {"added": True, "topic": topic_key, "difficulty": difficulty_key, "question": card["question"]}


def _persist_custom_card(topic_key: str, difficulty_key: str, card: dict) -> None:
    saved = {}
    if os.path.exists(CUSTOM_CARDS_PATH):
        with open(CUSTOM_CARDS_PATH) as f:
            saved = json.load(f)
    saved.setdefault(topic_key, {}).setdefault(difficulty_key, []).append(card)
    with open(CUSTOM_CARDS_PATH, "w") as f:
        json.dump(saved, f, indent=2)


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


def _resolve_choice_letter(user_answer: str, choices: list | None) -> str:
    """In multiple-choice mode, let 'A'/'b)'/'C.' resolve to that option's
    full text before grading — covers a user typing the letter instead of
    clicking the button."""
    if not choices:
        return user_answer
    letter = user_answer.strip().rstrip(".):").upper()
    if len(letter) == 1 and letter.isalpha():
        index = ord(letter) - ord("A")
        if 0 <= index < len(choices):
            return choices[index]
    return user_answer


def check_answer_and_next(user_answer: str) -> dict:
    """Grade the user's answer to the current flashcard, update the score, and
    immediately return the next question from the same topic — combined into
    one tool call (instead of a separate check_answer + get_flashcard) so a
    3-question quiz costs ~3-4 API round-trips instead of ~6-8, which matters
    a lot on the free tier's 5-requests/minute cap."""
    global current_question, last_answer_correct, last_graded_question
    if current_question is None:
        return {"error": "No active question — call get_flashcard first."}

    correct_answer = current_question["answer"]
    topic_key = current_question["topic"]
    difficulty_key = current_question["difficulty"]
    user_answer = _resolve_choice_letter(user_answer, current_question.get("choices"))
    is_correct = _is_close_enough(user_answer, correct_answer)
    last_answer_correct = is_correct
    last_graded_question = {
        "question": current_question["question"],
        "correct_answer": correct_answer,
        "topic": topic_key,
    }
    topic_tally = topic_stats.setdefault(topic_key, {"correct": 0, "incorrect": 0})
    if is_correct:
        score["correct"] += 1
        topic_tally["correct"] += 1
    else:
        score["incorrect"] += 1
        topic_tally["incorrect"] += 1

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
        current_question = None
    else:
        # Draw from the session's chosen topic_key (may be "mixed"), not this
        # card's own real topic — that's only for scoring/stats attribution.
        result["next_question"] = _draw_next_card(quiz_state["topic_key"], difficulty_key, quiz_state["multiple_choice"])
        result["question_num"] = quiz_state["question_num"]
        result["num_questions"] = quiz_state["num_questions"]
        result["difficulty"] = difficulty_key
        result["topic"] = current_question["topic"]
        if quiz_state["multiple_choice"]:
            result["choices"] = current_question["choices"]

    return result


def get_score() -> dict:
    """Report how many questions the user has gotten right vs wrong so far this session."""
    total = score["correct"] + score["incorrect"]
    return {"correct": score["correct"], "incorrect": score["incorrect"], "total": total}


def explain_answer() -> dict:
    """Return the most recently graded question, its correct answer, and its
    topic — not an explanation itself. The model is expected to use its own
    knowledge to compose a short teaching explanation from these facts,
    which is why this deliberately has no entry in _format_tool_reply: the
    whole point is letting the model phrase it, not returning canned text."""
    if last_graded_question is None:
        return {"error": "No graded question yet to explain — answer a question first."}
    return dict(last_graded_question)


def get_current_hint() -> str | None:
    """Return the hint for whichever flashcard question is currently active,
    or None if there isn't one (no quiz running, or the quiz just ended).
    Not a model tool — server.py calls this directly so the frontend can
    render a hint alongside the question without spending an extra API call."""
    return current_question["hint"] if current_question else None


def get_current_choices() -> list | None:
    """Return the multiple-choice options for the currently active flashcard
    question, or None if the quiz isn't in multiple-choice mode (or there's
    no active question). Not a model tool — same side-channel as
    get_current_hint, so the frontend can render answer buttons."""
    return current_question.get("choices") if current_question else None


def get_current_visual() -> dict | None:
    """Return the visual for whichever flashcard question is currently
    active: a real photo (image_url) where one exists (the bugs deck), or
    an emoji glyph as the fallback/default for every other card. None if
    there's no active question. Same no-extra-API-call side channel as hint."""
    if not current_question:
        return None
    return {
        "emoji": current_question.get("emoji"),
        "image_url": current_question.get("image_url"),
    }


def get_last_answer_correct() -> bool | None:
    """Consume (return-then-clear) whether the most recently graded answer
    was correct. Returns None on any turn that didn't just grade an answer,
    so the frontend only plays a correct/incorrect sound on the right turn."""
    global last_answer_correct
    value = last_answer_correct
    last_answer_correct = None
    return value


# Map each tool's name (as Gemini will refer to it) to the function that runs it.
TOOL_FUNCTIONS = {
    "get_flashcard": get_flashcard,
    "check_answer_and_next": check_answer_and_next,
    "get_score": get_score,
    "explain_answer": explain_answer,
    "add_flashcard": add_flashcard,
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
                    "description": f"Topic deck to pull from. One of: {', '.join(FLASHCARDS)}. Pass 'mixed' if the user wants questions drawn from every topic in one quiz.",
                },
                "num_questions": {
                    "type": "integer",
                    "description": "How many questions the user asked for in this quiz (default 3 if they didn't say).",
                },
                "difficulty": {
                    "type": "string",
                    "enum": list(DIFFICULTIES),
                    "description": "Difficulty tier to draw from. Only pass this if the user names a tier themselves (e.g. 'quiz me on hard bugs questions'). Otherwise omit it entirely — the app auto-picks a tier based on the user's accuracy in this topic so far.",
                },
                "multiple_choice": {
                    "type": "boolean",
                    "description": "Set true if the user wants multiple-choice options instead of typing a free-text answer (e.g. they say 'multiple choice' or 'give me options'). Default false.",
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
    {
        "type": "function",
        "name": "explain_answer",
        "description": (
            "Call this when the user asks why an answer is correct, asks for more detail after "
            "seeing a graded result, or otherwise wants the most recently graded question "
            "explained (e.g. 'why?', 'explain that', 'I don't get it'). Returns the question, "
            "its correct answer, and its topic — then compose a short (1-3 sentence) teaching "
            "explanation yourself using that context and your own knowledge of the topic."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "function",
        "name": "add_flashcard",
        "description": (
            "Adds a new flashcard the user dictates, e.g. 'add a flashcard to bugs: what's the "
            "insect stage after pupa? answer: adult'. Creates the topic if it doesn't exist yet. "
            "Persists to disk, so it's available in future sessions too."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic deck to add this card to (existing or brand new).",
                },
                "question": {
                    "type": "string",
                    "description": "The flashcard's question text.",
                },
                "answer": {
                    "type": "string",
                    "description": "The correct answer.",
                },
                "hint": {
                    "type": "string",
                    "description": "A short hint for this question. Make one up if the user didn't give one.",
                },
                "difficulty": {
                    "type": "string",
                    "enum": list(DIFFICULTIES),
                    "description": "Difficulty tier for this card. Default to 'easy' unless the user says otherwise.",
                },
            },
            "required": ["topic", "question", "answer"],
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


def _is_quota_exhausted_error(exc: Exception) -> bool:
    """Detect a persistent daily-quota exhaustion (as opposed to a transient
    per-minute burst) from the error text — e.g. "generate_content_free_tier_
    requests" or "plan and billing". This kind of 429 includes a "retry in
    Ns" hint too, but it's misleading: the real reset is midnight Pacific, so
    waiting out that hint and retrying the same model is pointless. Falling
    back to the next model immediately is faster and actually works."""
    text = str(exc).lower()
    return "free_tier_requests" in text or "plan and billing" in text


def _rate_limit_wait_seconds(exc: Exception, attempt: int) -> float:
    """Parse the "Please retry in N.NNs" hint out of the error message, adding
    a small buffer. Falls back to a fixed backoff schedule if no hint is found."""
    match = RETRY_HINT_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 1.0
    fallback_schedule = [5, 10, 20, 20, 20]
    return fallback_schedule[min(attempt - 1, len(fallback_schedule) - 1)]


def _format_question_header(result: dict) -> str:
    """Render the '**Question X of Y (difficulty)**' header, adding the
    card's own topic in mixed mode since the deck varies question to question."""
    header = f"**Question {result['question_num']} of {result['num_questions']} ({result['difficulty']})"
    if quiz_state["topic_key"] in MIXED_TOPIC_KEYS:
        header += f" — {result['topic']}"
    return header + ":**"


def _format_choices(choices: list | None) -> str:
    """Render multiple-choice options as a lettered list, prefixed with a
    blank line — empty string (no-op) when there are no choices."""
    if not choices:
        return ""
    letters = "ABCD"
    lines = "\n".join(f"{letters[i]}) {choice}" for i, choice in enumerate(choices))
    return f"\n\n{lines}"


def _format_tool_reply(call_name: str, result: dict):
    """Render a tool result as the user-facing reply directly, instead of
    spending a whole extra API call just to have the model restate the same
    data as prose. Returns None for anything not covered here, which tells
    run_agent_turn to fall back to letting the model phrase it (safe default
    for tools/cases this hasn't been taught to format)."""
    if call_name == "get_flashcard":
        if "error" in result:
            return result["error"]
        return (
            f"{_format_question_header(result)}\n\n{result['question']}"
            f"{_format_choices(result.get('choices'))}"
        )

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
            f"{_format_question_header(result)}\n\n{result['next_question']}"
            f"{_format_choices(result.get('choices'))}"
        )

    if call_name == "get_score":
        return f"You've gotten {result['correct']} correct and {result['incorrect']} incorrect out of {result['total']} so far."

    if call_name == "add_flashcard":
        if "error" in result:
            return result["error"]
        return f"Added to **{result['topic']}** ({result['difficulty']}): \"{result['question']}\""

    return None


def run_agent_turn(client, history: list, max_steps: int = 12) -> str:
    """Run one Reason -> Act -> Observe turn against the shared history and
    return the model's reply text for this turn. `history` is mutated in
    place so the conversation (and the quiz) carries over to the next turn."""
    for step_num in range(1, max_steps + 1):
        # --- REASON ----------------------------------------------------
        interaction = None
        while interaction is None:
            model = _current_model()
            for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
                _throttle_before_request()
                try:
                    interaction = client.interactions.create(
                        model=model,
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
                    if _is_quota_exhausted_error(exc):
                        # A daily-quota 429 won't clear by waiting out its
                        # (misleading) "retry in Ns" hint, so skip straight
                        # to the next model instead of burning retries here.
                        if not _advance_to_next_model():
                            raise
                        break
                    if not _is_rate_limit_error(exc):
                        raise
                    if attempt == MAX_RATE_LIMIT_RETRIES:
                        if not _advance_to_next_model():
                            raise
                        break
                    wait_time = _rate_limit_wait_seconds(exc, attempt)
                    print(f"  [step {step_num}] {model} rate limited — retrying in {wait_time:.0f}s...")
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
