# Study Buddy Agent — Presentation Notes

Speaker notes for the Part 4 share-out. Structured so you can talk through it live in the terminal rather than needing slides.

---

## 1. One-sentence description

> An AI agent, built from scratch with the `google-genai` SDK's Interactions API, that quizzes you on flashcards, grades your answers (leniently — typos are OK), and tracks your score across the conversation.

---

## 2. What it actually is (architecture, 30 seconds)

- A single Python file (`study_buddy_agent.py`), no framework.
- One `while True:` REPL loop in `__main__` that keeps a single `history` list alive across turns — this is what lets the agent remember "we're on question 2 of 3" from one message to the next.
- Each turn runs a **Reason → Act → Observe** loop (`run_agent_turn`):
  1. **Reason** — send `history` + tool definitions to Gemini.
  2. **Act** — if Gemini asks for a tool, run the real Python function.
  3. **Observe** — feed the result back into `history`.
  4. Repeat until Gemini has nothing left to call.

Three tools, intentionally lean (not one tool per "concept" — see §4 for why):

| Tool | Does |
|---|---|
| `get_flashcard(topic, num_questions)` | Starts a quiz: picks a topic deck, sets up quiz state, returns question 1 |
| `check_answer_and_next(user_answer)` | Grades the current answer, updates score, returns the *next* question in the same call |
| `get_score()` | Reports the running tally on demand |

Three flashcard decks, chosen as a pun: **JS syntax**, **bugs** (entomology), **debugs** (software debugging).

---

## 3. Live demo script

Run `python3 study_buddy_agent.py`, then:

```
You: Quiz me on bugs for 3 questions and keep score.
```

Walk through what happens step-by-step (the `[step N] agent wants to call: ...` print lines make this visible live — that's the actual point of the demo):

1. Gemini reads the message, decides to call `get_flashcard(topic="bugs", num_questions=3)`.
2. Python runs the real function — draws a random card from a shuffled per-topic deck, no repeats until the deck is exhausted.
3. The result goes back into `history`; Python formats "Question 1 of 3" **itself** (see §5 — no extra API call for this).
4. You answer. Gemini calls `check_answer_and_next("your answer")`.
5. Python fuzzy-matches your answer (exact match, or close-enough for word-like answers — typos and partial words are forgiven), updates score, draws the next card, returns both the grade and question 2 in one tool result.
6. Repeat for question 3; on the last question, the tool returns `quiz_complete: true` with a final score summary instead of another question.

Good follow-up prompts to show flexibility:
- `What's my score so far?` — calls `get_score()` directly, no quiz needed.
- Ask something totally unrelated ("what's 15% of 60?") — shows the agent doesn't force a tool call when one isn't needed.

---

## 4. The engineering story (this is the part worth talking about)

The happy path above is the easy 20%. Most of the actual work was fixing things that broke in realistic use — this is the more interesting material for a technical audience.

### It wasn't a multi-turn conversation at first
The initial version took one `input()`, ran the loop once, and exited — so it could ask "Question 1" but had no way to ever hear an answer. Fixed by pulling `history` out to a persistent list in `__main__` and wrapping the whole thing in a REPL, so state carries across turns instead of resetting every message.

### The rate-limit crash, and a real lesson about SDK internals
Free-tier quota is 5 requests/minute. Hitting it should be recoverable, not a crash — so retry-with-backoff was added, catching `google.genai.errors.ClientError`. It still crashed. Turns out the SDK actually raises `google.genai._gaos.lib.compat_errors.RateLimitError` for this endpoint — a completely unrelated internal class hierarchy with no relationship to the public error types. The fix: catch broadly at that one call site, and use attribute/message sniffing (`.status_code`, "quota", "429") to decide whether to retry or re-raise — because the *public* exception type documented in the SDK wasn't actually the type being thrown. Good story about not trusting a library's public API surface to match its actual runtime behavior.

### Cutting API calls to the theoretical floor
Originally, grading an answer and fetching the next question were two separate tool calls, and *every* tool call also costs a second API round-trip (the model has to see the tool's result before it can phrase a reply). For a 3-question quiz that's ~8-9 API calls — brutal against a 5/minute cap.

Two optimizations, in order:
1. **Merged tools**: `check_answer` + `get_flashcard` became one `check_answer_and_next` call — half the tool calls.
2. **Skipped the redundant "phrase it" call**: since the quiz's replies are entirely formulaic ("Question X of N", "Correct!"), Python now renders them directly and appends a synthetic `model_output` step to history — no second API call needed. This took the app to **exactly 1 API call per user message**, which is the actual floor for this architecture (the model still has to interpret each message).

Net result: a 3-question quiz went from ~8-9 calls down to 4.

### A proactive rate limiter, not just a reactive one
Even at 1 call/turn, a fast quiz can still brush a 5/minute ceiling. Rather than firing immediately and reactively waiting out a 429 after the fact, the agent now tracks its own request timestamps and paces itself to stay under the limit *before* it ever gets throttled by the API.

### Answer grading had to get smarter, carefully
"debug" should count as correct for "debugging" — but blind substring/fuzzy matching is dangerous for short or symbolic answers. `==` is a substring of `===`, but they're different operators; `6` vs `60` would wrongly match too. The fix: leniency (substring + typo-tolerant fuzzy matching) only applies to word-like answers (alphabetic, 4+ characters); short/symbolic/numeric answers require an exact match. Worth mentioning as a "don't over-generalize a fix" example.

### The API key leak (a real security lesson)
A copy-paste mistake put the raw Gemini API key directly into the Python source as a dict key instead of the string `"GEMINI_API_KEY"` — meaning the key ended up in plaintext in a crash traceback. Fixed properly: `.env` file + `python-dotenv` + `.gitignore`, and — the actually important part — treating an exposed key as compromised and rotating it, not just hiding it better next time.

---

## 5. Key takeaways for the group

- An agent loop is genuinely just a `while` loop with an API call and some `if` statements — no magic.
- Tool design matters as much as tool correctness: merging `check_answer` + `get_flashcard`, and adding `num_questions` so the *server* (not the model's memory) tracks quiz progress, made the app both cheaper and more reliable.
- Not every "agent" turn needs the model to talk — sometimes the deterministic Python answer is better and faster than asking an LLM to restate it.
- Treat any secret that ever touched a terminal, a file, or a chat transcript as compromised — rotate it, don't just relocate it.

---

## 6. If asked "what would you do next?"

- Enable billing to remove the free-tier ceiling entirely (deferred by choice — free tier was good enough for a workshop demo).
- More topics/decks, or let users add their own flashcards without editing source.
- A `--topic`/`--questions` CLI flag instead of relying on the model to parse quiz parameters from free text.
