# Build Your Own Agent

You do not need to know Python for this. You need to know _what you want the agent to do_ — your AI coding assistant (Claude, ChatGPT, whatever you normally use) will write the Python. You already know JS, so you'll be able to read most of what it writes; when you hit something unfamiliar, ask your assistant to explain it by comparing it to the JS equivalent.

Today you'll work in **two separate projects**. `agent-workshop` (this repo) is for learning how the loop and tools work — you won't add anything to it. In Part 2, you'll spin up a second, brand-new project for the actual agent you build, with only the tools that agent needs. More on why below.

These instructions assume a Mac, and that you've already cloned this repo into `Documents/dev/agent-workshop`.

---

## Part 1 — Setup

Do these in order. If you get stuck on any step, flag a facilitator rather than debugging alone — we'd rather unblock you in 30 seconds than have you lose 10 minutes.

### 📂 Step 1. Open the project in VS Code

Open VS Code → **File → Open Folder** → select `agent-workshop`.

Then open the integrated terminal: **Terminal → New Terminal**. It opens already rooted inside this project — every command below runs right there, no `cd` needed.

### 🐍 Step 2. Get Python 3.12

Macs ship with an old system Python (3.9) that's too old for the SDK we're using today. Install a current one via Homebrew, in the VS Code terminal:

```bash
brew install python@3.12
```

Confirm it worked:

```bash
python3.12 --version
```

Should print `Python 3.12.x`.

### 🧪 Step 3. Create and activate a virtual environment

This keeps today's packages isolated from anything else on your Mac, and sidesteps a permissions error modern macOS/Homebrew throws if you `pip install` directly.

```bash
python3.12 -m venv venv
source venv/bin/activate
```

Your prompt should now show `(venv)` at the start of the line. **Run that `source` line again every time you open a new terminal window or tab** — it doesn't stick permanently. If a later command can't find `google-genai`, check for `(venv)` first.

### 🔑 Step 4. Get a free Gemini API key

1. Go to **[aistudio.google.com/apikey](https://aistudio.google.com/apikey)**
2. Sign in with a Google account
3. Click **Create API key** — no credit card needed
4. Copy the key somewhere you can find it in the next step

### ⚙️ Step 5. Set the key as an environment variable

With your venv still active, in the same terminal:

```bash
export GEMINI_API_KEY="paste-your-key-here"
```

> Like the venv, this only lasts for the current terminal session. Closing the terminal, or opening a new tab, means running this again.

### 📦 Step 6. Install the SDK

```bash
pip install google-genai
```

(Inside an active venv, plain `pip` — no `3` needed — points at the right Python automatically.)

### 📝 Step 7. Create `agent_loop.py`

In VS Code's file explorer (left sidebar), right-click `agent-workshop` → **New File** → name it `agent_loop.py`. Paste in the code below, and save.

```python
"""
agent_loop.py — A minimal, from-scratch AI agent.

This shows the actual mechanics of an agent loop:
  1. REASON  -> send the conversation so far to Gemini
  2. ACT     -> if Gemini wants to call a tool, run that real Python function
  3. OBSERVE -> send the result back to Gemini as new information
  4. Repeat until Gemini gives a final text answer (or we hit max_steps).

No framework. No magic. Just a while-loop you can read top to bottom.
"""

import json
import os

from google import genai

MODEL = "gemini-3.6-flash"


# ---------------------------------------------------------------------------
# 1. TOOLS — the actions your agent is allowed to take.
#
#    Each tool needs two things:
#      (a) a real Python function that does the work
#      (b) a "declaration" that tells Gemini the tool exists, what it does,
#          and what arguments it takes
# ---------------------------------------------------------------------------

def calculate(expression: str) -> dict:
    """Safely evaluate a basic arithmetic expression, e.g. '85 * 0.20'."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return {"error": "Only digits and + - * / ( ) . are allowed."}
    try:
        # Safe here only because `expression` was just restricted to the
        # character set above — never eval() untrusted input in general.
        return {"result": eval(expression)}
    except Exception as e:
        return {"error": str(e)}


def get_current_time(city: str) -> dict:
    """Return the current time in one of a handful of supported cities."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    city_zones = {
        "tokyo": "Asia/Tokyo",
        "london": "Europe/London",
        "new york": "America/New_York",
        "san francisco": "America/Los_Angeles",
        "sydney": "Australia/Sydney",
    }
    zone = city_zones.get(city.lower())
    if not zone:
        return {"error": f"No timezone for '{city}'. Try: {', '.join(city_zones)}."}
    now = datetime.now(ZoneInfo(zone))
    return {"time": now.strftime("%A, %I:%M %p")}


# --- ADD YOUR OWN TOOLS BELOW THIS LINE -------------------------------------
#
# Ask your AI coding assistant to write a new function here, in the same
# shape as the two above:
#   - a short docstring describing what it does
#   - typed arguments
#   - a dict return value (never raise an exception — return {"error": ...})
#
# Example prompt to give your assistant:
#   "Add a new tool function to agent_loop.py called get_flashcard that takes
#    a topic (string) and returns a random question/answer pair from a small
#    hardcoded dictionary of flashcards about that topic. Follow the same
#    style as calculate() and get_current_time(), including the matching
#    entries in TOOL_FUNCTIONS and TOOL_DECLARATIONS below."
# -----------------------------------------------------------------------------


# Map each tool's name (as Gemini will refer to it) to the function that runs it.
TOOL_FUNCTIONS = {
    "calculate": calculate,
    "get_current_time": get_current_time,
    # "your_new_tool": your_new_tool,
}

# Tell Gemini what each tool does and what arguments it takes.
TOOL_DECLARATIONS = [
    {
        "type": "function",
        "name": "calculate",
        "description": "Evaluates a basic arithmetic expression, e.g. '15 * 0.2'.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression using + - * / ( ) and numbers.",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "type": "function",
        "name": "get_current_time",
        "description": "Gets the current time in a supported city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo' or 'London'.",
                },
            },
            "required": ["city"],
        },
    },
    # --- ADD YOUR OWN TOOL DECLARATIONS BELOW THIS LINE ---------------------
]


# ---------------------------------------------------------------------------
# 2. THE AGENT LOOP — this is the part that makes it an "agent" and not just
#    a single chatbot reply. It keeps going, step by step, until the model
#    is done, executing real code in between each reasoning step.
# ---------------------------------------------------------------------------

def run_agent(user_message: str, max_steps: int = 6) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # `history` is the full transcript so far. We manage it ourselves (in
    # "stateless" mode) so we can see exactly what the agent is doing at
    # every step, instead of letting the API hide it from us.
    history = [
        {"type": "user_input", "content": [{"type": "text", "text": user_message}]}
    ]

    for step_num in range(1, max_steps + 1):
        # --- REASON ----------------------------------------------------
        interaction = client.interactions.create(
            model=MODEL,
            store=False,
            input=history,
            tools=TOOL_DECLARATIONS,
        )

        # Record everything the model just did (its reasoning + any tool calls)
        # so the next request has the full context.
        for step in interaction.steps:
            history.append(step.model_dump())

        tool_calls = [s for s in interaction.steps if s.type == "function_call"]

        if not tool_calls:
            # No tool calls left to make — the model has a final answer.
            return interaction.output_text

        # --- ACT + OBSERVE ----------------------------------------------
        for call in tool_calls:
            print(f"  [step {step_num}] agent wants to call: {call.name}({call.arguments})")
            fn = TOOL_FUNCTIONS.get(call.name)
            result = fn(**call.arguments) if fn else {"error": f"Unknown tool: {call.name}"}
            print(f"  [step {step_num}] tool result: {result}")

            history.append({
                "type": "function_result",
                "name": call.name,
                "call_id": call.id,
                "result": [{"type": "text", "text": json.dumps(result)}],
            })

    return "Agent stopped: hit max_steps without reaching a final answer."


if __name__ == "__main__":
    question = input("Ask your agent something: ")
    answer = run_agent(question)
    print("\nAgent's final answer:")
    print(answer)
```

### 👀 Step 8. Read the code — don't skip this

Before running it, spend some time actually reading `agent_loop.py` top to bottom. You don't need to understand every line — just get the gist:

- Where are the tools defined?
  > The actual work happens in plain Python functions like `calculate()` and `get_current_time()` (lines 104–133). Gemini can't see Python code though, so each tool also needs a "declaration" describing its name, description, and arguments in `TOOL_DECLARATIONS` (lines 161–193) — that's what actually gets sent to the model. `TOOL_FUNCTIONS` (lines 154–158) is the map that connects a declaration's name back to the real function that runs it.
- Where's the actual loop?
  > Inside `run_agent()`, the `for step_num in range(1, max_steps + 1):` loop (lines 212–244). Each pass through it does one REASON step (call the model), then, if the model asked for a tool, one ACT + OBSERVE step (run the Python function and feed the result back into `history`).
- Where does it decide it's done and stop?
  > Right after the REASON step, `tool_calls = [s for s in interaction.steps if s.type == "function_call"]` checks whether the model asked for any tools this round. If `tool_calls` is empty (line 228), there's nothing left to act on, so it returns `interaction.output_text` as the final answer (line 230) and the loop ends. If the model keeps calling tools past `max_steps`, the loop exits on its own and falls through to the fallback message on line 246.

Ask your AI coding assistant to explain any chunk you don't follow — paste a section and ask something like _"explain this to me like I know JavaScript."_ This is the single best habit for today: read first, ask questions, then run it.

**Bonus challenge:** find the line near the bottom that says `input("Ask your agent something: ")` and change what happens when the script launches — e.g., hardcode a specific question instead of asking, or change the prompt text itself. Small, contained edit, but it confirms you understand exactly where user input enters the loop before you build a new agent from scratch in Part 2.

### ▶️ Step 9. Run it

In the terminal (venv still active):

```bash
python agent_loop.py
```

You should see a prompt: `Ask your agent something:`. Try:

```
What's 20% of 85, and what time is it in Tokyo right now?
```

If you see the agent print out `[step 1] agent wants to call: ...` lines and then a final answer, **you're set up correctly.** This is the same scaffold from the live demo.

**If something breaks:** first check for `(venv)` at the start of your prompt — if it's missing, run `source venv/bin/activate` again. Then copy the _entire_ error message (not just the last line) and paste it to your AI coding assistant along with "I'm running this Python script and got this error, what's wrong?" It's usually one of: venv not activated, key not set, or a typo from copy-pasting.

---

## Part 2 — Build Your Own Agent

Real agents don't usually get built by piling every tool into one growing file. A well-built agent has a _lean, specific_ toolset — just what its job needs, nothing extra. Every tool you register gets sent to the model on every single call, whether it's used or not, so an agent carrying tools it doesn't need is slower, pricier, and more likely to misfire.

`agent-workshop` was for learning the mechanics. Now you'll build the real thing — a new, purpose-built agent, in its own project, with an AI coding assistant writing the code from scratch.

### First, start a new project

1. Pick a name for your agent (e.g. `study-buddy-agent`) and create a **new folder outside `agent-workshop`** — e.g. `~/Documents/dev/study-buddy-agent`. Open it in a new VS Code window.
2. Open its integrated terminal and set up a fresh venv, same as before:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install google-genai
   ```
3. Set your key again — environment variables don't cross between projects (or terminal windows):
   ```bash
   export GEMINI_API_KEY="paste-your-key-here"
   ```

### Then, pick your agent idea

### Option A: Study Buddy Agent

An agent that quizzes you on flashcards and tracks how you're doing.

**Tools it needs:**

- `get_flashcard(topic)` — returns a random question/answer pair from a small hardcoded dictionary (you pick the topic and content — could be vocab, historical dates, code syntax, anything)
- `check_answer(question, user_answer)` — compares the user's answer to the correct one and returns feedback (exact match is fine to start)

**Stretch goal:** add a `score` that persists across the conversation (e.g., a global variable the tool functions update) and a tool to report the current score.

**Prompt to try once built:** _"Quiz me on \[your topic\] for 3 questions and keep score."_

---

### Option B: Trip Planner Agent

An agent that helps rough out a simple trip between two cities.

**Tools it needs:**

- `get_distance(city_a, city_b)` — a small hardcoded lookup table of distances between 5-6 cities you choose (no need for a real maps API)
- `suggest_packing_list(weather)` — takes a category like `"hot"`, `"cold"`, or `"rainy"` and returns a list of suggested items

**Stretch goal:** chain the two tools in one question — e.g., "How far is it from Chicago to Denver, and what should I pack if it's cold?" — and watch the agent call both tools before answering.

**Prompt to try once built:** _"How far is it from \[city\] to \[city\], and what should I pack if it's \[weather\]?"_

---

### Option C: Build Your Own Idea

Invent your own concept. To keep it scoped for the time you have, your agent must satisfy:

- **At least one custom tool** built specifically for this agent's job
- **The agent needs 2+ loop iterations to answer at least one good test question** (i.e., it should call a tool, then either call another tool or need the result before it can respond — not just answer from the first message)
- **You've tested it with at least 3 different questions**, including one that shouldn't need any tools at all (to confirm the agent still answers directly when it should)

Some ideas if you want a starting point: a recipe/pantry agent, a D&D dice-roller/character-sheet agent, a unit-converter-plus-currency agent, a "what should I watch tonight" agent with a small hardcoded movie list.

### Now, have your coding agent build it

Tell your AI coding assistant what you want the agent to do, and that you're using the `google-genai` Python SDK's Interactions API — point it at `agent_loop.py` in `agent-workshop` as the loop pattern to follow. You don't need to spec out the tools yourself; talk through what the agent needs and let your assistant help you figure out the right ones.

Something like:

> _"I want to build an agent that quizzes me on flashcards and tracks my score. I'm using the google-genai Python SDK's Interactions API — follow the same Reason/Act/Observe loop as `agent_loop.py` in my `agent-workshop` repo. What tools do you think it needs?" or you can tell it what tools you want if you already know_

Then run it the same way as before, from inside this new project's terminal:

```bash
python study_buddy_agent.py
```

---

## Part 3 — Working With Your AI Coding Assistant

You're directing, not typing every line yourself. A few things that make this go smoothly:

1. **Describe the tools you want, and point at the pattern.** Reference `agent-workshop`'s loop structure explicitly rather than assuming your assistant will infer it — spell out the Reason → Act → Observe shape, or just say "match the loop in agent_loop.py from the agent-workshop repo."
2. **Paste back the whole error**, not just the last line. Python error messages ("tracebacks") read bottom-to-top and the real cause is often several lines up — your assistant needs the whole thing.
3. **Ask for JS comparisons when syntax is unfamiliar.** Python dictionaries ≈ JS objects. List comprehensions ≈ `.map()`/`.filter()`. `**kwargs` ≈ spreading an object into named params. Just ask: _"explain this like I know JavaScript."_
4. **Change one thing, test, then change the next thing.** Don't ask for five tools at once — add one, run it, confirm it works, then move to the next. Small steps are much easier to debug than a pile of new code all at once.
5. **If the agent loops forever or gives a weird answer**, that's not necessarily a bug — it might be your tool's `description` being unclear to the model. Try rewording it to be more specific about what it does and when to use it.

---

## Part 4 — Share

Be ready to show the group:

- What your agent does (one sentence)
- One question you asked it, and what it did step-by-step to answer

No slides needed — just run it live in your terminal.
