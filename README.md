# ⭐ NorthStar

**One Mentor. Every Student's Path.**

NorthStar is an AI mentor for school and college students. It learns about the student, turns one goal into milestones, plans a realistic week, collects quick daily check-ins, and every week reviews what happened and suggests changes, which the student decides on.

It is built **for the student**: no parent monitoring, no leaderboards, no diagnosis, no doing graded work, and nothing changes without the student's OK.

Built with Python, LangChain, LangGraph, SQLAlchemy (SQLite) and Streamlit.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
cp .env.example .env                                           # then put your API key in .env
.venv/Scripts/python.exe -m northstar.db.seed --demo           # optional: load the demo student "Aarav"
.venv/Scripts/python.exe -m streamlit run app.py
```

Open http://localhost:8501.

### Configuration (`.env`)

| Variable | Example | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | *(secret)* | Server-side credential required for AI features; OpenRouter is the only supported LLM endpoint |
| `DATABASE_URL` | *(optional)* | Defaults to `data/northstar.db` (SQLite). Use a `postgresql+psycopg://...` URL for Postgres |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | *(optional)* | Kaggle API credentials to import curated datasets in Research Studio |
| `KAGGLE_MAX_DATASET_SIZE_MB` | `50.0` | *(optional)* Maximum size in MB allowed for a downloaded dataset |
| `KAGGLE_STUDENT_STORAGE_CAP_MB` | `100.0` | *(optional)* Total disk quota in MB per student across project workspaces |
| `DATASET_WORKSPACE_ROOT` | `data/workspaces` | *(optional)* Storage directory for imported datasets |

Without an OpenRouter key the app still runs: forms, plans, check-ins and reviews you already have all work, and AI actions explain what's missing. The key is read from server environment variables or Streamlit secrets and is never saved with chat state. The sidebar offers only the OpenRouter free router and current catalog entries with verified zero pricing. Automatic model fallback is off; users can select another free model and retry without losing chat state.

---

## What the student can do

| Page | What happens |
|---|---|
| **Home (Today)** | Tick off today's tasks, see what's coming up, this week's progress, planned vs actual minutes, and the last review's win |
| **Onboarding** | Chat with NorthStar (or fill in a form) to build a profile: subjects, interests, strengths, challenges, free time, activities, exams |
| **Goal & Roadmap** | Set one goal → AI suggests 4–8 milestones with checkable "done when…" criteria → edit and confirm |
| **This Week** | AI proposes tasks; the scheduler fits them into each day's free time; preview, then save |
| **Daily Check-in** | Under a minute: did you do it? how long? too easy / just right / too hard, enjoyment, stress, blockers, a link to your work |
| **Weekly Review** | The week's numbers and detected patterns, a kind AI review, and up to 3 suggested changes to **Accept** or **Dismiss** |
| **Explore** | Interest signals (profile, check-ins, your own words, past experiments) → 3–5 specific niches that cite them → a free 3–14 day curiosity experiment → a reflection that feeds back into your interests |
| **Opportunities** | Competitions, programmes and courses from a curated list. Python checks age and deadlines; the AI explains the best fits; every fact (eligibility, deadline, cost, time) comes from the data. Save or dismiss; "Open official page" - never a register button |
| **Research Studio** | A guided 8-step mini-project (question → background → hypothesis → ethical method → data → analysis → artifact → limitations). You write; the coach asks questions. In the **Data** section, discover curated Kaggle datasets, import them into your workspace, run deterministic pandas starter EDA, and attach them as evidence cited as `[S#]`. Sources also come from a curated library or your own links; the downloadable report is built only from your notes |

---

## How it works

**The LLM reasons and writes; Python decides and guarantees.**

```
Streamlit pages
     │
     ├── graph.run(intent) ── LangGraph StateGraph ──┬── weekly_review node → Progress Agent
     │                        (plain-Python router)  └── plan_week node    → Schedule Agent
     ├── Profile Agent (create_agent tool-calling loop)
     └── Roadmap Agent (one structured-output call)
                 │
     services/ (plain Python: scheduler, patterns, progress, goals, review)
                 │
     SQLAlchemy models → SQLite / Postgres
```

| Job | Done by | Why |
|---|---|---|
| Interview the student | **Profile Agent**: a tool-calling loop (`get_student_profile`, `get_missing_profile_fields`, `update_student_profile`) | Open-ended conversation needs an agent; tools are the only way it can touch data, and it can't choose which student |
| Goal → milestones | **Roadmap Agent**: one structured-output call | Python checks count, order and dates, and sends problems back for one retry |
| What to do this week | **Schedule Agent**: one structured-output call | Proposes tasks (what, how long, how hard), never days |
| Which day each task goes on | `services/scheduler.py` | No day over capacity, 20% rest buffer, required before optional, ≤2 per area per day, milestone order kept, overflow reported |
| Spot patterns | `services/patterns.py` | too easy / too hard / skipping / overloaded / losing interest / strong interest / bad week: named thresholds, with evidence |
| Weekly review | **Progress Agent**: one structured-output call | Explains the patterns kindly; picks changes only from a fixed menu |
| Apply a change | `services/review.decide_change`, called **only** by the Accept button | Deliberately not an AI tool. Accepted changes become rules the next plan must follow (lighter load, shorter tasks, easier steps), enforced in Python |
| Suggest niches | **Niche Mentor**: one structured-output call (graph node `suggest_niches`) | Every niche must cite real interest-signal ids (Python checks); the page shows the cited signals from the database, not AI text. Experiments must be free, safe, 3–14 days |
| Coach a research step | **Research Coach**: one structured-output call (graph node `coach_step`) | Feedback + 1–3 questions + one next action. Python rejects links and citations of sources the project doesn't have. It never writes the student's work |
| Match opportunities | `services/opportunities.py` + `data/opportunities.json`, then the **Opportunity Agent** (graph node `match_opportunities`) | Python filters by age, deadline and cost and ranks by interests; the AI only writes "why it fits" and never sees costs or deadlines. Location is never used - NorthStar doesn't store it |
| Sources, citations, report | `services/research.py` + `data/sources.json` | No AI. Sources come only from the curated library or the student; numbers never change; the report removes citations that aren't stored sources and flags unknown links; the method step needs the ethics checklist |
| Datasets & starter EDA | `services/kaggle_dataset_service.py` + `data/kaggle_catalog.json`, then the **Dataset Agent** (graph node `recommend_datasets`) | Python filters and ranks approved datasets; LLM selects top fits with match reasons; deterministic pandas templates run starter EDA (head, missing values, describe, counts). Enforces zip-slip/zip-bomb protection and storage quotas |

### Safety and privacy by design

- **Student in control:** the AI only *proposes*; milestones, plans and plan changes are saved only when the student clicks.
- **No guilt:** daily replies are rule-based and tested against guilt words; reviews are instructed to describe, not judge.
- **Boundaries in every prompt** (`northstar/prompts/shared_safety.md`): no diagnosis, no medical/legal/financial advice, no graded work, no invented sources, no registering for anything.
- **Privacy:** nicknames only; no location, contacts or browsing data; the nickname isn't sent to the LLM when planning or reviewing.
- **Curated datasets only:** Students cannot pass free-text queries or arbitrary download URLs to the Kaggle API. Downloads are restricted to the curated catalog, verified for zip traversal (zip-slip) and zip bombs, size-capped, and analyzed using deterministic pandas templates instead of arbitrary AI code.

---

## Project structure

```
app.py                     Home: today's checklist + dashboard + demo tools
pages/                     Onboarding, Goal & Roadmap, This Week, Daily Check-in, Weekly Review, Explore, Research, Opportunities
northstar/
  config.py, llm.py        Settings; get_llm() and the API-key check
  graph.py                 LangGraph orchestrator (weekly_review, plan_week, recommend_datasets)
  agents/                  profile_agent, roadmap_agent, schedule_agent, progress_agent, dataset_agent
  tools/profile_tools.py   The Profile Agent's tools
  prompts/                 shared_safety.md + one prompt per agent
  services/                scheduler, patterns, progress, goals, review, tasks, profile, kaggle_dataset_service (no LLM)
  db/                      models.py, session.py, seed.py (demo data)
  schemas.py               Pydantic schemas shared by forms, tools, agents and tests
tests/                     pytest; no API key needed (agents are tested with a fake LLM)
data/
  kaggle_catalog.json      Curated, safe, age-appropriate tabular datasets
  opportunities.json       Curated student opportunities
  sources.json             Vetted research library
```

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q
```

They cover the scheduling guarantees, pattern detection, the review → accept → replan loop through the LangGraph graph, safety rules (e.g. the AI can't choose the student, decided changes can't be re-decided), and no-guilt wording. Agents are tested with a fake LLM, so tests are free and repeatable.

## Demo data

```bash
.venv/Scripts/python.exe -m northstar.db.seed --demo           # Aarav: goal, milestones, a stressed past week, this week's plan
.venv/Scripts/python.exe -m northstar.db.seed --week good      # refill last week with a good / bad / stressed week
```

The same resets are in the app under **🧪 Demo tools** in the home page's sidebar.

## Demo script (about 7 minutes)

1. **Hook:** "Students have goals but no system. NorthStar plans around their real life, adapts every week, and works for the student, never to monitor them."
2. **Onboarding** *(after "Start empty")*: chat: "I'm in grade 10, I love building robots, algebra is hard, I get about 2 hours on weekdays." Watch the profile card fill in.
3. **Goal → milestones:** "Score 85%+ in maths and finish my line-follower robot". Show checkable "done when…" criteria; edit one; confirm.
4. **This week:** plan it. Point out the capacity bars: "The AI suggests tasks; plain Python guarantees no day is overloaded."
5. **Check-in:** mark a task "Partly · too hard · blocker: factorising". Note the kind, guilt-free reply.
6. **Weekly review** *(load the full demo first)*: last week's stressed pattern, the review, and suggested changes. Accept one → This Week shows the lighter plan.
7. **Safety:** in the chat, ask it to write a graded essay; it declines and offers to help plan it instead.

## Roadmap

MVP (phases 1–8), niche discovery (phase 9), the research studio (phase 10) and opportunity discovery (phase 11) are complete. Next: testing, polish and deployment to Streamlit Community Cloud. See `PLAN.md`.

**Before real students use Opportunities:** each entry in `data/opportunities.json` is marked "not yet checked". Have a person open each official page, confirm the details, and fill in `verified_on` (and `deadline`, if there is one). Entries with a past deadline are hidden automatically.
