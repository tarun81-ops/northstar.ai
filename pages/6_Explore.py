"""Explore: interest signals -> niches that cite them -> a short curiosity experiment -> reflection."""
from datetime import date

import streamlit as st
from pydantic import ValidationError

from northstar.agents.niche_agent import NicheError
from northstar.db.session import get_session, init_db
from northstar.graph import run
from northstar.llm import ai_problem
from northstar.schemas import Reflection
from northstar.services.niches import (
    active_experiment, add_signal, dismiss_niche, finish_experiment, past_experiments, refresh_signals, remove_signal,
    set_step, start_experiment, stop_experiment, suggestions,
)
from northstar.services.profile import demo_student_id, get_profile
from northstar.ui.model_selector import render_model_selector

SOURCE = {"profile": "from your profile", "checkins": "from your check-ins", "student": "you said", "experiment": "from an experiment"}
SCALE = {1: "1 · not at all", 2: "2", 3: "3 · somewhat", 4: "4", 5: "5 · a lot"}
QUESTIONS = {
    "curiosity": "Did it make you curious to learn more?",
    "energy": "Did it give you energy (instead of draining you)?",
    "confidence": "Do you feel more confident about it now?",
    "meaning": "Does it feel meaningful to you?",
}

st.set_page_config(page_title="Explore · NorthStar", page_icon="⭐", layout="wide")
render_model_selector(required_capabilities={"structured_output"})
init_db()
today = date.today()

with get_session() as s:
    student_id = demo_student_id(s)
    profile = get_profile(s, student_id) if student_id else None
if profile is None:
    st.info("First, tell NorthStar about yourself.")
    st.page_link("pages/1_Onboarding.py", label="Go to onboarding", icon="📝")
    st.stop()

with get_session() as s:
    signals = refresh_signals(s, student_id, profile)  # cheap, and keeps signals in step with new check-ins
    experiment = active_experiment(s, student_id)
    open_niches = suggestions(s, student_id)
    history = past_experiments(s, student_id)

def toggle_step(experiment_id: int, i: int) -> None:
    with get_session() as s:
        set_step(s, student_id, experiment_id, i, st.session_state[f"step_{experiment_id}_{i}"])


st.title("Explore your interests")
st.caption("Find a specific niche that sparks your curiosity, then try it with a short, free experiment. "
           "There are no wrong answers - finding out what you're *not* into is useful too.")

# ---- 1. The active experiment -------------------------------------------------------
if experiment:
    with st.container(border=True):
        st.subheader(f"🧪 {experiment.title}")
        day = (today - experiment.started_on).days + 1
        done = sum(step["done"] for step in experiment.steps)
        st.caption(f"Day {min(day, experiment.duration_days)} of {experiment.duration_days} · {done} of {len(experiment.steps)} steps done")
        for i, step in enumerate(experiment.steps):
            st.checkbox(step["text"], value=step["done"], key=f"step_{experiment.id}_{i}",
                        on_change=toggle_step, args=(experiment.id, i))

        with st.expander("Finish and reflect", expanded=done == len(experiment.steps)):
            with st.form("reflect"):
                scores = {k: st.select_slider(q, options=list(SCALE), value=3, format_func=SCALE.get) for k, q in QUESTIONS.items()}
                notes = st.text_area("What did you notice? What surprised you? (optional)")
                if st.form_submit_button("Save reflection", type="primary"):
                    try:
                        reflection = Reflection(**scores, notes=notes)
                    except ValidationError as e:
                        st.error(str(e))
                    else:
                        with get_session() as s:
                            became_signal = finish_experiment(s, student_id, experiment.id, reflection)
                        st.session_state["reflected"] = became_signal
                        st.rerun()
        if st.button("Stop this experiment - it's not for me"):
            with get_session() as s:
                stop_experiment(s, student_id, experiment.id)
            st.rerun()


if "reflected" in st.session_state:
    st.success("Reflection saved! " + ("It sparked something - NorthStar added it to your interests. ✨"
                                       if st.session_state.pop("reflected") else "Good to know - that's useful too."))

# ---- 2. Signals -----------------------------------------------------------------
st.subheader("What you're curious about")
if signals:
    for sig in signals:
        left, right = st.columns([6, 1], vertical_alignment="center")
        left.markdown(f"{'★' * sig.strength}{'☆' * (5 - sig.strength)}  {sig.signal}  \n:gray[{SOURCE[sig.source]}]")
        if sig.source == "student" and right.button("Remove", key=f"rm_{sig.id}"):
            with get_session() as s:
                remove_signal(s, student_id, sig.id)
            st.rerun()
else:
    st.write("Nothing yet - add something you're curious about below.")

with st.form("add_signal", clear_on_submit=True):
    text = st.text_input("Something you've been curious about lately", placeholder="e.g. how video games use physics")
    if st.form_submit_button("Add"):
        try:
            with get_session() as s:
                add_signal(s, student_id, text)
        except ValueError as e:
            st.error(str(e))
        else:
            st.rerun()

# ---- 3. Niches ------------------------------------------------------------------
st.subheader("Niches to explore")
if ai_off := ai_problem():
    st.warning(ai_off)
if st.button("✨ Suggest niches" if not open_niches else "✨ Suggest different niches",
             disabled=bool(ai_off) or not signals):
    try:
        with st.spinner("NorthStar is looking for niches that fit you…"):
            run("suggest_niches", student_id)
    except NicheError as e:
        st.error(f"{e}. Please try again.")
    except Exception as e:  # network / API key problems
        st.error(f"Couldn't reach the AI: {e}")
    else:
        st.rerun()

for niche in open_niches:
    with st.container(border=True):
        st.markdown(f"### {niche.niche_name}")
        st.write(niche.description)
        st.markdown(f"**Why it fits you:** {niche.why}")
        st.caption("Builds on: " + " · ".join(f"“{m['signal']}”" for m in niche.matched_signals))
        if niche.skills_required:
            st.caption("Skills you'd build: " + ", ".join(niche.skills_required))
        exp = niche.experiment
        with st.expander(f"🧪 Try-out: {exp['title']} ({exp['duration_days']} days, free)"):
            st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(exp["steps"], 1)))
        buttons = st.container(horizontal=True)
        if buttons.button("Try this", key=f"try_{niche.id}", type="primary", disabled=experiment is not None,
                          help="Finish or stop your current experiment first" if experiment else None):
            with get_session() as s:
                start_experiment(s, student_id, niche.id, today)
            st.rerun()
        if buttons.button("Not for me", key=f"no_{niche.id}"):
            with get_session() as s:
                dismiss_niche(s, student_id, niche.id)
            st.rerun()

# ---- 4. Past experiments ---------------------------------------------------------------
if history:
    st.subheader("Past experiments")
    for exp, niche in history:
        if exp.status == "stopped":
            st.markdown(f"- **{niche.niche_name}** - {exp.title} · *stopped: not for you, and that's useful to know*")
        else:
            st.markdown(f"- **{niche.niche_name}** - {exp.title} · curiosity {exp.curiosity_score}/5, energy {exp.energy_score}/5, "
                        f"confidence {exp.confidence_score}/5, meaning {exp.meaning_score}/5"
                        + (f"  \n  :gray[“{exp.reflection}”]" if exp.reflection else ""))
