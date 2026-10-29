"""One goal -> AI-suggested milestones -> student edits and confirms. Nothing is saved without a click."""
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from northstar.agents.roadmap_agent import RoadmapError, generate_milestones
from northstar.db.session import get_session, init_db
from northstar.llm import ai_problem
from northstar.schemas import CATEGORIES, GoalIn, MilestoneIn
from northstar.services.goals import archive_goal, create_goal, get_active_goal, save_milestones
from northstar.services.profile import demo_student_id, get_profile
from northstar.ui.model_selector import render_model_selector

st.set_page_config(page_title="Goal & Roadmap · NorthStar", page_icon="⭐", layout="wide")
render_model_selector(required_capabilities={"structured_output"})
init_db()

with get_session() as s:
    student_id = demo_student_id(s)
    profile = get_profile(s, student_id) if student_id else None
    goal = get_active_goal(s, student_id) if student_id else None

st.title("Your goal & roadmap")

if profile is None:
    st.info("First, tell NorthStar about yourself.")
    st.page_link("pages/1_Onboarding.py", label="Go to onboarding", icon="📝")
    st.stop()


def show_errors(e: ValidationError, row_offset: int | None = None) -> None:
    for err in e.errors():
        where = " → ".join(str(p) for p in err["loc"]) or "input"
        prefix = f"Row {row_offset + 1}, " if row_offset is not None else ""
        st.markdown(f"- {prefix}**{where.replace('_', ' ')}**: {err['msg']}")


# ---- 1. No goal yet: set one -------------------------------------------------
if goal is None:
    st.write("Pick **one** goal to focus on. NorthStar will break it into milestones you can edit.")
    with st.form("goal"):
        title = st.text_input("Goal", placeholder="e.g. Score 85%+ in maths and finish my line-follower robot")
        description = st.text_area("Why it matters to you (optional)")
        c1, c2 = st.columns(2)
        category = c1.selectbox("Main area", CATEGORIES)
        target = c2.date_input("Target date", value=date.today() + timedelta(days=60), min_value=date.today() + timedelta(days=7))
        submitted = st.form_submit_button("Set goal", type="primary")
    if submitted:
        try:
            new_goal = GoalIn(title=title, description=description, category=category, target_date=target)
        except ValidationError as e:
            st.error("Please fix:")
            show_errors(e)
        else:
            with get_session() as s:
                create_goal(s, student_id, new_goal)
            st.rerun()
    st.stop()

# ---- 2. Active goal ------------------------------------------------------------
st.subheader(f"🎯 {goal.title}")
days_left = (goal.target_date - date.today()).days
st.caption(f"{goal.category} · target {goal.target_date:%d %b %Y} · {days_left} days left")
if goal.description:
    st.write(goal.description)

saved = [MilestoneIn.model_validate(m, from_attributes=True) for m in goal.milestones]
if saved:
    done = sum(m.status == "done" for m in saved)
    st.progress(done / len(saved), text=f"{done} of {len(saved)} milestones done")

draft_key = f"milestone_draft_{goal.id}"
draft: list[MilestoneIn] | None = st.session_state.get(draft_key)

if ai_off := ai_problem():
    st.warning(ai_off + " You can still add milestones by hand below.")
buttons = st.container(horizontal=True)  # buttons sized to their labels, side by side
if buttons.button("✨ Suggest milestones" if not saved else "✨ Suggest a new set", disabled=bool(ai_off)):
    try:
        goal_in = GoalIn(title=goal.title, description=goal.description, category=goal.category, target_date=goal.target_date)
        with st.spinner("NorthStar is planning your milestones…"):
            drafts = generate_milestones(profile, goal_in)
    except ValidationError:
        st.error("This goal's target date has passed - start a new goal below.")
    except RoadmapError as e:
        st.error(f"{e}. Try again, or add milestones by hand.")
    except Exception as e:  # network / API key problems
        st.error(f"Couldn't reach the AI: {e}")
    else:
        st.session_state[draft_key] = [MilestoneIn(**d.model_dump()) for d in drafts]
        st.rerun()
if draft is not None and buttons.button("Discard suggestion"):
    del st.session_state[draft_key]
    st.rerun()

rows = draft if draft is not None else saved
if draft is not None:
    st.info("✏️ **Suggestion - not saved yet.** Edit anything you like, add or delete rows, then confirm.")
elif not saved:
    st.write("No milestones yet. Ask NorthStar to suggest some, or add your own below.")

edited = st.data_editor(
    pd.DataFrame([m.model_dump() for m in rows], columns=list(MilestoneIn.model_fields)),
    column_order=["title", "due_date", "completion_criteria", "category", "status", "description"],
    column_config={
        "id": None,  # hidden, but kept so edits update the right milestone
        "title": st.column_config.TextColumn("Milestone", width="medium", required=True),
        "due_date": st.column_config.DateColumn("Due", required=True, max_value=goal.target_date),
        "completion_criteria": st.column_config.TextColumn("Done when…", width="large", required=True),
        "category": st.column_config.SelectboxColumn("Area", options=list(CATEGORIES), required=True),
        "status": st.column_config.SelectboxColumn("Status", options=["todo", "in_progress", "done"], default="todo"),
        "description": st.column_config.TextColumn("Details", width="large"),
    },
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    key=f"editor_{goal.id}_{'draft' if draft is not None else 'saved'}",
)

kept_ids = {int(i) for i in edited["id"].dropna()}
removed = [m for m in saved if m.id not in kept_ids]
confirmed = True
if removed:  # deleting saved work is irreversible: ask explicitly
    confirmed = st.checkbox(f"Yes, delete {len(removed)} saved milestone(s): " + ", ".join(m.title for m in removed))

if st.button("Confirm & save milestones", type="primary", disabled=not confirmed):
    items, ok = [], True
    for i, r in enumerate(edited.dropna(how="all").to_dict("records")):
        try:
            items.append(MilestoneIn.model_validate({k: v for k, v in r.items() if not pd.isna(v)}))
        except ValidationError as e:
            ok = False
            show_errors(e, row_offset=i)
    if ok:
        try:
            with get_session() as s:
                save_milestones(s, goal.id, items)
        except ValueError as e:
            st.error(str(e))
        else:
            st.session_state.pop(draft_key, None)
            st.success("Milestones saved!")
            st.rerun()

# ---- 3. Change goal (irreversible-ish, so behind a confirmation) ----------------
with st.expander("Start a different goal"):
    st.write("Your current goal and its milestones will be archived, not deleted.")
    if st.checkbox("Yes, archive this goal") and st.button("Archive goal"):
        with get_session() as s:
            archive_goal(s, goal.id)
        st.session_state.pop(draft_key, None)
        st.rerun()
