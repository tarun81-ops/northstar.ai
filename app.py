"""NorthStar home: today's checklist and a small dashboard.
Run:  .venv/Scripts/python.exe -m streamlit run app.py
"""
import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from northstar.db.seed import reset_demo
from northstar.db.session import get_session, init_db
from northstar.schemas import CheckIn
from northstar.services.goals import get_active_goal
from northstar.services.profile import demo_student_id, get_profile
from northstar.services.progress import log_checkin, tasks_for_day, undo_checkin, week_summary
from northstar.services.review import adjustments_for_week, latest_review
from northstar.services.scheduler import exams_soon, week_start

st.set_page_config(page_title="NorthStar", page_icon="⭐")
init_db()
today = date.today()
monday = week_start(today)

# ---- Demo tools (sidebar, shown even when the database is empty) --------------------
with st.sidebar.expander("🧪 Demo tools"):
    st.caption("For demos and testing. Both options delete **all** data in this NorthStar.")
    resets = st.session_state.get("resets", 0)
    # a new key after every reset = a fresh, unticked box, so each reset needs its own confirmation
    sure = st.checkbox("I understand this deletes everything", key=f"sure_{resets}")
    full = st.button("Load the full demo (Aarav)", disabled=not sure, help="Profile, goal, milestones, a past week and this week's plan")
    empty = st.button("Start empty (demo onboarding)", disabled=not sure)
    if full or empty:
        reset_demo(full=full)
        st.session_state.clear()
        st.session_state["resets"] = resets + 1
        st.rerun()

st.title("⭐ NorthStar")
st.caption("One Mentor. Every Student's Path.")

with get_session() as s:
    student_id = demo_student_id(s)
    profile = get_profile(s, student_id) if student_id else None
    goal = get_active_goal(s, student_id) if student_id else None
    tasks = tasks_for_day(s, student_id, today) if student_id else []
    this_week = week_summary(s, student_id, monday) if student_id else None
    recent = [(start, week_summary(s, student_id, start)) for start in (monday - timedelta(days=7 * i) for i in range(3, -1, -1))] if student_id else []
    adjust = adjustments_for_week(s, student_id, monday) if student_id else None
    review = latest_review(s, student_id) if student_id else None

if profile is None:
    st.info("Let's start by getting to know you.")
    st.page_link("pages/1_Onboarding.py", label="Start onboarding", icon="📝")
    st.stop()
if goal is None or not goal.milestones:
    st.info(f"Hi {profile.alias}! Next step: pick one goal and NorthStar will break it into milestones.")
    st.page_link("pages/2_Goal_and_Roadmap.py", label="Set your goal", icon="🎯")
    st.stop()


def toggle(task_id: int, minutes: int) -> None:
    """Tick = done with the planned minutes; un-tick = undo. Details can be added on the check-in page."""
    with get_session() as s:
        if st.session_state[f"tick_{task_id}"]:
            log_checkin(s, student_id, today, CheckIn(task_id=task_id, completed="yes", time_spent_minutes=minutes))
        else:
            undo_checkin(s, student_id, task_id, today)


# ---- Today ---------------------------------------------------------------------
st.subheader(f"Today, {today:%A %d %B}")
if not tasks:
    st.write("Nothing planned for today. Enjoy your free time! 🌿")
else:
    done = sum(t.status == "done" for t in tasks)
    partly = sum(t.status == "partial" for t in tasks)  # partial work is acknowledged, never shown as zero
    st.progress(done / len(tasks), text=f"{done} of {len(tasks)} tasks done" + (f" · {partly} partly done 🌓" if partly else ""))
    for t in tasks:
        label = f"{t.title} · {t.duration_minutes} min" + (" · optional" if t.is_optional else "")
        if t.status in ("partial", "skipped"):  # already checked in with details: show, edit on the check-in page
            st.markdown(f"{'🌓' if t.status == 'partial' else '⏭️'} {label} *({t.status})*")
        else:
            st.checkbox(label, value=t.status == "done", key=f"tick_{t.id}", on_change=toggle, args=(t.id, t.duration_minutes))
    st.page_link("pages/4_Daily_Checkin.py", label="Add details: how it felt, what got in the way", icon="✍️")

st.subheader("Coming up")
upcoming = [f"📚 {e.subject} exam on {e.date:%a %d %b}" for e in exams_soon(profile, today)]
next_milestone = next((m for m in goal.milestones if m.status != "done"), None)
if next_milestone:
    upcoming.append(f"🎯 Milestone: {next_milestone.title} - due {next_milestone.due_date:%a %d %b}")
st.markdown("\n".join(f"- {u}" for u in upcoming) if upcoming else "Nothing due in the next two weeks.")

# ---- This week + recent weeks ---------------------------------------------------
st.subheader("This week")
if this_week["tasks_planned"]:
    st.progress(this_week["completion_rate"], text=f"{this_week['done']} of {this_week['tasks_planned']} tasks done"
                + (f" · {this_week['partial']} partly" if this_week["partial"] else ""))
else:
    st.write("No plan yet for this week.")
if adjust.notes:
    st.caption("🔁 Planning with the changes you accepted: " + "; ".join(adjust.notes))

if any(week["tasks_planned"] for _, week in recent):
    st.markdown("**Planned vs actual minutes**")
    st.bar_chart(
        pd.DataFrame({"Planned": [week["planned_minutes"] for _, week in recent],
                      "Actual": [week["actual_minutes"] for _, week in recent]},
                     index=[f"{start:%d %b}" for start, _ in recent]),
        stack=False, height=220, y_label="minutes",
    )

if review:
    st.subheader(f"From your review (week of {review.week_start:%d %b})")
    text = json.loads(review.summary)
    st.success(f"🌟 {text['win']}")
    st.caption(f"💭 {text['question']}")

st.divider()
st.page_link("pages/3_This_Week.py", label="This week's plan", icon="🗓️")
st.page_link("pages/5_Weekly_Review.py", label="Weekly review", icon="🔁")
st.page_link("pages/2_Goal_and_Roadmap.py", label="Your goal & roadmap", icon="🎯")
st.page_link("pages/1_Onboarding.py", label="Edit your profile", icon="📝")
