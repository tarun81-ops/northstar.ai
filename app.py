"""NorthStar home: today's checklist.
Run:  .venv/Scripts/python.exe -m streamlit run app.py
"""
from datetime import date

import streamlit as st

from northstar.db.session import get_session, init_db
from northstar.schemas import CheckIn
from northstar.services.goals import get_active_goal
from northstar.services.profile import demo_student_id, get_profile
from northstar.services.progress import log_checkin, tasks_for_day, undo_checkin
from northstar.services.scheduler import exams_soon

st.set_page_config(page_title="NorthStar", page_icon="⭐")
init_db()
today = date.today()

st.title("⭐ NorthStar")
st.caption("One Mentor. Every Student's Path.")

with get_session() as s:
    student_id = demo_student_id(s)
    profile = get_profile(s, student_id) if student_id else None
    goal = get_active_goal(s, student_id) if student_id else None
    tasks = tasks_for_day(s, student_id, today) if student_id else []

if profile is None:
    st.info("Let's start by getting to know you.")
    st.page_link("pages/1_Onboarding.py", label="Start onboarding", icon="📝")
    st.stop()


def toggle(task_id: int, minutes: int) -> None:
    """Tick = done with the planned minutes; un-tick = undo. Details can be added on the check-in page."""
    with get_session() as s:
        if st.session_state[f"tick_{task_id}"]:
            log_checkin(s, student_id, today, CheckIn(task_id=task_id, completed="yes", time_spent_minutes=minutes))
        else:
            undo_checkin(s, student_id, task_id, today)


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
next_milestone = next((m for m in goal.milestones if m.status != "done"), None) if goal else None
if next_milestone:
    upcoming.append(f"🎯 Milestone: {next_milestone.title} - due {next_milestone.due_date:%a %d %b}")
st.markdown("\n".join(f"- {u}" for u in upcoming) if upcoming else "Nothing due in the next two weeks.")

st.divider()
st.page_link("pages/3_This_Week.py", label="This week's plan", icon="🗓️")
st.page_link("pages/2_Goal_and_Roadmap.py", label="Your goal & roadmap", icon="🎯")
st.page_link("pages/1_Onboarding.py", label="Edit your profile", icon="📝")
