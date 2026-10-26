"""This week's plan. The AI proposes tasks, the scheduler picks the days, the student saves."""
from datetime import date, timedelta

import streamlit as st

from northstar.agents.schedule_agent import ScheduleError, WeekPlan, plan_week
from northstar.db.session import get_session, init_db
from northstar.schemas import MilestoneIn
from northstar.services.goals import get_active_goal
from northstar.services.profile import demo_student_id, get_profile
from northstar.services.scheduler import capacity, exams_soon, week_days, week_start
from northstar.services.tasks import busy_minutes, get_week_tasks, save_week

STATUS_ICON = {"todo": "⬜", "done": "✅", "partial": "🌓", "skipped": "⏭️"}

st.set_page_config(page_title="This Week · NorthStar", page_icon="⭐", layout="wide")
init_db()
today = date.today()

with get_session() as s:
    student_id = demo_student_id(s)
    profile = get_profile(s, student_id) if student_id else None
    goal = get_active_goal(s, student_id) if student_id else None

st.title("This week")
if profile is None:
    st.info("First, tell NorthStar about yourself.")
    st.page_link("pages/1_Onboarding.py", label="Go to onboarding", icon="📝")
    st.stop()
if goal is None or not goal.milestones:
    st.info("Set a goal and confirm its milestones first - your week is planned from them.")
    st.page_link("pages/2_Goal_and_Roadmap.py", label="Go to goal & roadmap", icon="🎯")
    st.stop()

which = st.radio("Week", ["This week", "Next week"], horizontal=True, label_visibility="collapsed")
start = week_start(today) + timedelta(days=7 if which == "Next week" else 0)
st.caption(f"{start:%d %b} – {start + timedelta(days=6):%d %b %Y} · goal: {goal.title}")

with get_session() as s:
    saved = get_week_tasks(s, student_id, start)
day_capacity = capacity(profile, start, today=start)  # full capacity of each day, for the bars

exams = exams_soon(profile, start)
if exams:
    st.warning("📚 **Exam mode:** " + ", ".join(f"{e.subject} on {e.date:%a %d %b}" for e in exams)
               + ". Revision comes first; other work is lighter or optional.")


def render_week(items: list[dict]) -> None:
    """items: dicts with day, title, minutes, optional, status, why."""
    planned = sum(i["minutes"] for i in items)
    st.caption(f"Planned **{planned} min** of {sum(day_capacity.values())} min available this week "
               f"(20% of your free time is always left unplanned).")
    for col, d in zip(st.columns(7), week_days(start)):
        day_items = [i for i in items if i["day"] == d]
        used = sum(i["minutes"] for i in day_items)
        with col:
            st.markdown(f"**{d:%a}** {d:%d %b}" + (" 📍" if d == today else ""))
            if day_capacity[d]:
                st.progress(min(used / day_capacity[d], 1.0), text=f"{used}/{day_capacity[d]} min")
            else:
                st.caption("No free time")
            for i in day_items:
                with st.container(border=True):
                    st.markdown(f"{STATUS_ICON[i['status']]} {i['title']}")
                    st.caption(f"{i['minutes']} min" + (" · optional" if i["optional"] else "") + (f" · {i['why']}" if i["why"] else ""))


preview_key = f"week_preview_{start}"
preview: WeekPlan | None = st.session_state.get(preview_key)
kept = [t for t in saved if t.status != "todo"]  # history: re-planning never touches these

buttons = st.container(horizontal=True)
if buttons.button("✨ Re-plan (keeps finished tasks)" if saved else "✨ Plan this week"):
    milestones = [MilestoneIn.model_validate(m, from_attributes=True) for m in goal.milestones]
    try:
        with st.spinner("NorthStar is planning your week…"):
            plan = plan_week(profile, goal.title, milestones, start, today, busy_minutes(saved))
    except ScheduleError as e:
        st.error(f"{e}. Please try again.")
    except Exception as e:  # network / API key problems
        st.error(f"Couldn't reach the AI: {e}")
    else:
        st.session_state[preview_key] = plan
        st.rerun()
if preview is not None and buttons.button("Discard preview"):
    del st.session_state[preview_key]
    st.rerun()

as_item = lambda t: dict(day=t.scheduled_date, title=t.title, minutes=t.duration_minutes, optional=t.is_optional,
                         status=t.status, why="")

if preview is not None:
    st.info("✏️ **Preview - not saved yet.** Save it to make it your plan" + (" (finished tasks stay as they are)." if kept else "."))
    if not preview.placed and not sum(preview.capacity.values()):
        st.write("There's no free time left to plan this week. Rest counts too! 🌙")
    render_week([as_item(t) for t in kept] + [
        dict(day=d, title=t.title, minutes=t.duration_minutes, optional=t.is_optional, status="todo", why=t.why)
        for d, t in preview.placed
    ])
    if preview.overflow:
        st.warning("Didn't fit this week without overloading you: "
                   + ", ".join(f"{t.title} ({t.duration_minutes} min)" for t in preview.overflow))
    if preview.placed and st.button("Save this plan", type="primary"):
        with get_session() as s:
            save_week(s, student_id, start, preview.placed)
        del st.session_state[preview_key]
        st.success("Your week is planned!")
        st.rerun()
elif saved:
    render_week([as_item(t) for t in saved])
else:
    st.write("No plan for this week yet. Ask NorthStar to plan it.")
