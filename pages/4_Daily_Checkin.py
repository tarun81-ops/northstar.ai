"""Daily check-in, designed to take under a minute. Only "Did you do it?" is needed."""
from datetime import date, timedelta

import streamlit as st
from pydantic import ValidationError

from northstar.db.session import get_session, init_db
from northstar.schemas import CheckIn
from northstar.services.profile import demo_student_id
from northstar.services.progress import encouragement, log_checkin, logs_for_day, tasks_for_day

DONE = {"Yes": "yes", "Partly": "partial", "No": "no"}
DIFFICULTY = {"Too easy": "too_easy", "Just right": "just_right", "Too hard": "too_hard"}
FACES = {1: "😣", 2: "🙁", 3: "😐", 4: "🙂", 5: "😄"}
STRESS = {1: "😌 calm", 2: "🙂 fine", 3: "😐 so-so", 4: "😟 stressed", 5: "😫 very stressed"}

st.set_page_config(page_title="Daily Check-in · NorthStar", page_icon="⭐", layout="wide")
init_db()
today = date.today()

with get_session() as s:
    student_id = demo_student_id(s)
if student_id is None:
    st.info("First, tell NorthStar about yourself.")
    st.page_link("pages/1_Onboarding.py", label="Go to onboarding", icon="📝")
    st.stop()

st.title("Daily check-in")
day = st.date_input("Which day?", value=today, max_value=today, min_value=today - timedelta(days=13))
with get_session() as s:
    tasks = tasks_for_day(s, student_id, day)
    existing = logs_for_day(s, student_id, day)

if not tasks:
    st.info("Nothing was planned for this day. Enjoy the free time! 🌿")
    st.page_link("pages/3_This_Week.py", label="See this week's plan", icon="🗓️")
    st.stop()


def label_of(options: dict, value) -> str | None:
    return next((label for label, v in options.items() if v == value), None)


def index_of(options: dict | list, label) -> int | None:
    return list(options).index(label) if label in options else None


st.caption("Only **Did you do it?** is needed. Everything else is optional - skip anything you like.")
with st.form("checkin"):
    entries = []
    for t in tasks:
        log = existing.get(t.id)
        key = f"{day}_{t.id}"  # fresh widgets per day and task
        with st.container(border=True):
            st.markdown(f"**{t.title}** · {t.duration_minutes} min planned" + (" · optional" if t.is_optional else ""))
            c1, c2, c3 = st.columns([2, 1, 3])
            done = c1.radio("Did you do it?", list(DONE), horizontal=True, key=f"done_{key}",
                            index=index_of(DONE, label_of(DONE, log.completed)) if log else None)
            minutes = c2.number_input("Minutes spent", 0, 600, step=5, key=f"min_{key}",
                                      value=log.time_spent_minutes if log else t.duration_minutes)
            feel = c3.radio("How did it feel?", list(DIFFICULTY), horizontal=True, key=f"diff_{key}",
                            index=index_of(DIFFICULTY, label_of(DIFFICULTY, log.difficulty_rating)) if log else None)
            with st.expander("More (optional)"):
                enjoyed = st.radio("Did you enjoy it?", list(FACES), format_func=FACES.get, horizontal=True, key=f"joy_{key}",
                                   index=index_of(FACES, log.enjoyment) if log else None)
                blockers = st.text_input("What got in the way?", log.blockers if log else "", key=f"block_{key}")
                note = st.text_input("Anything to remember?", log.student_notes if log else "", key=f"note_{key}")
                link = st.text_input("Link to your work (photo, doc, repo)", (log.evidence_url or "") if log else "",
                                     key=f"link_{key}", placeholder="https://…")
        entries.append(dict(task=t, done=done, minutes=minutes, feel=feel, enjoyed=enjoyed, blockers=blockers, note=note, link=link))

    stress_before = next((log.stress for log in existing.values() if log.stress), None)
    stress = st.radio("Overall, how stressful did the day feel?", list(STRESS), format_func=STRESS.get, horizontal=True,
                      key=f"stress_{day}", index=index_of(STRESS, stress_before))
    submitted = st.form_submit_button("Save check-in", type="primary")

if submitted:
    checkins, ok = [], True
    for e in entries:
        if e["done"] is None:
            continue  # unanswered tasks are simply left for later
        try:
            checkins.append(CheckIn(
                task_id=e["task"].id,
                completed=DONE[e["done"]],
                time_spent_minutes=0 if e["done"] == "No" else e["minutes"],
                difficulty_rating=DIFFICULTY.get(e["feel"]),
                enjoyment=e["enjoyed"],
                stress=stress,
                blockers=e["blockers"],
                student_notes=e["note"],
                evidence_url=e["link"].strip() or None,
            ))
        except ValidationError as err:
            ok = False
            for problem in err.errors():
                if problem["loc"][0] == "evidence_url":
                    message = "Link: paste a full web address starting with https://"
                else:
                    message = f"{str(problem['loc'][0]).replace('_', ' ')}: {problem['msg']}"
                st.error(f"{e['task'].title} - {message}")
    if ok and not checkins:
        st.warning("Answer **Did you do it?** for at least one task.")
    elif ok:
        with get_session() as s:
            for c in checkins:
                log_checkin(s, student_id, day, c)
        st.success(encouragement(checkins))
