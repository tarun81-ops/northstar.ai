"""Weekly review: the numbers and patterns (Python), a kind review (AI), and proposed changes the student decides on."""
import json
from datetime import date, timedelta

import streamlit as st

from northstar.agents.progress_agent import ReviewError
from northstar.db.session import get_session, init_db
from northstar.graph import run
from northstar.llm import ai_problem
from northstar.services.patterns import MIN_CHECKINS, detect
from northstar.services.profile import demo_student_id
from northstar.services.progress import week_summary
from northstar.services.review import decide_change, describe, get_review
from northstar.services.scheduler import week_start
from northstar.ui.model_selector import render_model_selector

PATTERN_LABEL = {
    "too_easy": "🚀 Feeling easy", "too_hard": "🧗 Feeling hard", "skipping_subject": "⏭️ Skipped often",
    "overloaded": "🏋️ Too much on", "losing_interest": "📉 Less fun lately", "strong_interest": "✨ Really enjoying",
    "bad_week": "🌧️ A tough week",
}
DECIDED = {"applied": "✅ Accepted - used next time you plan a week", "dismissed": "✖️ Dismissed"}

st.set_page_config(page_title="Weekly Review · NorthStar", page_icon="⭐", layout="wide")
render_model_selector(required_capabilities={"structured_output"})
init_db()
today = date.today()

with get_session() as s:
    student_id = demo_student_id(s)
if student_id is None:
    st.info("First, tell NorthStar about yourself.")
    st.page_link("pages/1_Onboarding.py", label="Go to onboarding", icon="📝")
    st.stop()

st.title("Weekly review")
monday = week_start(today)
weeks = {f"Last week ({monday - timedelta(days=7):%d %b})": monday - timedelta(days=7),
         f"This week so far ({monday:%d %b})": monday}
start = weeks[st.radio("Week", list(weeks), horizontal=True, label_visibility="collapsed")]

with get_session() as s:
    summary = week_summary(s, student_id, start)
    review, changes = get_review(s, student_id, start)

if not summary["tasks_planned"]:
    st.info("No tasks were planned that week, so there's nothing to review yet.")
    st.page_link("pages/3_This_Week.py", label="Plan a week", icon="🗓️")
    st.stop()

# ---- 1. The numbers (Python) ---------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tasks done", f"{summary['done']} of {summary['tasks_planned']}",
          help=f"{summary['partial']} partly done, {summary['skipped']} skipped")
c2.metric("Completion", f"{summary['completion_rate']:.0%}", help="Partly done tasks count as half")
c3.metric("Time spent", f"{summary['actual_minutes']} min",
          delta=f"{summary['actual_minutes'] - summary['planned_minutes_checked_in']:+} min vs plan", delta_color="off")
c4.metric("Average stress", f"{summary['avg_stress']}/5" if summary["avg_stress"] else "—")
if summary["not_checked_in"]:
    st.caption(f"{summary['not_checked_in']} task(s) have no check-in yet.")
    st.page_link("pages/4_Daily_Checkin.py", label="Check in", icon="✍️")

# ---- 2. Patterns (Python) --------------------------------------------------------
st.subheader("What NorthStar noticed")
patterns = detect(summary)
if patterns:
    for p in patterns:
        st.markdown(f"- **{PATTERN_LABEL[p.name]}{f' · {p.category}' if p.category else ''}** - {p.evidence}")
elif summary["done"] + summary["partial"] + summary["skipped"] < MIN_CHECKINS:
    st.write(f"Not enough check-ins yet to spot patterns (need at least {MIN_CHECKINS}).")
else:
    st.write("Nothing stands out - a steady week. 👍")

# ---- 3. Review + proposed changes (AI, student decides) ---------------------------
st.subheader("Your review")
if ai_off := ai_problem():
    st.warning(ai_off)
if st.button("✨ Write my weekly review" if review is None else "✨ Write it again", disabled=bool(ai_off)):
    try:
        with st.spinner("NorthStar is looking back at your week…"):
            run("weekly_review", student_id, start)
    except ReviewError as e:
        st.error(f"{e}. Please try again.")
    except Exception as e:  # network / API key problems
        st.error(f"Couldn't reach the AI: {e}")
    else:
        st.rerun()

if review:
    text = json.loads(review.summary)
    st.success(f"🌟 **Win:** {text['win']}")
    st.write(text["what_the_data_shows"])
    st.info(f"💭 {text['question']}")

    if changes:
        st.markdown("**Suggested changes for next week - you decide:**")
    else:
        st.caption("No changes suggested - keep going!")
    for c in changes:
        with st.container(border=True):
            st.markdown(f"**{describe(c.kind, c.payload)}**")
            st.caption(c.reason)
            if c.status == "proposed":
                buttons = st.container(horizontal=True)
                accept = buttons.button("Accept", key=f"accept_{c.id}", type="primary")
                dismiss = buttons.button("Dismiss", key=f"dismiss_{c.id}")
                if accept or dismiss:
                    with get_session() as s:
                        decide_change(s, student_id, c.id, accept=accept)
                    st.rerun()
            else:
                st.caption(DECIDED[c.status])
    if any(c.status == "applied" for c in changes):
        st.page_link("pages/3_This_Week.py", label="Plan your next week with these changes", icon="🗓️")
