"""Opportunities from NorthStar's curated list: eligibility checked in Python, "why it fits" from the AI,
every fact from the data. NorthStar never signs anyone up - there is only an "Open official page" link."""
from datetime import date

import streamlit as st

from northstar.agents.opportunity_agent import OpportunityError
from northstar.db.session import get_session, init_db
from northstar.graph import run
from northstar.llm import ai_problem
from northstar.schemas import Opportunity
from northstar.services.opportunities import (
    AGE_RANGE, by_key, decide, dismissed_keys, eligible, load_opportunities, permission_note, recommendations,
)
from northstar.services.profile import demo_student_id, get_profile

st.set_page_config(page_title="Opportunities · NorthStar", page_icon="⭐", layout="wide")
init_db()
today = date.today()

with get_session() as s:
    student_id = demo_student_id(s)
    profile = get_profile(s, student_id) if student_id else None
if profile is None:
    st.info("First, tell NorthStar about yourself.")
    st.page_link("pages/1_Onboarding.py", label="Go to onboarding", icon="📝")
    st.stop()

st.title("Opportunities")
st.caption("Competitions, programmes and courses from NorthStar's curated list. NorthStar never signs you up - "
           "you decide, together with a parent or teacher.")

only_free = st.toggle("Free only", key="only_free")
with get_session() as s:
    picked = recommendations(s, student_id, "suggested")
    saved = recommendations(s, student_id, "saved")
    skip = dismissed_keys(s, student_id)
catalogue = by_key()
open_to_you = [o for o in eligible(load_opportunities(), profile, today, only_free) if o.key not in skip]
low, high = AGE_RANGE[profile.age_band]
st.caption(f"{len(open_to_you)} of {len(catalogue)} opportunities in the list are open to ages {low}-{high}"
           + (" and free." if only_free else "."))


def facts(o: Opportunity) -> None:
    """Every detail comes from the curated data - never from the AI."""
    st.write(o.summary)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Who can join**  \n{o.eligibility}  \n:gray[Open to: {o.region}]")
    c2.markdown(f"**Deadline**  \n{o.deadline:%d %b %Y}" if o.deadline else "**Deadline**  \nCheck the official page")
    c2.markdown(f"**Cost**  \n{o.cost}")
    c3.markdown(f"**Time**  \n{o.time_commitment}  \n:gray[{o.format}]")
    if o.skills_required:
        st.caption("Skills: " + ", ".join(o.skills_required))
    if note := permission_note(o, profile):
        st.info(f"👪 {note}")
    st.caption(f"✓ Details checked on {o.verified_on:%d %b %Y}" if o.verified_on
               else "⚠️ Not yet checked by NorthStar - confirm every detail on the official page.")


def card(rec, o: Opportunity) -> None:
    with st.container(border=True):
        st.markdown(f"### {o.title}")
        st.caption(f"{o.organizer} · {o.type}")
        st.markdown(f"**Why it fits you:** {rec.reason}")
        facts(o)
        buttons = st.container(horizontal=True)
        buttons.link_button("Open official page ↗", str(o.url))
        if rec.status == "suggested":
            if buttons.button("Save", key=f"save_{rec.id}", type="primary"):
                with get_session() as s:
                    decide(s, student_id, rec.id, "saved")
                st.rerun()
            if buttons.button("Not for me", key=f"no_{rec.id}"):
                with get_session() as s:
                    decide(s, student_id, rec.id, "dismissed")
                st.rerun()
        elif buttons.button("Remove from saved", key=f"unsave_{rec.id}"):
            with get_session() as s:
                decide(s, student_id, rec.id, "suggested")
            st.rerun()


# ---- Picked for you (AI explains, Python decided eligibility) ---------------------------------
st.subheader("Picked for you")
if ai_off := ai_problem():
    st.warning(ai_off + " You can still browse everything you're eligible for below.")
if st.button("✨ Find my best matches" if not picked else "✨ Find matches again", disabled=bool(ai_off)):
    try:
        with st.spinner("NorthStar is matching opportunities to your interests…"):
            found = run("match_opportunities", student_id, only_free=only_free)
    except OpportunityError as e:
        st.error(f"{e}. Please try again.")
    except Exception as e:  # network / API key problems
        st.error(f"Couldn't reach the AI: {e}")
    else:
        if not found:
            st.info("Nothing you're eligible for matches your interests yet - browse the full list below, "
                    "or add interests on the Explore page.")
        else:
            st.rerun()
for rec in picked:
    if rec.opportunity_key in catalogue:  # skip entries removed from the curated file
        card(rec, catalogue[rec.opportunity_key])

if saved:
    st.subheader("Saved")
    for rec in saved:
        if rec.opportunity_key in catalogue:
            card(rec, catalogue[rec.opportunity_key])

# ---- Everything they're eligible for (no AI) --------------------------------------------------
with st.expander(f"Browse everything you're eligible for ({len(open_to_you)})"):
    for o in open_to_you:
        st.markdown(f"**{o.title}** - {o.organizer} · {o.type}")
        facts(o)
        st.link_button("Open official page ↗", str(o.url), key=f"browse_{o.key}")
        st.divider()
