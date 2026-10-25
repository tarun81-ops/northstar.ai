"""NorthStar home. Placeholder until the Today dashboard in Phase 8.
Run:  .venv/Scripts/python.exe -m streamlit run app.py
"""
import streamlit as st

from northstar.db.session import get_session, init_db
from northstar.services.profile import demo_student_id, get_profile

st.set_page_config(page_title="NorthStar", page_icon="⭐")
init_db()

st.title("⭐ NorthStar")
st.caption("One Mentor. Every Student's Path.")

with get_session() as s:
    student_id = demo_student_id(s)
    profile = get_profile(s, student_id) if student_id else None

if profile is None:
    st.info("Let's start by getting to know you.")
    st.page_link("pages/1_Onboarding.py", label="Start onboarding", icon="📝")
else:
    st.write(f"Welcome back, **{profile.alias}**!")
    st.page_link("pages/1_Onboarding.py", label="Edit your profile", icon="📝")
