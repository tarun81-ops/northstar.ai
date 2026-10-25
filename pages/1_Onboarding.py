"""Plain onboarding form (no AI). Phase 3 adds a chat interview that fills the same profile."""
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from northstar.db.session import get_session, init_db
from northstar.schemas import DAYS, DayAvailability, ProfileIn
from northstar.services.profile import demo_student_id, get_profile, save_profile

AGE_BANDS = ["11-13", "14-16", "17-19", "college"]
LEARNING_PREFS = ["Practice problems", "Projects", "Videos", "Reading", "Discussion", "Flashcards"]
# Error messages show the form's labels, not internal field names.
LABELS = {
    "alias": "Nickname", "grade": "Grade / year", "language": "Preferred language",
    "interests": "Interests & hobbies", "learning_preferences": "How do you like to learn?",
    "weekly_availability": "Your week", "school": "School hours", "free_minutes": "Free minutes",
}

st.set_page_config(page_title="Onboarding · NorthStar", page_icon="⭐", layout="wide")
init_db()

with get_session() as s:
    student_id = demo_student_id(s)
    current = get_profile(s, student_id) if student_id else None


def as_text(items: list[str]) -> str:
    return ", ".join(items)


def as_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def rows(df: pd.DataFrame) -> list[dict]:
    """Editor table -> list of dicts, skipping rows the student left completely empty."""
    return df.dropna(how="all").to_dict("records")


st.title("Tell NorthStar about you")
st.caption("Use a nickname. NorthStar never asks for your full name, school, address or location.")

with st.form("profile"):
    c1, c2, c3, c4 = st.columns(4)
    alias = c1.text_input("Nickname", current.alias if current else "")
    age_band = c2.selectbox("Age band", AGE_BANDS, index=AGE_BANDS.index(current.age_band) if current else 1)
    grade = c3.text_input("Grade / year", current.grade if current else "", placeholder="e.g. Grade 10")
    language = c4.text_input("Preferred language", current.language if current else "English")

    st.subheader("Learning")
    st.caption("Separate items with commas.")
    subjects = st.text_input("Subjects", as_text(current.subjects) if current else "")
    interests = st.text_input("Interests & hobbies", as_text(current.interests) if current else "")
    c1, c2 = st.columns(2)
    strengths = c1.text_input("Strengths", as_text(current.strengths) if current else "")
    challenges = c2.text_input("Challenges", as_text(current.challenges) if current else "")
    saved_prefs = current.learning_preferences if current else []
    prefs = st.multiselect(
        "How do you like to learn?",
        LEARNING_PREFS + [p for p in saved_prefs if p not in LEARNING_PREFS],
        default=saved_prefs,
    )

    st.subheader("Your week")
    availability = current.weekly_availability if current else {d: DayAvailability() for d in DAYS}
    week = st.data_editor(
        pd.DataFrame([{"day": d, "school": availability[d].school, "free_minutes": availability[d].free_minutes} for d in DAYS]),
        column_config={
            "day": st.column_config.TextColumn("Day", disabled=True),
            "school": st.column_config.TextColumn("School hours", help="HH:MM-HH:MM, leave blank if no school"),
            "free_minutes": st.column_config.NumberColumn("Free minutes for study & projects", min_value=0, max_value=720, step=15),
        },
        hide_index=True,
        width="stretch",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Activities** (sports, clubs, lessons)")
        activities = st.data_editor(
            pd.DataFrame([a.model_dump() for a in current.activities] if current else [], columns=["name", "day", "minutes"]),
            column_config={
                "name": st.column_config.TextColumn("Activity"),
                "day": st.column_config.SelectboxColumn("Day", options=list(DAYS)),
                "minutes": st.column_config.NumberColumn("Minutes", min_value=0, max_value=600, step=15),
            },
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
        )
    with c2:
        st.markdown("**Upcoming exams & deadlines**")
        exams = st.data_editor(
            pd.DataFrame([e.model_dump() for e in current.exams] if current else [], columns=["subject", "date"]),
            column_config={
                "subject": st.column_config.TextColumn("Subject"),
                "date": st.column_config.DateColumn("Date"),
            },
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
        )

    submitted = st.form_submit_button("Save profile", type="primary")

if submitted:
    try:
        profile = ProfileIn(
            alias=alias,
            age_band=age_band,
            grade=grade,
            language=language,
            subjects=as_list(subjects),
            interests=as_list(interests),
            strengths=as_list(strengths),
            challenges=as_list(challenges),
            learning_preferences=prefs,
            weekly_availability={r["day"]: {"school": r["school"] or "", "free_minutes": r["free_minutes"]} for r in rows(week)},
            activities=rows(activities),
            exams=rows(exams),
        )
    except ValidationError as e:
        st.error("Some answers need fixing:")
        for err in e.errors():
            where = " → ".join(LABELS.get(p, str(p).capitalize()) if isinstance(p, str) else f"row {p + 1}" for p in err["loc"])
            st.markdown(f"- **{where}**: {err['msg']}")
    else:
        with get_session() as s:
            save_profile(s, profile, student_id)
        st.success(f"Saved! Nice to meet you, {profile.alias}.")
