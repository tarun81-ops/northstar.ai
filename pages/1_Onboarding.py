"""Onboarding: chat with the Profile Agent, or fill in the plain form. Both save the same profile."""
import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from northstar.agents.profile_agent import build_profile_agent, greeting
from northstar.db.session import get_session, init_db
from northstar.llm import ai_problem
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


def profile_card(p: ProfileIn | None) -> None:
    if p is None:
        st.caption("Nothing saved yet - start chatting!")
        return
    weekly = sum(d.free_minutes for d in p.weekly_availability.values())
    st.markdown(
        f"**{p.alias}** · {p.grade} · age {p.age_band}\n\n"
        f"- **Subjects:** {as_text(p.subjects) or '—'}\n"
        f"- **Interests:** {as_text(p.interests) or '—'}\n"
        f"- **Strengths:** {as_text(p.strengths) or '—'}\n"
        f"- **Challenges:** {as_text(p.challenges) or '—'}\n"
        f"- **Learns by:** {as_text(p.learning_preferences) or '—'}\n"
        f"- **Free time:** {weekly / 60:.1f} h/week\n"
        f"- **Activities:** {as_text([f'{a.name} ({a.day} {a.minutes}m)' for a in p.activities]) or '—'}\n"
        f"- **Exams:** {as_text([f'{e.subject} {e.date:%d %b}' for e in p.exams]) or '—'}"
    )


st.title("Tell NorthStar about you")
st.caption("Use a nickname. NorthStar never asks for your full name, school, address or location.")
chat_tab, form_tab = st.tabs(["💬 Chat", "📝 Form"])

def render_chat() -> None:
    """The chat tab. Uses `return` (not st.stop) on problems, so the Form tab below still renders."""
    if ai_off := ai_problem():
        st.warning(ai_off + " Use the Form tab to fill in your profile.")
        return
    if "profile_chat" not in st.session_state:
        try:
            agent, state = build_profile_agent(student_id)
        except Exception as e:
            st.warning(f"Chat is unavailable ({e}). You can use the Form tab instead.")
            return
        st.session_state.profile_chat = {"agent": agent, "state": state, "messages": [AIMessage(greeting(current.alias if current else None))]}
    chat = st.session_state.profile_chat

    left, right = st.columns([3, 2])
    with left:
        box = st.container(height=520)
        for m in chat["messages"]:
            if m.type == "human":
                box.chat_message("user").markdown(m.text)
            elif m.type == "ai":
                saved = [k for c in m.tool_calls if c["name"] == "update_student_profile" for k in c["args"]]
                if saved:  # make the agent's tool use visible - handy for learning and for trust
                    box.caption("💾 Updating profile: " + ", ".join(LABELS.get(k, k.replace("_", " ")) for k in saved))
                if m.text:
                    box.chat_message("assistant").markdown(m.text)

        if prompt := st.chat_input("Type your answer…"):
            chat["messages"].append(HumanMessage(prompt))
            box.chat_message("user").markdown(prompt)
            with box.chat_message("assistant"), st.spinner("Thinking…"):
                try:
                    chat["messages"] = chat["agent"].invoke({"messages": chat["messages"]})["messages"]
                except Exception as e:
                    chat["messages"].pop()
                    st.error(f"Sorry, something went wrong talking to the AI: {e}")
                    return
            st.rerun()  # refresh the profile card with what was just saved

    with right:
        st.markdown("#### Your profile so far")
        sid = chat["state"]["student_id"]
        with get_session() as s:
            profile_card(get_profile(s, sid) if sid else None)
        if chat["state"]["draft"]:
            st.caption("Not saved yet (need nickname, age and grade): " + ", ".join(chat["state"]["draft"]))
        if st.button("Start the chat over"):
            del st.session_state.profile_chat
            st.rerun()


with chat_tab:
    render_chat()

with form_tab:
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
                "free_minutes": st.column_config.NumberColumn(
                    "Free minutes after school", min_value=0, max_value=720, step=15,
                    help="All your free time that day. Don't subtract activities - NorthStar does that from the list below.",
                ),
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
