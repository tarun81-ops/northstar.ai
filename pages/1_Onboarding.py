"""Onboarding: chat with the Profile Agent, or fill in the plain form. Both save the same profile."""
import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError
from uuid import uuid4

from northstar.agents.profile_agent import build_profile_agent, greeting
from northstar.ai.chat_recovery import ChatRecoveryError, classify_error, invoke_with_fallback
from northstar.ai.model_factory import OpenRouterModelSelectionError
from northstar.ui.model_selector import (
    FALLBACK_NOTICE_KEY,
    LAST_RESPONDING_MODEL_KEY,
    clear_error,
    render_model_selector,
    set_error,
)
from northstar.ai import llm_config, providers
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

# Model selector in sidebar - stores the user's chosen model in session state
selected_model = render_model_selector(required_capabilities={"tool_calling"})

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


PROFILE_CHAT_BACKEND = "openrouter-free-profile-v2"


def _profile_chat_model(selected: str) -> str | None:
    model = llm_config.provider_prefix(selected)
    return model if providers.provider_for_model(model) == "openrouter" else None


def render_chat() -> None:
    """The chat tab. Uses `return` (not st.stop) on problems, so the Form tab below still renders."""
    ai_off = None
    if not providers.has_api_key("openrouter"):
        ai_off = (
            "NorthStar's profile chat is switched off: no OPENROUTER_API_KEY was found. "
            "Add it to the .env file or Streamlit secrets. The profile form still works."
        )
    if ai_off and "profile_chat" not in st.session_state:
        st.warning(ai_off + " Use the Form tab to fill in your profile.")
        return
    if ai_off:
        st.warning(ai_off)
    requested_model = _profile_chat_model(selected_model)
    if providers.provider_for_model(llm_config.provider_prefix(selected_model)) != "openrouter":
        st.info("Profile chat uses current free OpenRouter models with tool-calling support; other provider selections are not used.")
    if "profile_chat" not in st.session_state:
        try:
            agent, state = build_profile_agent(student_id, model=requested_model)
        except Exception as error:
            category = classify_error(error)
            if isinstance(error, OpenRouterModelSelectionError):
                st.warning(f"{error} You can use the Form tab instead.")
            else:
                st.warning(f"Chat is unavailable ({category}). You can use the Form tab instead.")
            return
        st.session_state.profile_chat = {
            "agent": agent,
            "model": state["_model"],
            "requested_model": requested_model,
            "backend": PROFILE_CHAT_BACKEND,
            "model_valid": True,
            "state": state,
            "tools": state["_tools"],
            "checkpointer": state["_checkpointer"],
            "thread_id": uuid4().hex,
            "messages": [AIMessage(greeting(current.alias if current else None))],
            "pending": False,
            "incomplete": False,
        }
    chat = st.session_state.profile_chat
    if (
        chat.get("backend") != PROFILE_CHAT_BACKEND
        or chat.get("requested_model") != requested_model
    ):
        try:
            agent, state = build_profile_agent(
                student_id,
                model=requested_model,
                tools=chat["tools"],
                state=chat["state"],
                checkpointer=chat["checkpointer"],
            )
            chat["agent"] = agent
            chat["model"] = state["_model"]
            chat["requested_model"] = requested_model
            chat["backend"] = PROFILE_CHAT_BACKEND
            chat["model_valid"] = True
            chat.pop("error_category", None)
            clear_error()
        except Exception as error:
            category = classify_error(error)
            chat["agent"] = None
            chat["model"] = None
            chat["requested_model"] = requested_model
            chat["backend"] = PROFILE_CHAT_BACKEND
            chat["model_valid"] = False
            message = (
                str(error)
                if isinstance(error, OpenRouterModelSelectionError)
                else f"The selected model cannot run profile chat ({category}). "
                "Choose a current free OpenRouter model with tool-calling support."
            )
            set_error(message)

    left, right = st.columns([3, 2])
    with left:
        st.caption(f"Profile chat model: {chat.get('model', 'Unavailable')}")
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
        if chat.get("incomplete"):
            box.warning("The previous reply did not complete. Its pending request is preserved.")

        if prompt := st.chat_input(
            "Type your answer…",
            disabled=(
                chat.get("pending", False)
                or not chat.get("model_valid", False)
                or ai_off is not None
            ),
        ):
            message = HumanMessage(prompt, id=uuid4().hex)
            chat["messages"].append(message)
            chat["pending_message_id"] = message.id
            chat["pending"] = True
            chat["incomplete"] = True
            with st.spinner("Thinking…"):
                _run_pending_chat(chat)
            st.rerun()

    with right:
        st.markdown("#### Your profile so far")
        sid = chat["state"]["student_id"]
        with get_session() as s:
            profile_card(get_profile(s, sid) if sid else None)
        if chat["state"]["draft"]:
            st.caption("Not saved yet (need nickname, age and grade): " + ", ".join(chat["state"]["draft"]))
        if chat.get("pending"):
            st.warning("Your request is still pending. Choose another model in the sidebar, then retry.")
            if st.button(
                "Choose another model and retry",
                key="northstar_retry_pending",
                disabled=not chat.get("model_valid", False) or ai_off is not None,
            ):
                with st.spinner("Retrying the pending request…"):
                    _run_pending_chat(chat)
                st.rerun()
        if st.button("Start the chat over"):
            del st.session_state.profile_chat
            st.rerun()


def _run_pending_chat(chat: dict) -> None:
    """Resume the checkpointed inference without replaying completed tool nodes."""
    thread_config = {"configurable": {"thread_id": chat["thread_id"]}}

    def invoke_model(model: str, _resume: bool):
        if chat["model"] != model:
            agent, _ = build_profile_agent(
                chat["state"]["student_id"],
                model=model,
                tools=chat["tools"],
                state=chat["state"],
                checkpointer=chat["checkpointer"],
            )
            chat["agent"] = agent
            chat["model"] = model
        snapshot = chat["agent"].get_state(thread_config)
        checkpoint_messages = snapshot.values.get("messages", [])
        pending_id = chat.get("pending_message_id")
        pending_checkpointed = any(
            getattr(message, "id", None) == pending_id for message in checkpoint_messages
        )
        if pending_checkpointed:
            return chat["agent"].invoke(None, config=thread_config)
        return chat["agent"].invoke(
            {"messages": chat["messages"]},
            config=thread_config,
        )

    cooldowns = st.session_state.setdefault("northstar_model_cooldowns", {})
    try:
        result = invoke_with_fallback(
            invoke_model,
            chat["model"],
            enabled=False,
            cooldowns=cooldowns,
        )
    except ChatRecoveryError as error:
        category = error.category
        messages = {
            "rate_limit": "The provider rate-limited this request. Wait for its retry window or choose another model.",
            "daily_quota": "The OpenRouter account quota or credits appear exhausted. Wait for the quota to reset; the pending request and conversation are preserved.",
            "credentials": "OpenRouter rejected its credentials. Check the server-side OpenRouter key.",
            "configuration": "The selected model does not support a required chat parameter or capability.",
            "safety": "The provider declined this request for safety reasons; fallback was not attempted.",
            "outage": "The provider is temporarily unavailable. Choose another model or retry later.",
            "unknown": "The provider could not complete this request.",
        }
        chat["pending"] = True
        chat["incomplete"] = True
        chat["error_category"] = category
        message = messages.get(category, messages["unknown"])
        if category in {"rate_limit", "daily_quota", "outage"}:
            message += " The pending request and conversation are preserved."
        if error.retry_after_seconds:
            message += f" Provider retry guidance: wait at least {error.retry_after_seconds} seconds."
        set_error(message)
        st.error(message)
        return

    chat["messages"] = result.value["messages"]
    chat["pending"] = False
    chat["incomplete"] = False
    chat.pop("pending_message_id", None)
    actual_model = result.actual_model
    for message in reversed(chat["messages"]):
        response_model = (getattr(message, "response_metadata", None) or {}).get("model_name")
        if getattr(message, "type", None) == "ai" and isinstance(response_model, str) and response_model:
            actual_model = (
                response_model
                if response_model.startswith("openrouter:")
                else f"openrouter:{response_model}"
            )
            break
    chat["actual_model"] = actual_model
    st.session_state[LAST_RESPONDING_MODEL_KEY] = actual_model
    clear_error()
    st.session_state.pop(FALLBACK_NOTICE_KEY, None)


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
