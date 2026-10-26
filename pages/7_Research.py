"""Research Studio: a guided 8-step mini-project. The student writes; the coach asks; sources are real and stored."""
from datetime import date

import streamlit as st
from pydantic import ValidationError

from northstar.agents.research_agent import CoachError
from northstar.db.session import get_session, init_db
from northstar.graph import run
from northstar.llm import ai_problem
from northstar.schemas import SourceIn
from northstar.services.niches import suggestions
from northstar.services.profile import demo_student_id, get_profile
from northstar.services.research import (
    ARTIFACT_TYPES, ETHICS, STEP_GUIDE, active_project, add_library_source, add_source, build_report, ethics_problems,
    finish_project, past_projects, remove_source, save_step, set_artifact, set_ethics, start_project,
    suggest_library_sources, update_source,
)

st.set_page_config(page_title="Research Studio · NorthStar", page_icon="⭐", layout="wide")
init_db()
today = date.today()

with get_session() as s:
    student_id = demo_student_id(s)
    profile = get_profile(s, student_id) if student_id else None
if profile is None:
    st.info("First, tell NorthStar about yourself.")
    st.page_link("pages/1_Onboarding.py", label="Go to onboarding", icon="📝")
    st.stop()

with get_session() as s:
    project = active_project(s, student_id)
    chosen = suggestions(s, student_id, "chosen")
    history = past_projects(s, student_id)

st.title("Research Studio")
st.caption("Turn a curiosity into a small research or creation project. You do the thinking and writing - "
           "NorthStar's coach asks questions, and every source is real and saved.")


def download(p, key: str) -> None:
    report, _ = build_report(p, profile.alias, today)
    st.download_button("⬇️ Download report (Markdown)", report, file_name=f"{p.title[:40]}.md", mime="text/markdown", key=key)


# ---- No project yet -----------------------------------------------------------------------------
if project is None:
    with st.form("new_project"):
        st.subheader("Start a project")
        options = {"(none)": None} | {n.niche_name: n.id for n in chosen}
        niche = st.selectbox("Based on a niche you explored", list(options),
                             help="Niches you tried on the Explore page")
        title = st.text_input("Working title", placeholder="e.g. How does launch angle change a game character's jump?")
        if st.form_submit_button("Start project", type="primary"):
            try:
                with get_session() as s:
                    start_project(s, student_id, title, options[niche])
            except ValueError as e:
                st.error(str(e))
            else:
                st.rerun()
    if not chosen:
        st.page_link("pages/6_Explore.py", label="Explore niches first (optional)", icon="🧭")
    for p in history:
        with st.expander(f"📁 {p.title} ({p.status})"):
            download(p, key=f"dl_{p.id}")
    st.stop()

# ---- Active project -------------------------------------------------------------------------------
done = sum(step.done for step in project.steps)
st.subheader(f"📁 {project.title}")
st.progress(done / len(project.steps), text=f"{done} of {len(project.steps)} steps done")
# A keyed segmented control (not st.tabs) so the chosen section survives the rerun after every button press
labels = {"steps": "🪜 Steps", "sources": f"📚 Sources ({len(project.sources)})", "report": "📄 Report"}
section = st.segmented_control("Section", list(labels), format_func=labels.get, default="steps",
                               key="research_section", label_visibility="collapsed") or "steps"
ai_off = ai_problem()

if section == "steps":
    if ai_off:
        st.warning(ai_off)
    first_open = next((step.id for step in project.steps if not step.done), None)
    for i, step in enumerate(project.steps, 1):
        heading, purpose = STEP_GUIDE[step.step]
        with st.expander(f"{'✅' if step.done else '⬜'} {i}. {heading}", expanded=step.id == first_open):
            st.caption(purpose)
            with st.form(f"step_{step.id}"):
                notes = st.text_area("Your notes", step.notes, height=150, key=f"notes_{step.id}",
                                     placeholder="Write in your own words. Cite sources as [S1], [S2]…" if step.step == "background" else "")
                ethics = {}
                if step.step == "method":
                    st.markdown("**Ethics checklist**")
                    ethics = {k: st.checkbox(text, value=project.ethics.get(k, False), key=f"eth_{k}") for k, text in ETHICS.items()}
                if step.step == "artifact":
                    c1, c2 = st.columns([1, 2])
                    kind = c1.selectbox("Type", ARTIFACT_TYPES, index=ARTIFACT_TYPES.index(project.artifact_type)
                                        if project.artifact_type in ARTIFACT_TYPES else 0)
                    link = c2.text_input("Link to it (optional)", project.artifact_link or "")
                is_done = st.checkbox("This step is done", value=step.done, key=f"done_{step.id}")
                buttons = st.container(horizontal=True)
                save = buttons.form_submit_button("Save", type="primary")
                ask = buttons.form_submit_button("💬 Save & ask the coach", disabled=bool(ai_off))
            if save or ask:
                try:
                    with get_session() as s:
                        if step.step == "method":
                            set_ethics(s, student_id, project.id, ethics)
                        if step.step == "artifact":
                            set_artifact(s, student_id, project.id, kind, link)
                        save_step(s, student_id, step.id, notes, is_done)
                except ValueError as e:
                    st.error(str(e))
                    if step.step == "method":
                        st.markdown("\n".join(f"- {p}" for p in ethics_problems(ethics)))
                else:
                    if ask:
                        try:
                            with st.spinner("The coach is reading your notes…"):
                                run("coach_step", student_id, project_id=project.id, step_id=step.id)
                        except CoachError as e:
                            st.error(f"{e}. Please try again.")
                            st.stop()
                        except Exception as e:  # network / API key problems
                            st.error(f"Couldn't reach the AI: {e}")
                            st.stop()
                    st.rerun()
            if step.coach:
                with st.container(border=True):
                    st.markdown(f"**💬 Coach:** {step.coach['feedback']}")
                    st.markdown("\n".join(f"- {q}" for q in step.coach["questions"]))
                    st.markdown(f"**Next:** {step.coach['next_action']}")
                    for flag in step.coach["ethics_flags"]:
                        st.warning(f"⚠️ {flag}")

if section == "sources":
    st.caption("Only these can be cited in your report, as [S1], [S2]… Numbers never change, even if you remove one.")
    for src in project.sources:
        with st.container(border=True):
            st.markdown(f"**[S{src.number}] {src.title}**" + (f" - {src.publisher}" if src.publisher else "")
                        + f"  \n{src.url}  \n:gray[{'from NorthStar’s library' if src.added_by == 'library' else 'added by you'}]")
            with st.form(f"src_{src.id}"):
                note = st.text_input("What does it say / why is it useful? (your words)", src.note)
                opened = st.checkbox("I opened it and checked it", value=src.opened)
                buttons = st.container(horizontal=True)
                if buttons.form_submit_button("Save"):
                    with get_session() as s:
                        update_source(s, student_id, src.id, note, opened)
                    st.rerun()
                if buttons.form_submit_button("Remove"):
                    with get_session() as s:
                        remove_source(s, student_id, src.id)
                    st.rerun()

    st.markdown("**Suggested from NorthStar's library**")
    question = next((step.notes for step in project.steps if step.step == "question"), "")
    niche_name = next((n.niche_name for n in chosen if n.id == project.niche_id), "")
    matches = suggest_library_sources(" ".join([project.title, question, niche_name, *profile.interests]),
                                      {src.url for src in project.sources})
    if not matches:
        st.caption("No library matches yet - write your research question in step 1, or add your own source below.")
    for item in matches:
        left, right = st.columns([5, 1], vertical_alignment="center")
        left.markdown(f"**{item['title']}** - {item['publisher']}  \n:gray[{item['url']}]")
        if right.button("Add", key=f"lib_{item['url']}"):
            with get_session() as s:
                add_library_source(s, student_id, project.id, item["url"])
            st.rerun()

    with st.form("own_source", clear_on_submit=True):
        st.markdown("**Add your own source**")
        c1, c2 = st.columns(2)
        title = c1.text_input("Title")
        publisher = c2.text_input("Who made it (optional)")
        url = st.text_input("Link", placeholder="https://…")
        note = st.text_input("What does it say? (optional)")
        if st.form_submit_button("Add source"):
            try:
                src = SourceIn(title=title, url=url.strip(), publisher=publisher, note=note)
            except ValidationError as e:
                st.error("Please check: " + "; ".join(("link: paste a full web address starting with https://" if err["loc"][0] == "url"
                                                        else f"{err['loc'][0]}: {err['msg']}") for err in e.errors()))
            else:
                with get_session() as s:
                    add_source(s, student_id, project.id, src)
                st.rerun()

if section == "report":
    report, warnings = build_report(project, profile.alias, today)
    st.caption("Built only from your own notes. Citations that aren't your sources are removed.")
    for w in warnings:
        st.warning(w)
    if not warnings:
        st.success("Every citation points to one of your sources. ✅")
    with st.container(border=True):
        st.markdown(report)
    download(project, key="dl_active")
    if st.checkbox("I've finished this project") and st.button("Mark project finished"):
        with get_session() as s:
            finish_project(s, student_id, project.id)
        st.rerun()
