"""Research Studio: a guided 8-step mini-project. The student writes; the coach asks; sources are real and stored."""
from datetime import date

import streamlit as st
from pydantic import ValidationError

from northstar.agents.research_agent import CoachError
from northstar.db.models import DatasetRecommendation, ImportedDataset, KaggleDataset
from northstar.db.session import get_session, init_db
from northstar.graph import run
from northstar.llm import ai_problem
from northstar.schemas import SourceIn
from northstar.services.kaggle_dataset_service import (
    attach_dataset_as_source,
    import_dataset,
    preview_dataset,
    recommend_datasets,
    run_starter_eda,
)
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
labels = {
    "steps": "🪜 Steps",
    "sources": f"📚 Sources ({len(project.sources)})",
    "data": "🗂️ Data",
    "report": "📄 Report",
}
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

if section == "data":
    st.info("💡 **Suggestion only:** Datasets are for observation and exploration. Correlation does not imply causation — patterns in data suggest questions, not proven causes.")

    question_step = next((step for step in project.steps if step.step == "question"), None)
    has_question = bool(question_step and question_step.notes.strip())

    top_c1, top_c2 = st.columns([3, 1], vertical_alignment="center")
    top_c1.markdown("### 🗂️ Dataset Recommendations & Evidence")
    find_clicked = top_c2.button(
        "🔍 Find datasets",
        disabled=not has_question or bool(ai_off),
        type="primary",
        help=(
            "Write your research question in Step 1 first"
            if not has_question
            else ("AI is off" if ai_off else "Get curated dataset recommendations")
        ),
    )

    if not has_question:
        st.caption("Complete step 1 (**Research Question**) with your notes first to get tailored dataset recommendations.")

    if find_clicked:
        with st.spinner("Finding matching datasets from the curated catalog…"):
            try:
                run("recommend_datasets", student_id, project_id=project.id)
            except Exception:
                with get_session() as s:
                    recommend_datasets(s, student_id, project.id)
        st.rerun()

    with get_session() as s:
        recs = s.query(DatasetRecommendation).filter(
            DatasetRecommendation.research_project_id == project.id,
            DatasetRecommendation.student_id == student_id,
        ).all()
        recs_data = [
            (r.id, r.match_reason, r.status, r.kaggle_dataset)
            for r in recs if r.kaggle_dataset is not None
        ]

        imported_records = s.query(ImportedDataset).filter(
            ImportedDataset.research_project_id == project.id,
            ImportedDataset.student_id == student_id,
        ).order_by(ImportedDataset.id.desc()).all()
        imported = imported_records[0] if imported_records else None
        imported_ready = imported if (imported and imported.status == "ready") else None

    # ---- Imported Dataset Section ----
    if imported_ready and imported_ready.kaggle_dataset:
        ds = imported_ready.kaggle_dataset
        with st.container(border=True):
            st.markdown(f"#### 📊 Active Dataset: {ds.title}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{imported_ready.row_count:,}" if imported_ready.row_count else "N/A")
            c2.metric("Columns", f"{imported_ready.column_count}" if imported_ready.column_count else "N/A")
            c3.metric("Size", f"{(imported_ready.size_bytes or 0) / 1024:.1f} KB")
            c4.metric("License", ds.license)

            st.caption(f"**Kaggle Reference:** `{ds.kaggle_ref}` · [View on Kaggle]({ds.source_url})")

            already_attached = next((src for src in project.sources if ds.source_url and src.url == ds.source_url), None)

            action_c1, action_c2 = st.columns([1, 1])
            run_eda = action_c1.button("📈 Run Starter EDA", key="btn_run_eda", type="primary")
            if already_attached:
                action_c2.success(f"✅ Attached as research source [S{already_attached.number}]")
            else:
                if action_c2.button("📌 Attach as research evidence", key="btn_attach_source"):
                    with get_session() as s:
                        num = attach_dataset_as_source(s, student_id, project.id, ds.kaggle_ref)
                    st.success(f"Attached as source [S{num}]! You can now cite it in your report as [S{num}].")
                    st.rerun()

            with st.expander("👀 Preview dataset sample rows", expanded=not run_eda and f"eda_active_{project.id}" not in st.session_state):
                try:
                    with get_session() as s:
                        p_data = preview_dataset(s, student_id, project.id, ds.kaggle_ref)
                    if p_data.get("sample_rows"):
                        import pandas as pd
                        st.dataframe(pd.DataFrame(p_data["sample_rows"]), use_container_width=True)
                    else:
                        st.caption("No sample rows available to preview.")
                except Exception as e:
                    st.error(f"Could not load preview: {e}")

            if run_eda or st.session_state.get(f"eda_active_{project.id}"):
                st.session_state[f"eda_active_{project.id}"] = True
                with st.spinner("Running deterministic pandas EDA templates…"):
                    try:
                        with get_session() as s:
                            eda = run_starter_eda(s, student_id, project.id, dataset_id=imported_ready.id)
                    except Exception as e:
                        st.error(f"EDA failed: {e}")
                    else:
                        st.markdown("---")
                        st.markdown("#### 🔬 Starter EDA Output")
                        st.warning(f"⚠️ {eda.correlation_reminder}")

                        eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(["Head & Summary", "Missing Values", "Categorical Counts", "Column Types"])

                        with eda_tab1:
                            st.markdown("**First 5 rows (df.head()):**")
                            import pandas as pd
                            st.dataframe(pd.DataFrame(eda.head_records), use_container_width=True)

                            if eda.numeric_summary:
                                st.markdown("**Numeric Summary (df.describe()):**")
                                st.dataframe(pd.DataFrame(eda.numeric_summary).T, use_container_width=True)

                        with eda_tab2:
                            st.markdown("**Missing Values per Column (df.isnull().sum()):**")
                            null_data = [{"Column": col, "Missing Count": count} for col, count in eda.missing_values.items()]
                            import pandas as pd
                            st.dataframe(pd.DataFrame(null_data), use_container_width=True)

                        with eda_tab3:
                            if eda.categorical_value_counts:
                                st.markdown("**Low-Cardinality Categorical Distributions:**")
                                import pandas as pd
                                for col_name, counts in eda.categorical_value_counts.items():
                                    st.markdown(f"**Column: `{col_name}`**")
                                    st.bar_chart(pd.Series(counts))
                            else:
                                st.caption("No low-cardinality categorical columns detected.")

                        with eda_tab4:
                            st.markdown("**Column Data Types (df.info() equivalent):**")
                            col_info = [
                                {
                                    "Column": c.name,
                                    "Type": c.dtype,
                                    "Non-Null": c.non_null_count,
                                    "Unique Values": c.unique_count,
                                    "Samples": ", ".join(c.sample_values),
                                }
                                for c in eda.column_summaries
                            ]
                            import pandas as pd
                            st.dataframe(pd.DataFrame(col_info), use_container_width=True)

                        st.caption("📝 **Tip:** Copy key numbers or findings above into your **Analysis** step notes (Step 6)!")

    # ---- Recommendations Cards Section ----
    if recs_data:
        st.markdown("### 💡 Recommended Datasets for Your Project")
        st.caption("Curated, age-appropriate, size-limited datasets from Kaggle.")

        for rec_id, match_reason, status, ds in recs_data:
            with st.container(border=True):
                r_c1, r_c2 = st.columns([4, 1], vertical_alignment="center")
                r_c1.markdown(f"**{ds.title}**  \n:gray[`{ds.kaggle_ref}` · {ds.difficulty.capitalize()} · {ds.size_mb:.2f} MB · License: {ds.license}]")

                is_current = imported_ready and (imported_ready.kaggle_dataset_id == ds.id)
                if is_current:
                    r_c2.markdown(":green[**✅ Active**]")
                else:
                    if r_c2.button("📥 Use this dataset", key=f"use_rec_{rec_id}", type="secondary"):
                        with st.spinner(f"Importing {ds.title}…"):
                            try:
                                with get_session() as s:
                                    import_dataset(s, student_id, project.id, ds.kaggle_ref)
                                    rec_row = s.get(DatasetRecommendation, rec_id)
                                    if rec_row:
                                        rec_row.status = "chosen"
                                    s.commit()
                                st.success(f"Imported {ds.title}!")
                                st.rerun()
                            except ValueError as e:
                                st.error(str(e))
                            except Exception as e:
                                st.error(f"Import failed: {e}")

                st.markdown(f"**Why it matches:** {match_reason}")
                st.markdown(f"{ds.description}")

                if ds.attribution_required and ds.attribution_text:
                    st.caption(f"**Attribution Required:** {ds.attribution_text}")

                with st.expander("Questions to explore & limitations"):
                    if ds.sample_questions:
                        st.markdown("**Sample research questions:**")
                        for q in ds.sample_questions:
                            st.markdown(f"- {q}")
                    if ds.limitations:
                        st.markdown("**Important limitations:**")
                        for lim in ds.limitations:
                            st.markdown(f"- {lim}")
                    st.markdown(f"🔗 [Open on Kaggle]({ds.source_url})")
    elif has_question:
        st.info("Click **🔍 Find datasets** above to discover matching datasets from the curated catalog.")

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
