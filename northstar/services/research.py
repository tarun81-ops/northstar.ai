"""Research studio: a guided 8-step mini-project. Everything here is plain Python - no LLM.

Hard rules, enforced in code:
- Sources come only from the curated library (data/sources.json) or links the student adds. Nothing is generated.
- A project may cite only its own stored sources, as [S1], [S2]... The report builder removes anything else and warns.
- The method step can't be marked done until the ethics checklist is complete.
- The report is assembled from the STUDENT's notes - NorthStar never writes their work.
"""
import json
import re
from datetime import date
from functools import lru_cache

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from northstar.config import PROJECT_ROOT
from northstar.db.models import RESEARCH_STEPS, NicheSuggestion, ProjectStep, ResearchProject, Source
from northstar.schemas import CoachReply, SourceIn

# step -> (heading, what the student should write)
STEP_GUIDE = {
    "question": ("Research question", "What exactly do you want to find out? Make it specific enough to answer in 2-4 weeks."),
    "background": ("Background", "What do credible sources already say? Add sources, then write what you learned, citing them as [S1], [S2]..."),
    "hypothesis": ("Hypothesis", "What do you predict, and why? Or write the guiding question you'll explore."),
    "method": ("Method", "Step by step, how will you find out? Keep it simple, free and safe, and complete the ethics checklist."),
    "data": ("Data", "What data did you collect or find? Where is it? Link public data as a source."),
    "analysis": ("Analysis", "What patterns do you see? Does the data support your hypothesis? Remember: correlation isn't causation."),
    "artifact": ("Artifact", "What did you make to share your findings - a report, poster, video, dashboard, code or presentation?"),
    "reflection": ("Limitations and next steps", "What could have affected your results? What would you do differently, or explore next?"),
}
ETHICS = {
    "no_personal_data": "I won't collect names, contacts or anything that identifies a person.",
    "safe": "Nothing in my method is dangerous, and it costs nothing I can't afford.",
    "involves_people": "My project involves other people (a survey, interviews or observing people).",
    "permission": "A teacher or parent has said OK, and everyone taking part can say no.",
    "anonymous": "Answers are anonymous - I can't tell who said what.",
}
ARTIFACT_TYPES = ("report", "poster", "video", "dashboard", "code project", "presentation")
CITATION = re.compile(r"\[S(\d+)\]")
LINK = re.compile(r"https?://[^\s)\]>]+")
LIBRARY_SUGGESTIONS = 6


# ---- curated source library ------------------------------------------------------------

@lru_cache
def library() -> list[dict]:
    return json.loads((PROJECT_ROOT / "data" / "sources.json").read_text(encoding="utf-8"))["sources"]


def suggest_library_sources(text: str, exclude_urls: set[str]) -> list[dict]:
    """Library sources whose tags overlap the words in `text` (project title, question, niche). Keyword match, no LLM."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    scored = []
    for item in library():
        score = sum(1 for tag in item["tags"] if set(tag.split()) <= words or tag in words)
        if score and item["url"] not in exclude_urls:
            scored.append((-score, item["title"], item))
    return [item for *_, item in sorted(scored)[:LIBRARY_SUGGESTIONS]]


# ---- projects and steps --------------------------------------------------------------------

def _project(session: Session, student_id: int, project_id: int) -> ResearchProject:
    project = session.get(ResearchProject, project_id)
    if project is None or project.student_id != student_id:
        raise ValueError(f"Project {project_id} is not this student's")
    return project


def get_project(session: Session, student_id: int, project_id: int) -> ResearchProject:
    project = _project(session, student_id, project_id)
    _ = project.steps, project.sources  # load both before the session closes
    return project


def active_project(session: Session, student_id: int) -> ResearchProject | None:
    return session.scalar(select(ResearchProject)
                          .where(ResearchProject.student_id == student_id, ResearchProject.status == "active")
                          .options(selectinload(ResearchProject.steps), selectinload(ResearchProject.sources)))


def start_project(session: Session, student_id: int, title: str, niche_id: int | None = None) -> int:
    """One project at a time, with all 8 steps ready to fill in."""
    title = title.strip()
    if not 3 <= len(title) <= 200:
        raise ValueError("Give your project a title of 3-200 characters")
    if active_project(session, student_id):
        raise ValueError("Finish your current project first")
    if niche_id is not None:
        niche = session.get(NicheSuggestion, niche_id)
        if niche is None or niche.student_id != student_id:
            raise ValueError("That niche isn't yours")
    project = ResearchProject(student_id=student_id, niche_id=niche_id, title=title,
                              steps=[ProjectStep(order_index=i, step=s) for i, s in enumerate(RESEARCH_STEPS)])
    session.add(project)
    session.commit()
    return project.id


def ethics_problems(ethics: dict) -> list[str]:
    missing = [k for k in ("no_personal_data", "safe") if not ethics.get(k)]
    if ethics.get("involves_people"):
        missing += [k for k in ("permission", "anonymous") if not ethics.get(k)]
    return [ETHICS[k] for k in missing]


def set_ethics(session: Session, student_id: int, project_id: int, ethics: dict) -> None:
    _project(session, student_id, project_id).ethics = {k: bool(ethics.get(k)) for k in ETHICS}
    session.commit()


def save_step(session: Session, student_id: int, step_id: int, notes: str, done: bool) -> None:
    step = session.get(ProjectStep, step_id)
    project = _project(session, student_id, step.project_id) if step else None
    if step is None:
        raise ValueError(f"Step {step_id} not found")
    if done and step.step == "method" and ethics_problems(project.ethics):
        raise ValueError("Complete the ethics checklist before marking the method done")
    step.notes, step.done = notes.strip(), done
    session.commit()


def save_coach_reply(session: Session, student_id: int, step_id: int, reply: CoachReply) -> None:
    step = session.get(ProjectStep, step_id)
    _project(session, student_id, step.project_id)
    step.coach = reply.model_dump()
    session.commit()


def set_artifact(session: Session, student_id: int, project_id: int, kind: str, link: str) -> None:
    if kind not in ARTIFACT_TYPES:
        raise ValueError(f"Artifact type must be one of {ARTIFACT_TYPES}")
    project = _project(session, student_id, project_id)
    project.artifact_type, project.artifact_link = kind, (link.strip() or None)
    session.commit()


def finish_project(session: Session, student_id: int, project_id: int) -> None:
    _project(session, student_id, project_id).status = "done"
    session.commit()


def past_projects(session: Session, student_id: int) -> list[ResearchProject]:
    return list(session.scalars(select(ResearchProject).where(ResearchProject.student_id == student_id,
                                                              ResearchProject.status != "active")
                                .options(selectinload(ResearchProject.steps), selectinload(ResearchProject.sources))
                                .order_by(ResearchProject.id.desc())))


# ---- sources ------------------------------------------------------------------------------

def _next_number(session: Session, project_id: int) -> int:
    return (session.scalar(select(func.max(Source.number)).where(Source.project_id == project_id)) or 0) + 1


def add_source(session: Session, student_id: int, project_id: int, source: SourceIn) -> int:
    _project(session, student_id, project_id)
    number = _next_number(session, project_id)
    session.add(Source(project_id=project_id, number=number, title=source.title, url=str(source.url),
                       publisher=source.publisher, note=source.note, added_by="student"))
    session.commit()
    return number


def add_library_source(session: Session, student_id: int, project_id: int, url: str) -> int:
    """Only items that really are in the curated library can be added as library sources."""
    item = next((i for i in library() if i["url"] == url), None)
    if item is None:
        raise ValueError("That isn't in NorthStar's library")
    _project(session, student_id, project_id)
    number = _next_number(session, project_id)
    session.add(Source(project_id=project_id, number=number, title=item["title"], url=item["url"],
                       publisher=item["publisher"], added_by="library"))
    session.commit()
    return number


def update_source(session: Session, student_id: int, source_id: int, note: str, opened: bool) -> None:
    source = session.get(Source, source_id)
    _project(session, student_id, source.project_id)
    source.note, source.opened = note.strip(), opened
    session.commit()


def remove_source(session: Session, student_id: int, source_id: int) -> None:
    source = session.get(Source, source_id)
    _project(session, student_id, source.project_id)
    session.delete(source)
    session.commit()


# ---- citations and the report -------------------------------------------------------------

def _same_link(a: str, b: str) -> bool:
    return a.rstrip("/").lower() == b.rstrip("/").lower()


def build_report(project: ResearchProject, alias: str, today: date) -> tuple[str, list[str]]:
    """Markdown report made ONLY from the student's notes. Citations that aren't stored sources are removed;
    links that aren't sources, and cited sources not marked as opened, are flagged. Returns (report, warnings)."""
    numbers = {s.number: s for s in project.sources}
    warnings: list[str] = []
    lines = [f"# {project.title}", f"*A research project by {alias} · {today:%d %B %Y}*", ""]

    for step in project.steps:
        if not step.notes.strip():
            continue
        heading = STEP_GUIDE[step.step][0]

        def keep_known(match: re.Match) -> str:
            if int(match.group(1)) in numbers:
                return match.group(0)
            warnings.append(f"{heading}: [S{match.group(1)}] isn't one of your sources, so it was removed.")
            return ""

        text = re.sub(r"\s?" + CITATION.pattern, keep_known, step.notes)  # also drop the space before a removed citation
        for link in LINK.findall(text):
            if not any(_same_link(link, s.url) for s in project.sources):
                warnings.append(f"{heading}: the link {link} isn't one of your sources - add it as a source and cite it.")
        lines += [f"## {heading}", text.strip(), ""]

    if project.artifact_type:
        lines += ["## My artifact", f"{project.artifact_type.capitalize()}" + (f": {project.artifact_link}" if project.artifact_link else ""), ""]

    cited = sorted({int(n) for n in CITATION.findall("\n".join(lines))})
    if cited:
        lines.append("## References")
        for n in cited:
            s = numbers[n]
            lines.append(f"- [S{n}] {s.title}" + (f" - {s.publisher}" if s.publisher else "") + f". {s.url}")
            if not s.opened:
                warnings.append(f"You cite [S{n}] ({s.title}) but haven't marked it as opened - check it says what you wrote.")
    others = [s for s in project.sources if s.number not in cited]
    if others:
        lines += ["", "## Other sources I explored"] + [f"- {s.title}. {s.url}" for s in others]
    return "\n".join(lines).strip() + "\n", warnings
