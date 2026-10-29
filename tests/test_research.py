from datetime import date

import pytest
from sqlalchemy.orm import Session, sessionmaker

from northstar import graph
from northstar.agents import research_agent
from northstar.db.seed import demo_profile
from northstar.db.session import init_db, make_engine
from northstar.schemas import CoachReply, ExperimentDraft, NicheDraft, SourceIn
from northstar.services.niches import save_suggestions, suggestions
from northstar.services.profile import save_profile
from northstar.services.research import (
    active_project, add_library_source, add_source, build_report, ethics_problems, get_project, library, remove_source,
    save_step, set_ethics, start_project, suggest_library_sources, update_source,
)

TODAY = date(2026, 11, 2)
PHET = "https://phet.colorado.edu/"


@pytest.fixture
def engine():
    e = make_engine("sqlite://")
    init_db(e)
    return e


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


@pytest.fixture
def sid(session):
    return save_profile(session, demo_profile())


@pytest.fixture
def pid(session, sid):
    return start_project(session, sid, "How does launch angle change a jump?")


def step(session, sid, pid, name):
    return next(st for st in get_project(session, sid, pid).steps if st.step == name)


# ---- projects, steps, ethics -------------------------------------------------------------

def test_project_has_8_steps_and_one_at_a_time(session, sid, pid):
    assert [s.step for s in active_project(session, sid).steps] == [
        "question", "background", "hypothesis", "method", "data", "analysis", "artifact", "reflection"]
    with pytest.raises(ValueError, match="Finish your current project"):
        start_project(session, sid, "Another one")


def test_project_can_only_use_your_own_niche(session, sid):
    other = save_profile(session, demo_profile().model_copy(update={"alias": "Meera"}))
    save_suggestions(session, other, [NicheDraft(niche_name="Game physics", description="d", why="w", matched_signal_ids=[],
                                                 experiment=ExperimentDraft(title="Try it", duration_days=5,
                                                                            steps=["Step one", "Step two", "Step three"]))])
    with pytest.raises(ValueError, match="isn't yours"):
        start_project(session, sid, "Borrowed niche", suggestions(session, other)[0].id)


def test_ethics_rules():
    assert len(ethics_problems({})) == 2
    assert ethics_problems({"no_personal_data": True, "safe": True}) == []
    people = {"no_personal_data": True, "safe": True, "involves_people": True}
    assert len(ethics_problems(people)) == 2  # permission + anonymous now required
    assert ethics_problems(people | {"permission": True, "anonymous": True}) == []


def test_method_cant_be_done_until_ethics_complete(session, sid, pid):
    method = step(session, sid, pid, "method")
    with pytest.raises(ValueError, match="ethics checklist"):
        save_step(session, sid, method.id, "Survey my class about sleep", done=True)
    set_ethics(session, sid, pid, {"no_personal_data": True, "safe": True, "involves_people": True,
                                   "permission": True, "anonymous": True})
    save_step(session, sid, method.id, "Anonymous class survey about sleep", done=True)
    assert step(session, sid, pid, "method").done


# ---- sources --------------------------------------------------------------------------------

def test_source_numbers_never_change(session, sid, pid):
    one = add_library_source(session, sid, pid, PHET)
    two = add_source(session, sid, pid, SourceIn(title="My notes", url="https://example.org/notes"))
    three = add_source(session, sid, pid, SourceIn(title="A video", url="https://example.org/video"))
    assert (one, two, three) == (1, 2, 3)
    remove_source(session, sid, next(s.id for s in get_project(session, sid, pid).sources if s.number == 2))
    assert [s.number for s in get_project(session, sid, pid).sources] == [1, 3]
    assert add_source(session, sid, pid, SourceIn(title="New", url="https://example.org/new")) == 4  # 2 is never reused


def test_library_only_accepts_library_items(session, sid, pid):
    with pytest.raises(ValueError, match="isn't in NorthStar's library"):
        add_library_source(session, sid, pid, "https://made-up-source.example/")


def test_library_suggestions_match_tags():
    titles = [i["title"] for i in suggest_library_sources("physics of motion in video games", set())]
    assert titles[0] == "PhET Interactive Simulations"
    assert "PhET Interactive Simulations" not in [i["title"] for i in suggest_library_sources("physics motion", {PHET})]
    assert all(i["url"].startswith("https://") for i in library())


# ---- the report: only stored sources can be cited -------------------------------------------

def test_report_keeps_real_citations_and_strips_the_rest(session, sid, pid):
    add_library_source(session, sid, pid, PHET)
    add_source(session, sid, pid, SourceIn(title="Unused page", url="https://example.org/unused"))
    save_step(session, sid, step(session, sid, pid, "question").id, "Does a steeper angle give a higher jump?", done=True)
    save_step(session, sid, step(session, sid, pid, "background").id,
              "Projectiles follow curved paths [S1]. Games copy this [S7]. See https://random.example/page", done=False)

    report, warnings = build_report(get_project(session, sid, pid), "Aarav", TODAY)
    assert "[S1]" in report and "[S7]" not in report
    assert "Games copy this. See" in report  # no stray space where the citation was removed
    assert "- [S1] PhET Interactive Simulations - University of Colorado Boulder. https://phet.colorado.edu/" in report
    assert "## Other sources I explored\n- Unused page" in report
    assert "## Hypothesis" not in report  # empty steps are left out
    assert any("[S7] isn't one of your sources" in w for w in warnings)
    assert any("https://random.example/page isn't one of your sources" in w for w in warnings)
    assert any("haven't marked it as opened" in w for w in warnings)

    update_source(session, sid, get_project(session, sid, pid).sources[0].id, "Shows projectile paths", opened=True)
    save_step(session, sid, step(session, sid, pid, "background").id, "Projectiles follow curved paths [S1].", done=True)
    assert build_report(get_project(session, sid, pid), "Aarav", TODAY)[1] == []


# ---- Research Coach (fake LLM, through the graph) -------------------------------------------

def reply(**kw) -> CoachReply:
    return CoachReply(**{"feedback": "Good start.", "questions": ["What exactly will you measure?"],
                         "next_action": "Write your question in one sentence."} | kw)


def test_check_coach():
    assert research_agent.check_coach(reply(), set()) == []
    assert "links" in research_agent.check_coach(reply(next_action="Read https://site.example"), set())[0]
    assert "no source(s) [3]" in research_agent.check_coach(reply(feedback="As [S3] says..."), {1})[0]
    assert "1-3 questions" in research_agent.check_coach(reply(questions=[]), set())[0]


class FakeLLM:
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.outputs.pop(0)


def test_coach_step_through_the_graph(engine, session, sid, pid, monkeypatch):
    monkeypatch.setattr(graph, "get_session", sessionmaker(engine))
    question = step(session, sid, pid, "question")
    save_step(session, sid, question.id, "Do games use real gravity?", done=False)
    fake = FakeLLM(reply(feedback="Try www.example.com"), reply())
    monkeypatch.setattr(research_agent, "get_llm", lambda *args, **kwargs: fake)

    graph.run("coach_step", sid, project_id=pid, step_id=question.id)
    assert "don't include any links" in fake.calls[1][-1].content
    assert "Do games use real gravity?" in fake.calls[0][1].content and "Aarav" not in fake.calls[0][1].content
    session.expire_all()  # the graph saved through its own session; re-read from the database
    assert step(session, sid, pid, "question").coach["questions"] == ["What exactly will you measure?"]
