from datetime import date

import pytest
from sqlalchemy.orm import Session, sessionmaker

from northstar import graph
from northstar.agents import niche_agent
from northstar.db.seed import demo_profile, seed_week
from northstar.db.session import init_db, make_engine
from northstar.schemas import ExperimentDraft, NicheDraft, NicheList, Reflection
from northstar.services.niches import (
    active_experiment, add_signal, check_niches, dismiss_niche, finish_experiment, list_signals, refresh_signals,
    remove_signal, save_suggestions, set_step, start_experiment, stop_experiment, suggestions,
)
from northstar.services.profile import save_profile

TODAY = date(2026, 11, 2)


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


def niche(name: str, ids: list[int], days: int = 7, steps: int = 3) -> NicheDraft:
    return NicheDraft(niche_name=name, description="What it is.", why="Because you like robots.", matched_signal_ids=ids,
                      experiment=ExperimentDraft(title=f"Try {name}", duration_days=days, steps=[f"Step {i}" for i in range(steps)]))


# ---- signals ------------------------------------------------------------------------

def test_signals_come_from_profile_checkins_and_the_student(session, sid):
    seed_week(session, sid, TODAY, "good")  # technical tasks rated 5/5
    add_signal(session, sid, "how video games use physics")
    texts = {(s.source, s.signal) for s in refresh_signals(session, sid, demo_profile())}
    assert ("profile", "Interested in Robotics") in texts
    assert ("profile", "Strong at Physics") in texts
    assert ("checkins", "Enjoys technical tasks") in texts
    assert ("checkins", "Loved the task: Wire the IR line sensors") in texts
    assert ("student", "how video games use physics") in texts


def test_refresh_keeps_ids_stable_and_drops_stale_evidence(session, sid):
    first = {s.signal: s.id for s in refresh_signals(session, sid, demo_profile())}
    fewer = demo_profile().model_copy(update={"interests": ["Robotics"]})
    second = {s.signal: s.id for s in refresh_signals(session, sid, fewer)}
    assert second["Interested in Robotics"] == first["Interested in Robotics"]  # same evidence, same id
    assert "Interested in Electronics" not in second


def test_only_student_signals_can_be_removed(session, sid):
    add_signal(session, sid, "space weather")
    signals = refresh_signals(session, sid, demo_profile())
    mine = next(s for s in signals if s.source == "student")
    theirs = next(s for s in signals if s.source == "profile")
    with pytest.raises(ValueError, match="Only things you added"):
        remove_signal(session, sid, theirs.id)
    remove_signal(session, sid, mine.id)
    with pytest.raises(ValueError):
        add_signal(session, sid, "x")


# ---- niches ---------------------------------------------------------------------------

def test_check_niches():
    ok = [niche("Robot path-finding", [1]), niche("Sports data", [2]), niche("Game physics", [1, 2])]
    assert check_niches(ok, {1, 2}) == []
    assert any("don't exist" in p for p in check_niches(ok[:2] + [niche("Made up", [99])], {1, 2}))
    assert any("at least one signal" in p for p in check_niches(ok[:2] + [niche("No evidence", [])], {1, 2}))
    assert any("3-5 niches" in p for p in check_niches(ok[:2], {1, 2}))
    assert any("3-14 days" in p for p in check_niches(ok[:2] + [niche("Too long", [1], days=30)], {1, 2}))
    assert any("different" in p for p in check_niches(ok[:2] + [niche("Sports data", [1])], {1, 2}))


def test_suggestions_store_a_snapshot_of_the_cited_signals(session, sid):
    signals = refresh_signals(session, sid, demo_profile())
    robotics = next(s for s in signals if s.signal == "Interested in Robotics")
    save_suggestions(session, sid, [niche("Robot path-finding", [robotics.id])])
    assert suggestions(session, sid)[0].matched_signals == [{"id": robotics.id, "signal": "Interested in Robotics"}]


# ---- experiments ------------------------------------------------------------------------

def test_experiment_lifecycle(session, sid):
    signals = refresh_signals(session, sid, demo_profile())
    save_suggestions(session, sid, [niche("Robot path-finding", [signals[0].id]), niche("Sports data", [signals[0].id])])
    first, second = suggestions(session, sid)

    exp_id = start_experiment(session, sid, first.id, TODAY)
    with pytest.raises(ValueError, match="Finish or stop"):
        start_experiment(session, sid, second.id, TODAY)  # one at a time

    set_step(session, sid, exp_id, 1, True)
    assert [s["done"] for s in active_experiment(session, sid).steps] == [False, True, False]

    assert finish_experiment(session, sid, exp_id, Reflection(curiosity=5, energy=4, confidence=3, meaning=4, notes="Fun!"))
    assert any(s.source == "experiment" and s.signal == "Enjoyed trying Robot path-finding" for s in list_signals(session, sid))
    assert active_experiment(session, sid) is None

    dismiss_niche(session, sid, second.id)
    assert suggestions(session, sid) == [] and [n.niche_name for n in suggestions(session, sid, "dismissed")] == ["Sports data"]


def test_low_reflection_or_stopping_adds_no_signal(session, sid):
    signals = refresh_signals(session, sid, demo_profile())
    save_suggestions(session, sid, [niche("Robot path-finding", [signals[0].id]), niche("Sports data", [signals[0].id])])
    first, second = suggestions(session, sid)
    assert not finish_experiment(session, sid, start_experiment(session, sid, first.id, TODAY),
                                 Reflection(curiosity=2, energy=2, confidence=2, meaning=3))
    stop_experiment(session, sid, start_experiment(session, sid, second.id, TODAY))
    assert not any(s.source == "experiment" for s in list_signals(session, sid))


def test_cannot_touch_another_students_experiment(session, sid):
    signals = refresh_signals(session, sid, demo_profile())
    save_suggestions(session, sid, [niche("Robot path-finding", [signals[0].id])])
    exp_id = start_experiment(session, sid, suggestions(session, sid)[0].id, TODAY)
    other = save_profile(session, demo_profile().model_copy(update={"alias": "Meera"}))
    with pytest.raises(ValueError, match="not this student's"):
        stop_experiment(session, other, exp_id)


# ---- through the graph (fake LLM) -------------------------------------------------------

class FakeLLM:
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.outputs.pop(0)


def test_suggest_niches_through_the_graph(engine, session, sid, monkeypatch):
    monkeypatch.setattr(graph, "get_session", sessionmaker(engine))
    signals = refresh_signals(session, sid, demo_profile())
    real = signals[0].id
    bad = NicheList(niches=[niche("Robot path-finding", [999]), niche("Sports data", [real]), niche("Game physics", [real])])
    good = NicheList(niches=[niche("Robot path-finding", [real]), niche("Sports data", [real]), niche("Game physics", [real])])
    fake = FakeLLM(bad, good)
    monkeypatch.setattr(niche_agent, "get_llm", lambda: fake)

    assert graph.run("suggest_niches", sid) == 3
    assert "don't exist: [999]" in fake.calls[1][-1].content  # the made-up id was sent back to be fixed
    assert "Aarav" not in fake.calls[0][1].content
    assert {n.niche_name for n in suggestions(session, sid)} == {"Robot path-finding", "Sports data", "Game physics"}
