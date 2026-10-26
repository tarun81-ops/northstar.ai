from datetime import date, timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker

from northstar import graph
from northstar.agents import opportunity_agent
from northstar.db.seed import demo_profile
from northstar.db.session import init_db, make_engine
from northstar.schemas import Opportunity, OpportunityMatch, OpportunityMatchList
from northstar.services.opportunities import (
    check_matches, decide, eligible, load_opportunities, permission_note, rank, recommendations, save_matches,
)
from northstar.services.profile import save_profile

TODAY = date(2026, 11, 2)


def opp(key: str, age=(8, 18), tags=("robotics",), deadline=None, free=True, permission=False) -> Opportunity:
    return Opportunity(key=key, title=key.title(), organizer="Org", type="competition", summary="s", tags=list(tags),
                       age_min=age[0], age_max=age[1], eligibility="e", region="Worldwide", format="online",
                       cost="Free" if free else "Fee", is_free=free, time_commitment="t",
                       requires_permission=permission, deadline=deadline, url="https://example.org/")


# ---- the curated file -----------------------------------------------------------------------

def test_curated_file_is_valid():
    opportunities = load_opportunities()
    assert len(opportunities) >= 10
    assert len({o.key for o in opportunities}) == len(opportunities)
    assert all(o.age_min <= o.age_max and str(o.url).startswith("https://") for o in opportunities)


# ---- eligibility: expired or age-ineligible never appear ------------------------------------

def test_age_and_deadline_and_cost_filters():
    younger = demo_profile().model_copy(update={"age_band": "11-13"})
    opps = [opp("open"), opp("too-old", age=(14, 19)), opp("expired", deadline=TODAY - timedelta(days=1)),
            opp("due-today", deadline=TODAY), opp("paid", free=False)]
    assert [o.key for o in eligible(opps, younger, TODAY)] == ["open", "due-today", "paid"]
    assert [o.key for o in eligible(opps, younger, TODAY, only_free=True)] == ["open", "due-today"]
    college = demo_profile().model_copy(update={"age_band": "college"})
    assert [o.key for o in eligible([opp("kids-only", age=(8, 12)), opp("adults", age=(18, 99))], college, TODAY)] == ["adults"]


def test_rank_by_interest_overlap():
    opps = [opp("robots", tags=("robotics", "coding")), opp("maths", tags=("maths",)), opp("art", tags=("painting",))]
    assert [o.key for o in rank(opps, ["Robotics", "Coding", "Maths"])] == ["robots", "maths"]  # no overlap = left out
    assert [o.key for o in rank([opp("r", tags=("robots",))], ["robot"])] == ["r"]  # plural trimmed


def test_check_matches():
    ok = [OpportunityMatch(key="a", why="Fits your robotics interest.")]
    assert check_matches(ok, {"a", "b"}) == []
    assert any("only use these keys" in p for p in check_matches([OpportunityMatch(key="zzz", why="x")], {"a"}))
    assert any("only once" in p for p in check_matches(ok * 2, {"a"}))
    assert any("no links" in p for p in check_matches([OpportunityMatch(key="a", why="see www.x.com")], {"a"}))
    assert any("1-5" in p for p in check_matches([], {"a"}))


def test_permission_note():
    assert "guardian" in permission_note(opp("x", permission=True), demo_profile())
    assert "parent or teacher" in permission_note(opp("x"), demo_profile())
    assert permission_note(opp("x"), demo_profile().model_copy(update={"age_band": "college"})) is None


# ---- recommendations --------------------------------------------------------------------------

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


def test_saved_stay_and_dismissed_never_come_back(session, sid):
    save_matches(session, sid, [OpportunityMatch(key=k, why="w") for k in ("a", "b", "c")])
    a, b, c = recommendations(session, sid, "suggested")
    decide(session, sid, a.id, "saved")
    decide(session, sid, b.id, "dismissed")

    save_matches(session, sid, [OpportunityMatch(key=k, why="new") for k in ("b", "d")])
    assert [r.opportunity_key for r in recommendations(session, sid, "suggested")] == ["d"]  # c replaced, b not revived
    assert [r.opportunity_key for r in recommendations(session, sid, "saved")] == ["a"]
    assert [r.opportunity_key for r in recommendations(session, sid, "dismissed")] == ["b"]

    other = save_profile(session, demo_profile().model_copy(update={"alias": "Meera"}))
    with pytest.raises(ValueError, match="not this student's"):
        decide(session, other, a.id, "dismissed")


# ---- through the graph (fake LLM) ----------------------------------------------------------------

class FakeLLM:
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.outputs.pop(0)


def test_match_through_the_graph(engine, session, sid, monkeypatch):
    monkeypatch.setattr(graph, "get_session", sessionmaker(engine))
    monkeypatch.setattr(graph, "load_opportunities", lambda: [
        opp("robot-league", tags=("robotics", "engineering")), opp("code-course", tags=("coding",)),
        opp("too-old", age=(18, 99), tags=("robotics",)), opp("closed", tags=("robotics",), deadline=TODAY - timedelta(days=3)),
        opp("art-prize", tags=("painting",)),
    ])
    save_matches(session, sid, [OpportunityMatch(key="code-course", why="old")])
    decide(session, sid, recommendations(session, sid, "suggested")[0].id, "dismissed")

    fake = FakeLLM(OpportunityMatchList(matches=[OpportunityMatch(key="too-old", why="x")]),
                   OpportunityMatchList(matches=[OpportunityMatch(key="robot-league", why="You love building robots.")]))
    monkeypatch.setattr(opportunity_agent, "get_llm", lambda: fake)

    assert graph.run("match_opportunities", sid, today=TODAY) == 1
    sent = fake.calls[0][1].content
    assert "robot-league" in sent  # eligible and matching
    assert not any(k in sent for k in ("too-old", "closed", "art-prize", "code-course"))  # ineligible, no overlap, dismissed
    assert "deadline" not in sent and "cost" not in sent and "Aarav" not in sent
    assert "only use these keys" in fake.calls[1][-1].content  # the ineligible pick was sent back
    session.expire_all()
    assert [r.reason for r in recommendations(session, sid, "suggested")] == ["You love building robots."]


def test_no_matching_candidates_means_no_ai_call(engine, sid, monkeypatch):
    monkeypatch.setattr(graph, "get_session", sessionmaker(engine))
    monkeypatch.setattr(graph, "load_opportunities", lambda: [opp("art-prize", tags=("painting",))])
    monkeypatch.setattr(opportunity_agent, "get_llm", lambda: FakeLLM())  # would fail if called
    assert graph.run("match_opportunities", sid, today=TODAY) == 0
