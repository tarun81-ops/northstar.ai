from datetime import date

import pytest
from sqlalchemy.orm import sessionmaker

from northstar.db.seed import demo_profile
from northstar.db.session import init_db, make_engine
from northstar.services.profile import get_profile, save_profile
from northstar.tools.profile_tools import make_profile_tools


@pytest.fixture
def session_factory():
    engine = make_engine("sqlite://")
    init_db(engine)
    return sessionmaker(engine)


def test_draft_until_required_fields_then_saved(session_factory):
    (get, missing, update), state = make_profile_tools(session_factory=session_factory)

    result = update.invoke({"alias": "Meera", "interests": ["Art"]})
    assert result["saved"] is False and state["student_id"] is None
    assert result["still_missing"]["required"] == ["age_band", "grade"]

    result = update.invoke({"age_band": "14-16", "grade": "Grade 9"})
    assert result["saved"] is True and state["student_id"] is not None
    with session_factory() as s:
        saved = get_profile(s, state["student_id"])
    assert (saved.alias, saved.interests) == ("Meera", ["Art"])  # the draft was carried over


def test_week_updates_merge_per_day(session_factory):
    with session_factory() as s:
        sid = save_profile(s, demo_profile())
    (get, missing, update), _ = make_profile_tools(sid, session_factory)

    update.invoke({"weekly_availability": {"sat": {"free_minutes": 60}}})
    week = get.invoke({})["profile"]["weekly_availability"]
    assert week["sat"]["free_minutes"] == 60
    assert week["mon"] == {"school": "08:00-15:00", "free_minutes": 120}  # untouched


def test_bad_value_is_rejected_and_nothing_changes(session_factory):
    with session_factory() as s:
        sid = save_profile(s, demo_profile())
    (get, missing, update), _ = make_profile_tools(sid, session_factory)

    result = update.invoke({"age_band": "adult", "interests": ["Chess"]})
    assert result["saved"] is False and result["errors"]
    assert get.invoke({})["profile"]["interests"] == demo_profile().interests


def test_missing_fields_for_complete_profile(session_factory):
    with session_factory() as s:
        sid = save_profile(s, demo_profile())
    (_, missing, _), _ = make_profile_tools(sid, session_factory)

    result = missing.invoke({})
    assert result == {"required": [], "not_yet_discussed": [], "today": date.today().isoformat()}


def test_llm_cannot_choose_the_student(session_factory):
    tools, _ = make_profile_tools(session_factory=session_factory)
    for t in tools:
        assert "student_id" not in t.args


def test_replayed_profile_update_does_not_repeat_database_write(session_factory, monkeypatch):
    with session_factory() as s:
        sid = save_profile(s, demo_profile())
    (_, _, update), _ = make_profile_tools(sid, session_factory)
    original_save = save_profile
    saves = []

    def counted_save(session, profile, student_id=None):
        saves.append(student_id)
        return original_save(session, profile, student_id)

    monkeypatch.setattr("northstar.tools.profile_tools.save_profile", counted_save)
    update.invoke({"interests": ["Art", "Astronomy"]})
    update.invoke({"interests": ["Art", "Astronomy"]})

    assert saves == [sid]
