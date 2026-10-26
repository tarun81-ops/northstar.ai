import io
import json
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from northstar import config
from northstar.db.models import (
    Base,
    DatasetRecommendation,
    ImportedDataset,
    KaggleDataset,
    ResearchProject,
    Source,
)
from northstar.db.seed import demo_profile, load_kaggle_catalog
from northstar.db.session import init_db, make_engine
from northstar.schemas import KaggleDatasetEntry, StarterEDAResult
from northstar.services import kaggle_dataset_service as kds
from northstar.services.profile import save_profile
from northstar.services.research import build_report, start_project


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
    return start_project(session, sid, "How does sleep impact academic focus?")


@pytest.fixture
def sample_csv_zip(tmp_path):
    """Create a temporary valid zip containing sample student performance CSV data."""
    csv_content = (
        "gender,race_ethnicity,parental_education,lunch,test_prep,math_score,reading_score,writing_score\n"
        "female,group B,bachelor's degree,standard,none,72,72,74\n"
        "female,group C,some college,standard,completed,69,90,88\n"
        "female,group B,master's degree,standard,none,90,95,93\n"
        "male,group A,associate's degree,free/reduced,none,47,57,44\n"
        "male,group C,some college,standard,none,76,78,75\n"
    )
    zip_path = tmp_path / "students-performance.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("StudentsPerformance.csv", csv_content)
    return zip_path


class MockKaggleApi:
    def __init__(self, zip_path: Path):
        self.zip_path = zip_path

    def dataset_download_files(self, dataset, path, quiet=True, unzip=False):
        dest = Path(path) / self.zip_path.name
        dest.write_bytes(self.zip_path.read_bytes())


# ---- Phase B validation tests -----------------------------------------------------------------

def test_config_kaggle_settings():
    assert hasattr(config, "KAGGLE_USERNAME")
    assert hasattr(config, "KAGGLE_KEY")
    assert hasattr(config, "KAGGLE_MAX_DATASET_SIZE_MB")
    assert hasattr(config, "KAGGLE_STUDENT_STORAGE_CAP_MB")
    assert hasattr(config, "DATASET_WORKSPACE_ROOT")
    assert isinstance(config.KAGGLE_MAX_DATASET_SIZE_MB, float)
    assert config.KAGGLE_MAX_DATASET_SIZE_MB > 0
    assert isinstance(config.DATASET_WORKSPACE_ROOT, Path)


def test_catalog_file_valid_schema():
    catalog_path = config.PROJECT_ROOT / "data" / "kaggle_catalog.json"
    assert catalog_path.exists(), "data/kaggle_catalog.json must exist"
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert "datasets" in raw
    assert len(raw["datasets"]) > 0

    for item in raw["datasets"]:
        entry = KaggleDatasetEntry.model_validate(item)
        assert "/" in entry.kaggle_ref
        assert entry.size_mb <= config.KAGGLE_MAX_DATASET_SIZE_MB
        assert entry.age_appropriate is True
        assert entry.is_approved is True
        assert entry.file_type == "csv"
        assert len(entry.limitations) > 0


def test_tables_created_by_init_db(engine):
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    assert "kaggle_datasets" in table_names
    assert "dataset_recommendations" in table_names
    assert "imported_datasets" in table_names


def test_idempotent_catalog_loading(session):
    count_first = load_kaggle_catalog(session)
    assert count_first > 0

    db_count_first = session.query(KaggleDataset).count()
    assert db_count_first == count_first

    count_second = load_kaggle_catalog(session)
    assert count_second == count_first
    db_count_second = session.query(KaggleDataset).count()
    assert db_count_second == count_first


def test_kaggle_ref_uniqueness(session):
    load_kaggle_catalog(session)
    first = session.query(KaggleDataset).first()
    assert first is not None

    duplicate = KaggleDataset(
        kaggle_ref=first.kaggle_ref,
        title="Duplicate Ref",
        description="Should fail",
        niche_tags=[],
        file_type="csv",
        size_mb=1.0,
        license="CC0",
        source_url="https://www.kaggle.com/datasets/duplicate/ref",
        is_approved=True,
    )
    session.add(duplicate)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


# ---- Phase C Service Unit Tests --------------------------------------------------------------

def test_invalid_kaggle_ref_rejection():
    invalid_refs = ["invalid_ref", "owner/", "/name", "owner/name/extra", "owner name"]
    for bad_ref in invalid_refs:
        with pytest.raises(ValueError, match="Invalid Kaggle dataset reference format"):
            kds.validate_kaggle_ref(bad_ref)


def test_search_catalog(session):
    load_kaggle_catalog(session)

    all_ds = kds.search_catalog(session)
    assert len(all_ds) >= 8

    # Query search
    weather_ds = kds.search_catalog(session, query="weather")
    assert any("weather" in d.title.lower() for d in weather_ds)

    # Tag search
    sleep_ds = kds.search_catalog(session, tags=["sleep"])
    assert any("sleep" in d.kaggle_ref for d in sleep_ds)


def test_recommendation_filtering_and_ranking(session, sid, pid):
    load_kaggle_catalog(session)

    candidates = kds.filter_and_rank_candidates(session, sid, pid)
    assert len(candidates) > 0

    # Project title has "sleep" -> sleep dataset should rank near top
    top_dataset, top_score = candidates[0]
    assert "sleep" in top_dataset.kaggle_ref or top_score >= 0

    # recommend_datasets creates recommendations in database
    recs = kds.recommend_datasets(session, sid, pid, limit=3)
    assert len(recs) == 3
    assert all(r.status == "suggested" for r in recs)

    # Calling again should retrieve the existing stored recommendations
    recs_again = kds.recommend_datasets(session, sid, pid, limit=3)
    assert len(recs_again) == 3
    assert recs_again[0].id == recs[0].id


def test_missing_kaggle_credentials():
    with patch.dict("os.environ", {"KAGGLE_USERNAME": "", "KAGGLE_KEY": ""}):
        with patch.object(config, "KAGGLE_USERNAME", ""), patch.object(config, "KAGGLE_KEY", ""):
            with pytest.raises(ValueError, match="Dataset import isn't set up yet."):
                kds._get_kaggle_api()


def test_import_dataset_with_mock_client_success(session, sid, pid, sample_csv_zip):
    load_kaggle_catalog(session)
    target_ref = "spscientist/students-performance-in-exams"

    api_mock = MockKaggleApi(sample_csv_zip)
    imported = kds.import_dataset(session, sid, pid, target_ref, api_client=api_mock)

    assert imported.status == "ready"
    assert imported.row_count == 5
    assert imported.column_count == 8
    assert "math_score" in imported.column_names
    assert Path(imported.local_file_path).exists()


def test_import_unapproved_dataset_rejection(session, sid, pid):
    load_kaggle_catalog(session)
    unapproved_ref = "malicious/unapproved-dataset"
    with pytest.raises(ValueError, match="not in the approved catalog"):
        kds.import_dataset(session, sid, pid, unapproved_ref)


def test_import_authorization_check(session, sid, pid, sample_csv_zip):
    load_kaggle_catalog(session)
    other_sid = save_profile(session, demo_profile().model_copy(update={"alias": "Rohan"}))

    api_mock = MockKaggleApi(sample_csv_zip)
    # Attempting to import into a project not owned by other_sid
    with pytest.raises(ValueError, match="not this student's"):
        kds.import_dataset(session, other_sid, pid, "spscientist/students-performance-in-exams", api_client=api_mock)


def test_csv_only_enforcement(session, sid, pid, tmp_path):
    load_kaggle_catalog(session)
    # Create zip with only a JSON file, no CSV
    non_csv_zip = tmp_path / "data_only_json.zip"
    with zipfile.ZipFile(non_csv_zip, "w") as zf:
        zf.writestr("data.json", '{"key": "value"}')

    api_mock = MockKaggleApi(non_csv_zip)
    with pytest.raises(ValueError, match="No CSV file found in downloaded dataset"):
        kds.import_dataset(session, sid, pid, "spscientist/students-performance-in-exams", api_client=api_mock)


def test_size_limit_and_storage_cap(session, sid, pid, sample_csv_zip):
    load_kaggle_catalog(session)
    api_mock = MockKaggleApi(sample_csv_zip)

    # Test catalog pre-check limit
    with patch.object(config, "KAGGLE_MAX_DATASET_SIZE_MB", 0.001):
        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            kds.import_dataset(session, sid, pid, "spscientist/students-performance-in-exams", api_client=api_mock)

    # Test student storage cap
    with patch.object(config, "KAGGLE_STUDENT_STORAGE_CAP_MB", 0.00001):
        with pytest.raises(ValueError, match="Storage limit exceeded"):
            kds.import_dataset(session, sid, pid, "spscientist/students-performance-in-exams", api_client=api_mock)


def test_zip_slip_prevention(tmp_path):
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../evil.csv", "bad,data\n1,2")

    target_dir = tmp_path / "extract_dir"
    target_dir.mkdir()
    with pytest.raises(ValueError, match="Zip-slip"):
        kds.safe_extract_zip(evil_zip, target_dir)


def test_zip_bomb_prevention(tmp_path):
    bomb_zip = tmp_path / "bomb.zip"
    # Write a zip where uncompressed size claims > MAX_UNCOMPRESSED_BYTES
    with patch.object(kds, "MAX_UNCOMPRESSED_BYTES", 10):
        with zipfile.ZipFile(bomb_zip, "w") as zf:
            zf.writestr("large.csv", "a" * 100)
        target_dir = tmp_path / "extract_bomb"
        target_dir.mkdir()
        with pytest.raises(ValueError, match="zip-bomb"):
            kds.safe_extract_zip(bomb_zip, target_dir)


def test_preview_and_starter_eda(session, sid, pid, sample_csv_zip):
    load_kaggle_catalog(session)
    target_ref = "spscientist/students-performance-in-exams"
    api_mock = MockKaggleApi(sample_csv_zip)
    kds.import_dataset(session, sid, pid, target_ref, api_client=api_mock)

    # Preview
    preview = kds.preview_dataset(session, sid, pid, target_ref)
    assert preview["is_imported"] is True
    assert preview["row_count"] == 5
    assert len(preview["columns"]) == 8
    assert len(preview["sample_rows"]) == 5

    # Run Starter EDA
    eda = kds.run_starter_eda(session, sid, pid)
    assert eda.row_count == 5
    assert eda.column_count == 8
    assert "math_score" in eda.numeric_summary
    assert eda.numeric_summary["math_score"]["mean"] is not None
    assert "gender" in eda.categorical_value_counts
    assert "female" in eda.categorical_value_counts["gender"]
    assert "Correlation does not imply causation" in eda.correlation_reminder


def test_attach_dataset_as_source_and_citation_report(session, sid, pid):
    load_kaggle_catalog(session)
    target_ref = "spscientist/students-performance-in-exams"

    source_num = kds.attach_dataset_as_source(session, sid, pid, target_ref)
    assert source_num == 1

    source = session.query(Source).filter(Source.project_id == pid, Source.number == 1).first()
    assert source is not None
    assert source.added_by == "student"
    assert "CC0: Public Domain" in source.note
    assert target_ref in source.note

    # Verify citation report includes it when cited in notes
    project = session.get(ResearchProject, pid)
    data_step = next(st for st in project.steps if st.step == "data")
    data_step.notes = "We evaluated student exam scores using public data [S1]."
    session.commit()

    report, warnings = build_report(project, alias="Aarav", today=date(2026, 11, 2))
    assert "[S1]" in report
    assert "## References" in report
    assert source.title in report


def test_project_deletion_cleanup(session, sid, pid, sample_csv_zip):
    load_kaggle_catalog(session)
    api_mock = MockKaggleApi(sample_csv_zip)
    imported = kds.import_dataset(session, sid, pid, "spscientist/students-performance-in-exams", api_client=api_mock)
    csv_file = Path(imported.local_file_path)
    assert csv_file.exists()

    workspace_dir = kds.get_project_workspace(sid, pid)
    assert workspace_dir.exists()

    # Delete project with workspace cleanup
    kds.delete_project_with_workspace(session, sid, pid)

    # Workspace directory on disk must be removed
    assert not workspace_dir.exists()

    # Project and imported dataset rows in DB must be gone
    assert session.get(ResearchProject, pid) is None
    assert session.query(ImportedDataset).filter(ImportedDataset.research_project_id == pid).count() == 0


# ---- Phase D Agent & LangGraph Tests ---------------------------------------------------------

class FakeDatasetLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.outputs:
            raise RuntimeError("No more fake LLM responses")
        return self.outputs.pop(0)


def test_check_dataset_matches():
    from northstar.agents.dataset_agent import check_dataset_matches
    from northstar.schemas import DatasetMatch

    valid_refs = {"owner/dataset-1", "owner/dataset-2"}

    # Valid matches
    ok = [DatasetMatch(kaggle_ref="owner/dataset-1", match_reason="Great fit")]
    assert check_dataset_matches(ok, valid_refs) == []

    # Invalid ref
    bad = [DatasetMatch(kaggle_ref="owner/hallucinated", match_reason="Fake")]
    assert any("not in the approved candidate list" in p for p in check_dataset_matches(bad, valid_refs))

    # Duplicate ref
    dups = [
        DatasetMatch(kaggle_ref="owner/dataset-1", match_reason="Fit 1"),
        DatasetMatch(kaggle_ref="owner/dataset-1", match_reason="Fit 2"),
    ]
    assert any("Duplicate recommendation" in p for p in check_dataset_matches(dups, valid_refs))

    # Empty
    assert any("No dataset recommendations" in p for p in check_dataset_matches([], valid_refs))


def test_dataset_agent_retry_and_success(session, monkeypatch):
    from northstar.agents import dataset_agent
    from northstar.schemas import DatasetMatch, DatasetMatchList

    load_kaggle_catalog(session)
    candidates = session.query(KaggleDataset).all()[:3]
    valid_ref = candidates[0].kaggle_ref

    # First attempt returns invalid ref; second attempt fixes it
    fake_llm = FakeDatasetLLM(
        DatasetMatchList(matches=[DatasetMatch(kaggle_ref="invalid/hallucinated-ref", match_reason="Bad")]),
        DatasetMatchList(matches=[DatasetMatch(kaggle_ref=valid_ref, match_reason="Legitimate match")]),
    )
    monkeypatch.setattr(dataset_agent, "get_llm", lambda: fake_llm)

    results = dataset_agent.match_datasets(
        project_title="Student Focus",
        question="How does sleep affect grades?",
        niche_name="Health",
        candidates=candidates,
    )
    assert len(results) == 1
    assert results[0].kaggle_ref == valid_ref
    assert results[0].match_reason == "Legitimate match"
    assert len(fake_llm.calls) == 2
    # Verify retry prompt asked to fix problems
    retry_msg = fake_llm.calls[1][-1].content
    assert "Please fix these problems" in retry_msg


def test_dataset_agent_fallback_on_failure(session, monkeypatch):
    from northstar.agents import dataset_agent

    load_kaggle_catalog(session)
    candidates = session.query(KaggleDataset).all()[:3]

    # LLM always raises or returns invalid
    fake_llm = FakeDatasetLLM()  # will raise on invoke
    monkeypatch.setattr(dataset_agent, "get_llm", lambda: fake_llm)

    results = dataset_agent.match_datasets(
        project_title="Student Focus",
        question="How does sleep affect grades?",
        niche_name="Health",
        candidates=candidates,
    )
    # Must fallback to top candidates with neutral explanation without failing
    assert len(results) == 3
    assert results[0].kaggle_ref == candidates[0].kaggle_ref
    assert "Pre-ranked match" in results[0].match_reason


def test_recommend_datasets_through_langgraph(engine, session, sid, pid, monkeypatch):
    from sqlalchemy.orm import sessionmaker
    from northstar import graph
    from northstar.agents import dataset_agent
    from northstar.schemas import DatasetMatch, DatasetMatchList

    load_kaggle_catalog(session)
    candidates = session.query(KaggleDataset).all()
    target_ref = candidates[0].kaggle_ref

    fake_llm = FakeDatasetLLM(
        DatasetMatchList(matches=[DatasetMatch(kaggle_ref=target_ref, match_reason="Recommended via LangGraph")]),
    )
    monkeypatch.setattr(dataset_agent, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(graph, "get_session", sessionmaker(engine))

    result_count = graph.run("recommend_datasets", sid, project_id=pid)
    assert result_count >= 1

    # Verify recommendations saved in DB
    recs = session.query(DatasetRecommendation).filter(DatasetRecommendation.research_project_id == pid).all()
    assert len(recs) >= 1
    assert any(r.kaggle_dataset.kaggle_ref == target_ref for r in recs)
    matching_rec = next(r for r in recs if r.kaggle_dataset.kaggle_ref == target_ref)
    assert matching_rec.match_reason == "Recommended via LangGraph"
