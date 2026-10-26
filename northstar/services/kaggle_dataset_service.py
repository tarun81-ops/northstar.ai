"""Kaggle dataset service for student research studio.

All downloads and analyses run deterministically on approved datasets.
Rules:
- kaggle package is imported lazily.
- Kaggle credentials are read only from environment variables.
- References must exist in the curated, approved catalog.
- Only CSV files are parsed.
- Strict protection against zip-slip and zip-bomb attacks.
- Enforces max dataset size and per-student storage limits.
- Starter EDA is computed using deterministic pandas operations.
- Evidence attachment creates a normal Source row with added_by="student".
"""
from __future__ import annotations

import os
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from northstar import config
from northstar.db.models import (
    DatasetRecommendation,
    ImportedDataset,
    KaggleDataset,
    NicheSuggestion,
    ResearchProject,
    Source,
)
from northstar.schemas import (
    CORRELATION_REMINDER,
    ColumnSummary,
    DatasetMatch,
    StarterEDAResult,
)

KAGGLE_REF_PATTERN = re.compile(r"^[a-zA-Z0-9\-_.]+/[a-zA-Z0-9\-_.]+$")
MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024  # 100 MB max uncompressed
MAX_COMPRESSION_RATIO = 100  # zip bomb threshold


def _project(session: Session, student_id: int, project_id: int) -> ResearchProject:
    """Ownership check: ensure project exists and belongs to the given student."""
    project = session.get(ResearchProject, project_id)
    if project is None or project.student_id != student_id:
        raise ValueError(f"Project {project_id} is not this student's")
    return project


def validate_kaggle_ref(kaggle_ref: str) -> str:
    """Strictly validate Kaggle dataset reference format owner/dataset-name."""
    ref = (kaggle_ref or "").strip()
    if not KAGGLE_REF_PATTERN.match(ref):
        raise ValueError(
            f"Invalid Kaggle dataset reference format: '{kaggle_ref}'. Expected 'owner/dataset-name'."
        )
    return ref


def _get_approved_dataset(session: Session, kaggle_ref: str) -> KaggleDataset:
    """Retrieve an approved, age-appropriate dataset from the database catalog."""
    ref = validate_kaggle_ref(kaggle_ref)
    dataset = session.query(KaggleDataset).filter(
        KaggleDataset.kaggle_ref == ref,
        KaggleDataset.is_approved.is_(True),
        KaggleDataset.age_appropriate.is_(True),
    ).first()
    if not dataset:
        raise ValueError(f"Dataset '{kaggle_ref}' is not in the approved catalog.")
    return dataset


def get_project_workspace(student_id: int, project_id: int) -> Path:
    """Per-student, per-project workspace directory."""
    ws = config.DATASET_WORKSPACE_ROOT / str(student_id) / str(project_id)
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def get_student_storage_bytes(student_id: int) -> int:
    """Calculate total stored bytes for a student across all project workspaces."""
    student_dir = config.DATASET_WORKSPACE_ROOT / str(student_id)
    if not student_dir.exists():
        return 0
    total = 0
    for p in student_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _get_kaggle_api():
    """Lazily import and initialize the Kaggle API from environment variables only."""
    username = os.getenv("KAGGLE_USERNAME") or config.KAGGLE_USERNAME
    key = os.getenv("KAGGLE_KEY") or config.KAGGLE_KEY
    if not username or not key:
        raise ValueError("Dataset import isn't set up yet.")

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        raise ValueError(f"Dataset import isn't set up yet: {e}")


def safe_extract_zip(zip_path: Path, target_dir: Path) -> list[Path]:
    """Extract a ZIP archive safely with protection against zip-slip and zip-bomb attacks."""
    extracted_files: list[Path] = []
    target_dir_resolved = target_dir.resolve()
    total_uncompressed = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Pre-validation scan
        for member in zf.infolist():
            member_path = (target_dir / member.filename).resolve()
            try:
                member_path.relative_to(target_dir_resolved)
            except ValueError:
                raise ValueError(
                    f"Security error: Zip-slip path traversal detected in '{member.filename}'."
                )

            total_uncompressed += member.file_size
            if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "Security error: Archive exceeds maximum allowed uncompressed size (zip-bomb protection)."
                )

            if member.compress_size > 0:
                ratio = member.file_size / member.compress_size
                if ratio > MAX_COMPRESSION_RATIO:
                    raise ValueError(
                        f"Security error: Suspicious compression ratio ({ratio:.1f}x) detected (zip-bomb protection)."
                    )

        # Safe extraction
        for member in zf.infolist():
            member_path = (target_dir / member.filename).resolve()
            if member.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
            else:
                member_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(member_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_files.append(member_path)

    return extracted_files


# ---- Search and Recommendations -------------------------------------------------------------

def search_catalog(
    session: Session, query: str = "", tags: list[str] | None = None
) -> list[KaggleDataset]:
    """Search approved, age-appropriate datasets in the catalog by query or tags."""
    q = session.query(KaggleDataset).filter(
        KaggleDataset.is_approved.is_(True),
        KaggleDataset.age_appropriate.is_(True),
    )
    datasets = q.all()

    if not query and not tags:
        return datasets

    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower())) if query else set()
    tag_tokens = {t.lower() for t in tags} if tags else set()
    all_tokens = query_tokens | tag_tokens

    scored = []
    for d in datasets:
        d_tags = {t.lower() for t in d.niche_tags}
        d_text = f"{d.title} {d.description}".lower()
        score = sum(3 for t in all_tokens if t in d_tags)
        score += sum(1 for t in all_tokens if t in d_text)
        if score > 0:
            scored.append((-score, d))

    if scored:
        scored.sort(key=lambda x: x[0])
        return [d for _, d in scored]

    return datasets


def filter_and_rank_candidates(
    session: Session, student_id: int, project_id: int
) -> list[tuple[KaggleDataset, int]]:
    """Filter approved catalog entries within size limit and rank them by tag/question overlap."""
    project = _project(session, student_id, project_id)
    question_step = next((st for st in project.steps if st.step == "question"), None)
    question_text = (question_step.notes if question_step else "") or project.title

    tokens = set(re.findall(r"[a-z0-9]+", f"{project.title} {question_text}".lower()))
    if project.niche_id:
        niche = session.get(NicheSuggestion, project.niche_id)
        if niche:
            tokens |= set(re.findall(r"[a-z0-9]+", f"{niche.niche_name} {niche.description}".lower()))

    approved = session.query(KaggleDataset).filter(
        KaggleDataset.is_approved.is_(True),
        KaggleDataset.age_appropriate.is_(True),
        KaggleDataset.size_mb <= config.KAGGLE_MAX_DATASET_SIZE_MB,
    ).all()

    scored = []
    for d in approved:
        d_tags = {t.lower() for t in d.niche_tags}
        d_text = f"{d.title} {d.description}".lower()
        score = sum(3 for t in tokens if t in d_tags)
        score += sum(1 for t in tokens if t in d_text)
        scored.append((d, score))

    scored.sort(key=lambda item: (item[1], item[0].usability_score or 0.0), reverse=True)
    return scored


def recommend_datasets(
    session: Session, student_id: int, project_id: int, limit: int = 5
) -> list[DatasetRecommendation]:
    """Retrieve existing recommendations or compute and store new suggestions."""
    _project(session, student_id, project_id)
    existing = session.query(DatasetRecommendation).filter(
        DatasetRecommendation.research_project_id == project_id,
        DatasetRecommendation.student_id == student_id,
    ).all()
    if existing:
        return existing

    ranked = filter_and_rank_candidates(session, student_id, project_id)
    candidates = ranked[:limit] if ranked else []

    recs = []
    for d, _ in candidates:
        reason = (
            f"Matches research question with relevant tags ({', '.join(d.niche_tags[:3])}). "
            f"Provides structured tabular evidence for student investigation."
        )
        rec = DatasetRecommendation(
            student_id=student_id,
            research_project_id=project_id,
            kaggle_dataset_id=d.id,
            match_reason=reason,
            status="suggested",
        )
        session.add(rec)
        recs.append(rec)

    session.commit()
    return recs


def save_dataset_recommendations(
    session: Session, student_id: int, project_id: int, matches: list[DatasetMatch]
) -> list[DatasetRecommendation]:
    """Save or update dataset recommendations for a project based on matched refs."""
    _project(session, student_id, project_id)
    saved_recs = []
    for m in matches:
        dataset = session.query(KaggleDataset).filter(KaggleDataset.kaggle_ref == m.kaggle_ref).first()
        if not dataset:
            continue
        rec = session.query(DatasetRecommendation).filter(
            DatasetRecommendation.research_project_id == project_id,
            DatasetRecommendation.kaggle_dataset_id == dataset.id,
        ).first()
        if not rec:
            rec = DatasetRecommendation(
                student_id=student_id,
                research_project_id=project_id,
                kaggle_dataset_id=dataset.id,
                match_reason=m.match_reason,
                status="suggested",
            )
            session.add(rec)
        else:
            rec.match_reason = m.match_reason
            rec.status = "suggested"
        saved_recs.append(rec)
    session.commit()
    return saved_recs


# ---- Preview and Import --------------------------------------------------------------------

def preview_dataset(
    session: Session, student_id: int, project_id: int, kaggle_ref: str, num_rows: int = 5
) -> dict[str, Any]:
    """Preview metadata and sample rows if imported, or catalog details if not yet imported."""
    _project(session, student_id, project_id)
    dataset = _get_approved_dataset(session, kaggle_ref)

    imported = session.query(ImportedDataset).filter(
        ImportedDataset.research_project_id == project_id,
        ImportedDataset.kaggle_dataset_id == dataset.id,
        ImportedDataset.status == "ready",
    ).first()

    preview_data: dict[str, Any] = {
        "kaggle_ref": dataset.kaggle_ref,
        "title": dataset.title,
        "description": dataset.description,
        "size_mb": dataset.size_mb,
        "license": dataset.license,
        "attribution_required": dataset.attribution_required,
        "attribution_text": dataset.attribution_text,
        "difficulty": dataset.difficulty,
        "sample_questions": dataset.sample_questions,
        "limitations": dataset.limitations,
        "source_url": dataset.source_url,
        "is_imported": imported is not None,
        "columns": [],
        "dtypes": {},
        "sample_rows": [],
        "row_count": None,
    }

    if imported and imported.local_file_path:
        csv_file = Path(imported.local_file_path)
        if csv_file.exists():
            import pandas as pd

            df = pd.read_csv(csv_file, nrows=num_rows)
            preview_data["columns"] = list(df.columns)
            preview_data["dtypes"] = {col: str(df[col].dtype) for col in df.columns}
            preview_data["sample_rows"] = df.fillna("").to_dict(orient="records")
            preview_data["row_count"] = imported.row_count

    return preview_data


def import_dataset(
    session: Session,
    student_id: int,
    project_id: int,
    kaggle_ref: str,
    api_client: Any = None,
) -> ImportedDataset:
    """Download and prepare an approved dataset for a student project workspace.

    Enforces catalog verification, file size limits, storage caps, zip-slip/zip-bomb checks,
    and CSV-only parsing.
    """
    _project(session, student_id, project_id)
    dataset = _get_approved_dataset(session, kaggle_ref)

    # Pre-download size check from catalog
    if dataset.size_mb > config.KAGGLE_MAX_DATASET_SIZE_MB:
        raise ValueError(
            f"Dataset catalog size ({dataset.size_mb} MB) exceeds maximum allowed limit of {config.KAGGLE_MAX_DATASET_SIZE_MB} MB."
        )

    # Per-student storage cap check
    max_storage_bytes = int(config.KAGGLE_STUDENT_STORAGE_CAP_MB * 1024 * 1024)
    current_storage = get_student_storage_bytes(student_id)
    estimated_bytes = int(dataset.size_mb * 1024 * 1024)
    if current_storage + estimated_bytes > max_storage_bytes:
        raise ValueError(
            f"Storage limit exceeded: Student storage cap of {config.KAGGLE_STUDENT_STORAGE_CAP_MB} MB would be exceeded."
        )

    workspace = get_project_workspace(student_id, project_id)
    api = api_client or _get_kaggle_api()

    # Create or retrieve ImportedDataset row in "importing" state
    imported = session.query(ImportedDataset).filter(
        ImportedDataset.research_project_id == project_id,
        ImportedDataset.kaggle_dataset_id == dataset.id,
    ).first()

    if not imported:
        imported = ImportedDataset(
            student_id=student_id,
            research_project_id=project_id,
            kaggle_dataset_id=dataset.id,
            status="importing",
        )
        session.add(imported)
    else:
        imported.status = "importing"
        imported.error_message = None
    session.commit()

    try:
        # Download files into workspace
        download_dir = workspace / "raw"
        download_dir.mkdir(parents=True, exist_ok=True)

        api.dataset_download_files(dataset.kaggle_ref, path=str(download_dir), quiet=True, unzip=False)

        # Inspect downloaded files
        downloaded_items = list(download_dir.glob("*"))
        if not downloaded_items:
            raise ValueError("Kaggle download produced no files.")

        csv_files: list[Path] = []
        for item in downloaded_items:
            if item.is_file() and item.suffix.lower() == ".zip":
                extracted = safe_extract_zip(item, download_dir)
                item.unlink(missing_ok=True)
                csv_files.extend([f for f in extracted if f.suffix.lower() == ".csv"])
            elif item.is_file() and item.suffix.lower() == ".csv":
                csv_files.append(item)

        if not csv_files:
            raise ValueError("No CSV file found in downloaded dataset. Only CSV datasets are supported.")

        # Pick the largest CSV if multiple exist
        primary_csv = max(csv_files, key=lambda f: f.stat().st_size)

        # Move to project workspace root
        final_csv = workspace / primary_csv.name
        if final_csv.resolve() != primary_csv.resolve():
            shutil.move(str(primary_csv), str(final_csv))

        # Cleanup raw directory
        shutil.rmtree(download_dir, ignore_errors=True)

        # Post-download actual size check
        actual_size = final_csv.stat().st_size
        max_bytes = int(config.KAGGLE_MAX_DATASET_SIZE_MB * 1024 * 1024)
        if actual_size > max_bytes:
            final_csv.unlink(missing_ok=True)
            raise ValueError(
                f"Actual downloaded dataset size ({actual_size / (1024 * 1024):.2f} MB) exceeds limit of {config.KAGGLE_MAX_DATASET_SIZE_MB} MB."
            )

        # Inspect CSV with pandas
        import pandas as pd

        df = pd.read_csv(final_csv)
        row_count, col_count = df.shape
        dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

        imported.local_file_path = str(final_csv)
        imported.original_file_name = final_csv.name
        imported.row_count = row_count
        imported.column_count = col_count
        imported.column_names = list(df.columns)
        imported.dtypes = dtypes_dict
        imported.size_bytes = actual_size
        imported.status = "ready"
        imported.error_message = None
        imported.imported_at = datetime.now()
        session.commit()
        return imported

    except Exception as e:
        imported.status = "failed"
        imported.error_message = str(e)
        session.commit()
        raise


# ---- Starter EDA ---------------------------------------------------------------------------

def run_starter_eda(
    session: Session, student_id: int, project_id: int, dataset_id: int | None = None
) -> StarterEDAResult:
    """Run deterministic pandas EDA on the imported CSV dataset."""
    _project(session, student_id, project_id)

    query = session.query(ImportedDataset).filter(
        ImportedDataset.student_id == student_id,
        ImportedDataset.research_project_id == project_id,
        ImportedDataset.status == "ready",
    )
    if dataset_id:
        query = query.filter(ImportedDataset.id == dataset_id)
    imported = query.order_by(ImportedDataset.id.desc()).first()

    if not imported or not imported.local_file_path:
        raise ValueError("No imported dataset is ready for this project.")

    csv_path = Path(imported.local_file_path)
    if not csv_path.exists():
        raise ValueError(f"Dataset file missing at {csv_path}.")

    import pandas as pd

    df = pd.read_csv(csv_path)

    row_count, col_count = df.shape
    columns = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in df.columns}

    # df.head()
    head_df = df.head(5)
    head_records = head_df.fillna("").to_dict(orient="records")

    # df.isnull().sum()
    null_counts = {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}

    # df.describe() for numeric columns
    numeric_df = df.select_dtypes(include=["number"])
    numeric_summary: dict[str, dict[str, float | None]] = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().to_dict()
        for col, stats in desc.items():
            numeric_summary[str(col)] = {
                str(stat): (float(val) if pd.notnull(val) else None)
                for stat, val in stats.items()
            }

    # Categorical low-cardinality value counts
    categorical_value_counts: dict[str, dict[str, int]] = {}
    cat_df = df.select_dtypes(include=["object", "string", "category", "bool"])
    for col in cat_df.columns:
        n_unique = df[col].nunique(dropna=True)
        if 1 <= n_unique <= 20:
            top_counts = df[col].value_counts(dropna=False).head(10).to_dict()
            categorical_value_counts[str(col)] = {str(k): int(v) for k, v in top_counts.items()}

    # Column summaries
    column_summaries = []
    for col in columns:
        unique_samples = [str(x) for x in df[col].dropna().unique()[:3]]
        column_summaries.append(
            ColumnSummary(
                name=str(col),
                dtype=dtypes[col],
                non_null_count=int(df[col].notnull().sum()),
                null_count=int(null_counts.get(col, 0)),
                unique_count=int(df[col].nunique(dropna=True)),
                sample_values=unique_samples,
            )
        )

    return StarterEDAResult(
        row_count=row_count,
        column_count=col_count,
        columns=columns,
        dtypes=dtypes,
        head_records=head_records,
        missing_values=null_counts,
        numeric_summary=numeric_summary,
        categorical_value_counts=categorical_value_counts,
        column_summaries=column_summaries,
        correlation_reminder=CORRELATION_REMINDER,
        explanation=None,
    )


# ---- Evidence Attachment & Cleanup ---------------------------------------------------------

def attach_dataset_as_source(
    session: Session, student_id: int, project_id: int, kaggle_ref: str
) -> int:
    """Attach an approved Kaggle dataset as a student Source row [S#].

    Stores Kaggle URL, license, and attribution information in the note.
    """
    _project(session, student_id, project_id)
    dataset = _get_approved_dataset(session, kaggle_ref)

    note_lines = [
        f"Kaggle Dataset: {dataset.title} ({dataset.kaggle_ref})",
        f"License: {dataset.license}",
    ]
    if dataset.attribution_text:
        note_lines.append(f"Attribution: {dataset.attribution_text}")
    note_lines.append(f"Source URL: {dataset.source_url}")
    note_lines.append(
        "Suggestion only: Correlation observed in this dataset does not imply causation."
    )

    full_note = "\n".join(note_lines)

    next_number = (
        session.scalar(select(func.max(Source.number)).where(Source.project_id == project_id))
        or 0
    ) + 1

    source = Source(
        project_id=project_id,
        number=next_number,
        title=f"{dataset.title} (Kaggle Dataset)",
        url=dataset.source_url,
        publisher="Kaggle",
        note=full_note,
        added_by="student",
        opened=True,
    )
    session.add(source)
    session.commit()
    return next_number


def cleanup_project_workspace(student_id: int, project_id: int) -> bool:
    """Delete a project's dataset workspace directory on disk."""
    ws = config.DATASET_WORKSPACE_ROOT / str(student_id) / str(project_id)
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
        return True
    return False


def delete_project_with_workspace(session: Session, student_id: int, project_id: int) -> None:
    """Delete a research project, all cascading database records, and disk workspace files."""
    project = _project(session, student_id, project_id)
    session.delete(project)
    session.commit()
    cleanup_project_workspace(student_id, project_id)
