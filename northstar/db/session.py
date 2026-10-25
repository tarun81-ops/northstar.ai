"""One engine for the app. Use `with get_session() as s:` for each unit of work."""
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from northstar import config
from northstar.db.models import Base


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)
    if engine.dialect.name == "sqlite":
        # SQLite ignores foreign keys unless asked, per connection.
        event.listen(engine, "connect", lambda conn, _: conn.execute("PRAGMA foreign_keys=ON"))
    return engine


engine = make_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db(bind: Engine = engine) -> None:
    # ponytail: create_all only adds missing tables; switch to Alembic once the schema changes after real data exists.
    Base.metadata.create_all(bind)


def get_session() -> Session:
    return SessionLocal()
