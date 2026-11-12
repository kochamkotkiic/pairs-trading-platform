"""Checks that the PostgreSQL container from docker-compose.yml is reachable.

Skipped automatically when the database libraries are not installed or the database
is not running, so it never breaks CI. Run locally after `docker compose up -d`:

    pytest tests/integration/test_db_connection.py -v
"""

import os

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")

DEFAULT_URL = "postgresql+psycopg://ptp:change-me@localhost:5432/ptp"


@pytest.mark.integration
def test_can_connect_to_postgres() -> None:
    url = os.environ.get("DATABASE_URL", DEFAULT_URL)
    engine = sqlalchemy.create_engine(url, connect_args={"connect_timeout": 3})
    try:
        with engine.connect() as conn:
            version = conn.execute(sqlalchemy.text("SELECT version()")).scalar_one()
    except sqlalchemy.exc.OperationalError as exc:
        pytest.skip(
            f"PostgreSQL not reachable ({exc.__class__.__name__}); run `docker compose up -d`"
        )
    finally:
        engine.dispose()
    assert "PostgreSQL 16" in version
