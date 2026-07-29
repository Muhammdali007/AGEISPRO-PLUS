import os
import subprocess
import sys


def test_session_import_registers_complete_model_metadata() -> None:
    script = """
from sqlalchemy.orm import configure_mappers

from app.db.base import Base
from app.db.session import engine

configure_mappers()
required_tables = {"users", "cameras", "camera_secrets", "user_sessions"}
missing_tables = required_tables.difference(Base.metadata.tables)
assert not missing_tables, f"Missing model tables: {sorted(missing_tables)}"
"""
    environment = os.environ.copy()
    environment["ENVIRONMENT"] = "test"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
