import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd


_DATA_TABLE = "shared_dataset"
_METADATA_TABLE = "shared_dataset_metadata"
_PYTHON_FILES_TABLE = "shared_python_files"


def load_shared_dataset(database_path: Path):
    if not database_path.exists():
        return None, None

    with closing(sqlite3.connect(database_path)) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (_DATA_TABLE,),
        ).fetchone()
        if table_exists is None:
            return None, None

        dataframe = pd.read_sql_query(
            f'SELECT * FROM "{_DATA_TABLE}"',
            connection,
        )
        metadata_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (_METADATA_TABLE,),
        ).fetchone()
        saved_at = None
        if metadata_exists is not None:
            row = connection.execute(
                f'SELECT saved_at FROM "{_METADATA_TABLE}" WHERE id = 1'
            ).fetchone()
            if row is not None:
                saved_at = row[0]

    return dataframe, saved_at


def save_shared_dataset(dataframe: pd.DataFrame, database_path: Path):
    if len(dataframe.columns) == 0:
        raise ValueError("The shared dataset must have at least one column.")
    if dataframe.columns.has_duplicates:
        raise ValueError("The shared dataset cannot have duplicate column names.")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            dataframe.to_sql(_DATA_TABLE, connection, if_exists="replace", index=False)
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{_METADATA_TABLE}" (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    saved_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                f"""
                INSERT OR REPLACE INTO "{_METADATA_TABLE}" (id, saved_at)
                VALUES (1, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """
            )


def load_shared_python_files(database_path: Path):
    if not database_path.exists():
        return {}

    with closing(sqlite3.connect(database_path)) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (_PYTHON_FILES_TABLE,),
        ).fetchone()
        if table_exists is None:
            return {}

        rows = connection.execute(
            f'SELECT filename, code FROM "{_PYTHON_FILES_TABLE}" '
            "ORDER BY filename COLLATE NOCASE"
        ).fetchall()

    return dict(rows)


def save_shared_python_file(filename: str, code: str, database_path: Path):
    filename = filename.strip()
    if (
        not filename
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or Path(filename).suffix.casefold() != ".py"
    ):
        raise ValueError("Enter a valid .py filename without a directory path.")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{_PYTHON_FILES_TABLE}" (
                    filename TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    saved_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                f"""
                INSERT INTO "{_PYTHON_FILES_TABLE}" (filename, code, saved_at)
                VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                ON CONFLICT(filename) DO UPDATE SET
                    code = excluded.code,
                    saved_at = excluded.saved_at
                """,
                (filename, code),
            )

    return filename
