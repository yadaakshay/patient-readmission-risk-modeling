"""
Unit tests for the SQLite analytical database and clinical SQL query execution.
"""
import pytest
import sqlite3
import pandas as pd
from pathlib import Path
from src.database.init_db import DB_PATH
from src.database.queries import CLINICAL_QUERIES, execute_sql_query, execute_all_queries


def test_database_exists_and_has_tables():
    assert DB_PATH.exists(), f"Database not found at {DB_PATH}"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    assert "encounters" in tables
    assert "predictions" in tables
    assert "model_benchmark" in tables


@pytest.mark.parametrize("query_key", list(CLINICAL_QUERIES.keys()))
def test_all_clinical_queries_execute(query_key):
    df_res = execute_sql_query(query_key, DB_PATH)
    assert isinstance(df_res, pd.DataFrame)
    assert len(df_res) > 0, f"Query '{query_key}' returned empty result set"
