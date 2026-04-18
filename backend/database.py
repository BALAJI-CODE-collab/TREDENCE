"""SQLite persistence layer for PruneVision upload history."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class UploadRecord:
    """Represents one persisted prediction upload."""

    id: int
    original_filename: str
    stored_filename: str
    label: str
    confidence: float
    confidence_status: str
    recommendation: str
    warning: Optional[str]
    quality_score: float
    leaf_detected: bool
    enhancements_applied: list[str]
    processing_time_ms: float
    created_at: str


def _connect(db_path: Path) -> sqlite3.Connection:
    """Create a SQLite connection with row mapping enabled."""

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_database(db_path: Path) -> None:
    """Create required database schema if it does not already exist."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                confidence_status TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                warning TEXT,
                quality_score REAL NOT NULL,
                leaf_detected INTEGER NOT NULL,
                enhancements_applied_json TEXT NOT NULL,
                processing_time_ms REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()
    LOGGER.info("Database initialized at %s", db_path)


def insert_upload_record(
    db_path: Path,
    *,
    original_filename: str,
    stored_filename: str,
    label: str,
    confidence: float,
    confidence_status: str,
    recommendation: str,
    warning: Optional[str],
    quality_score: float,
    leaf_detected: bool,
    enhancements_applied: list[str],
    processing_time_ms: float,
) -> int:
    """Insert one prediction record and return its generated identifier."""

    created_at = datetime.now(timezone.utc).isoformat()
    enhancements_json = json.dumps(enhancements_applied)

    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO uploads (
                original_filename,
                stored_filename,
                label,
                confidence,
                confidence_status,
                recommendation,
                warning,
                quality_score,
                leaf_detected,
                enhancements_applied_json,
                processing_time_ms,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                original_filename,
                stored_filename,
                label,
                confidence,
                confidence_status,
                recommendation,
                warning,
                quality_score,
                int(leaf_detected),
                enhancements_json,
                processing_time_ms,
                created_at,
            ),
        )
        connection.commit()
        upload_id = int(cursor.lastrowid)

    LOGGER.info("Stored upload record id=%d filename=%s", upload_id, original_filename)
    return upload_id


def _row_to_record(row: sqlite3.Row) -> UploadRecord:
    """Convert a SQLite row into an UploadRecord object."""

    enhancements = json.loads(row["enhancements_applied_json"])
    return UploadRecord(
        id=int(row["id"]),
        original_filename=str(row["original_filename"]),
        stored_filename=str(row["stored_filename"]),
        label=str(row["label"]),
        confidence=float(row["confidence"]),
        confidence_status=str(row["confidence_status"]),
        recommendation=str(row["recommendation"]),
        warning=row["warning"],
        quality_score=float(row["quality_score"]),
        leaf_detected=bool(row["leaf_detected"]),
        enhancements_applied=list(enhancements),
        processing_time_ms=float(row["processing_time_ms"]),
        created_at=str(row["created_at"]),
    )


def list_upload_records(db_path: Path, limit: int = 100) -> list[UploadRecord]:
    """List recent upload records ordered by newest first."""

    safe_limit = max(1, min(limit, 500))
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM uploads
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    return [_row_to_record(row) for row in rows]


def get_upload_record(db_path: Path, upload_id: int) -> Optional[UploadRecord]:
    """Fetch one upload record by identifier."""

    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM uploads
            WHERE id = ?
            """,
            (upload_id,),
        ).fetchone()

    if row is None:
        return None
    return _row_to_record(row)
