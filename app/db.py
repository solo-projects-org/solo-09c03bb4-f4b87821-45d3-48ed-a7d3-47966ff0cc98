"""SQLite persistence for catalog comparisons.

Everything that mutates results does so inside a single transaction: an
upload only becomes visible once the first computation succeeds, and a
decision batch is written together with the recomputed result or not at all.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple

from . import domain
from .matching import Candidate, DecisionConflict, Solution, solve_catalog

MUSEUMS = ("A", "B")
MUSEUM_LABELS = {"A": "甲馆", "B": "乙馆"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    a_filename TEXT NOT NULL,
    b_filename TEXT NOT NULL,
    a_rows INTEGER NOT NULL,
    b_rows INTEGER NOT NULL,
    selected_count INTEGER NOT NULL DEFAULT 0,
    gap_sum INTEGER NOT NULL DEFAULT 0,
    results_computed_at TEXT NOT NULL,
    last_error TEXT
);
CREATE TABLE IF NOT EXISTS uploads (
    comparison_id INTEGER NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
    museum TEXT NOT NULL CHECK (museum IN ('A','B')),
    filename TEXT NOT NULL,
    content BLOB NOT NULL,
    PRIMARY KEY (comparison_id, museum)
);
CREATE TABLE IF NOT EXISTS cat_rows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comparison_id INTEGER NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
    museum TEXT NOT NULL CHECK (museum IN ('A','B')),
    line_no INTEGER NOT NULL,
    collection_number TEXT NOT NULL,
    field_number TEXT NOT NULL,
    county_code TEXT NOT NULL,
    collected_on TEXT NOT NULL,
    UNIQUE (comparison_id, museum, collection_number)
);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comparison_id INTEGER NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
    a_row_id INTEGER NOT NULL REFERENCES cat_rows(id) ON DELETE CASCADE,
    b_row_id INTEGER NOT NULL REFERENCES cat_rows(id) ON DELETE CASCADE,
    gap_days INTEGER NOT NULL,
    UNIQUE (comparison_id, a_row_id, b_row_id)
);
CREATE TABLE IF NOT EXISTS decisions (
    candidate_id INTEGER PRIMARY KEY REFERENCES candidates(id) ON DELETE CASCADE,
    comparison_id INTEGER NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
    action TEXT NOT NULL CHECK (action IN ('confirm','exclude')),
    decided_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    candidate_id INTEGER PRIMARY KEY REFERENCES candidates(id) ON DELETE CASCADE,
    comparison_id INTEGER NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    selected INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rows_comparison ON cat_rows(comparison_id, museum);
CREATE INDEX IF NOT EXISTS idx_candidates_comparison ON candidates(comparison_id);
CREATE INDEX IF NOT EXISTS idx_results_comparison ON results(comparison_id);
CREATE INDEX IF NOT EXISTS idx_decisions_comparison ON decisions(comparison_id);
"""


class UploadError(ValueError):
    def __init__(self, message: str, details: Iterable[str] | None = None):
        super().__init__(message)
        self.details = list(details or [])


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _insert_rows(
    conn: sqlite3.Connection, comparison_id: int, museum: str, rows: list[dict]
) -> list[int]:
    ids: list[int] = []
    for row in rows:
        cur = conn.execute(
            """INSERT INTO cat_rows
               (comparison_id, museum, line_no, collection_number, field_number, county_code, collected_on)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                comparison_id,
                museum,
                row["line_no"],
                row["collection_number"],
                row["field_number"],
                row["county_code"],
                row["collected_on"],
            ),
        )
        ids.append(int(cur.lastrowid))
    return ids


def _build_candidates(
    a_rows: list[dict], b_rows: list[dict], a_ids: list[int], b_ids: list[int]
) -> list[Candidate]:
    index: dict[tuple[str, str], dict[str, list[tuple[dict, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row, row_id in zip(a_rows, a_ids):
        index[(row["field_number"], row["county_code"])][row["collected_on"]].append(
            (row, row_id)
        )
    candidates: list[Candidate] = []
    for b_row, b_id in zip(b_rows, b_ids):
        by_date = index.get((b_row["field_number"], b_row["county_code"]))
        if not by_date:
            continue
        b_date = dt.date.fromisoformat(b_row["collected_on"])
        for delta in (-1, 0, 1):
            date_key = (b_date + dt.timedelta(days=delta)).isoformat()
            for a_row, a_id in by_date.get(date_key, ()):
                gap = abs(delta)
                candidates.append(
                    Candidate(id=0, a_row_id=a_id, b_row_id=b_id, gap_days=gap)
                )
    candidates.sort(key=lambda c: (c.a_row_id, c.b_row_id))
    return candidates


def _persist_solution(
    conn: sqlite3.Connection, comparison_id: int, candidates: list[Candidate], solution: Solution
) -> None:
    conn.execute("DELETE FROM results WHERE comparison_id = ?", (comparison_id,))
    conn.executemany(
        "INSERT INTO results (candidate_id, comparison_id, status, selected) VALUES (?, ?, ?, ?)",
        [
            (
                candidate.id,
                comparison_id,
                solution.status[candidate.id],
                1 if solution.selected.get(candidate.id) else 0,
            )
            for candidate in candidates
        ],
    )
    conn.execute(
        """UPDATE comparisons
           SET selected_count = ?, gap_sum = ?, results_computed_at = ?, last_error = NULL
           WHERE id = ?""",
        (solution.selected_count, solution.gap_sum, _now(), comparison_id),
    )


def create_comparison(
    db_path: str | Path,
    a_filename: str,
    a_content: bytes,
    b_filename: str,
    b_content: bytes,
) -> int:
    """Parse both catalogues and store the comparison with its first result.

    Any parsing problem or computation failure leaves the database untouched.
    """
    parsed: dict[str, list[dict]] = {}
    for museum, filename, content in (
        ("A", a_filename, a_content),
        ("B", b_filename, b_content),
    ):
        try:
            rows, errors = domain.parse_csv(content)
        except domain.RowError as exc:
            raise UploadError(f"{MUSEUM_LABELS[museum]}文件无法导入：{exc}") from exc
        if errors:
            raise UploadError(
                f"{MUSEUM_LABELS[museum]}文件校验未通过，未导入任何数据",
                [f"{MUSEUM_LABELS[museum]} {message}" for message in errors[:50]],
            )
        parsed[museum] = rows

    conn = connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            """INSERT INTO comparisons
               (created_at, a_filename, b_filename, a_rows, b_rows, results_computed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                _now(),
                a_filename,
                b_filename,
                len(parsed["A"]),
                len(parsed["B"]),
                _now(),
            ),
        )
        comparison_id = int(cur.lastrowid)
        for museum, filename, content in (
            ("A", a_filename, a_content),
            ("B", b_filename, b_content),
        ):
            conn.execute(
                "INSERT INTO uploads (comparison_id, museum, filename, content) VALUES (?, ?, ?, ?)",
                (comparison_id, museum, filename, content),
            )
        a_ids = _insert_rows(conn, comparison_id, "A", parsed["A"])
        b_ids = _insert_rows(conn, comparison_id, "B", parsed["B"])
        candidates = _build_candidates(parsed["A"], parsed["B"], a_ids, b_ids)
        stored: list[Candidate] = []
        for candidate in candidates:
            row = conn.execute(
                "INSERT INTO candidates (comparison_id, a_row_id, b_row_id, gap_days) VALUES (?, ?, ?, ?)",
                (comparison_id, candidate.a_row_id, candidate.b_row_id, candidate.gap_days),
            )
            stored.append(
                Candidate(
                    id=int(row.lastrowid),
                    a_row_id=candidate.a_row_id,
                    b_row_id=candidate.b_row_id,
                    gap_days=candidate.gap_days,
                )
            )
        solution = solve_catalog(stored)
        _persist_solution(conn, comparison_id, stored, solution)
        conn.commit()
        return comparison_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_comparisons(db_path: str | Path) -> list[sqlite3.Row]:
    conn = connect(db_path)
    try:
        return conn.execute(
            "SELECT * FROM comparisons ORDER BY id DESC"
        ).fetchall()
    finally:
        conn.close()


def get_comparison(db_path: str | Path, comparison_id: int) -> sqlite3.Row | None:
    conn = connect(db_path)
    try:
        return conn.execute(
            "SELECT * FROM comparisons WHERE id = ?", (comparison_id,)
        ).fetchone()
    finally:
        conn.close()


STATUS_ORDER = {"certain": 0, "uncertain": 1, "unmatched": 2, "excluded": 3}

_FILTER_STATUSES = ("certain", "uncertain", "unmatched", "excluded")

_RESULT_PAGE_SQL = """
    SELECT c.id AS candidate_id, c.gap_days, r.status, r.selected,
           a.line_no AS a_line, a.collection_number AS a_collection,
           a.field_number, a.county_code, a.collected_on AS a_date,
           b.line_no AS b_line, b.collection_number AS b_collection,
           b.collected_on AS b_date,
           d.action AS decision
    FROM candidates c
    JOIN results r ON r.candidate_id = c.id
    JOIN cat_rows a ON a.id = c.a_row_id
    JOIN cat_rows b ON b.id = c.b_row_id
    LEFT JOIN decisions d ON d.candidate_id = c.id
    {where}
    ORDER BY CASE r.status
                WHEN 'certain' THEN 0 WHEN 'uncertain' THEN 1
                WHEN 'unmatched' THEN 2 ELSE 3 END,
             c.gap_days, a.line_no, b.line_no, c.id
"""


def _result_counts(conn: sqlite3.Connection, comparison_id: int) -> dict[str, int]:
    counts = {"all": 0, "certain": 0, "uncertain": 0, "unmatched": 0, "excluded": 0}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM results WHERE comparison_id = ? GROUP BY status",
        (comparison_id,),
    ):
        counts[row["status"]] = row["n"]
        counts["all"] += row["n"]
    return counts


def _result_where(comparison_id: int, status: str) -> tuple[str, list]:
    where = "WHERE c.comparison_id = ?"
    params: list = [comparison_id]
    if status in _FILTER_STATUSES:
        where += " AND r.status = ?"
        params.append(status)
    return where, params


def _result_total(conn: sqlite3.Connection, where: str, params: list) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM candidates c JOIN results r ON r.candidate_id = c.id {where}",
        params,
    ).fetchone()[0]


def _result_rows(
    conn: sqlite3.Connection, where: str, params: list, page: int, page_size: int
) -> list[sqlite3.Row]:
    offset = max(0, (page - 1) * page_size)
    return conn.execute(
        _RESULT_PAGE_SQL.format(where=where) + " LIMIT ? OFFSET ?",
        [*params, page_size, offset],
    ).fetchall()


def fetch_result_page(
    db_path: str | Path,
    comparison_id: int,
    status: str = "all",
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[sqlite3.Row], int, dict[str, int]]:
    """Read one page of results inside a single snapshot transaction."""
    conn = connect(db_path)
    try:
        conn.execute("BEGIN")
        counts = _result_counts(conn, comparison_id)
        where, params = _result_where(comparison_id, status)
        total = _result_total(conn, where, params)
        rows = _result_rows(conn, where, params, max(1, page), page_size)
        conn.commit()
        return rows, total, counts
    finally:
        conn.close()


class DetailPage(NamedTuple):
    """Everything the detail page renders, from one committed snapshot."""

    comparison: sqlite3.Row
    rows: list[sqlite3.Row]
    total: int
    counts: dict[str, int]
    page: int
    pages: int


def fetch_detail_page(
    db_path: str | Path,
    comparison_id: int,
    status: str = "all",
    page: int = 1,
    page_size: int = 50,
) -> DetailPage | None:
    """Load the whole detail page from a single committed computation result.

    The summary row, per-status counts, filtered total, page of candidates
    and the curator decisions shown on each row are read inside one deferred
    read transaction, so they always describe the same committed state even
    when another instance commits a decision concurrently.  In WAL mode the
    snapshot never blocks writers, and the transaction ends before this
    function returns.  Out-of-range page numbers are clamped against this
    same snapshot.  Returns ``None`` when the comparison does not exist.
    """
    conn = connect(db_path)
    try:
        conn.execute("BEGIN")
        comparison = conn.execute(
            "SELECT * FROM comparisons WHERE id = ?", (comparison_id,)
        ).fetchone()
        if comparison is None:
            conn.rollback()
            return None
        counts = _result_counts(conn, comparison_id)
        where, params = _result_where(comparison_id, status)
        total = _result_total(conn, where, params)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(max(1, page), pages)
        rows = _result_rows(conn, where, params, page, page_size)
        conn.commit()
        return DetailPage(comparison, rows, total, counts, page, pages)
    finally:
        conn.close()


def get_decisions(db_path: str | Path, comparison_id: int) -> dict[int, str]:
    conn = connect(db_path)
    try:
        return {
            row["candidate_id"]: row["action"]
            for row in conn.execute(
                "SELECT candidate_id, action FROM decisions WHERE comparison_id = ?",
                (comparison_id,),
            )
        }
    finally:
        conn.close()


def apply_decision(
    db_path: str | Path,
    comparison_id: int,
    candidate_id: int,
    action: str,
    next_path: str = "",
) -> None:
    """Apply one decision and recompute atomically.

    ``action`` is ``confirm``, ``exclude`` or ``clear`` (remove the decision).
    On any error the previous decisions and results are preserved.
    """
    del next_path  # navigation only; kept out of the transaction
    conn = connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id FROM candidates WHERE id = ? AND comparison_id = ?",
            (candidate_id, comparison_id),
        ).fetchone()
        if row is None:
            raise DecisionConflict("决定引用了不存在的候选配对")
        if action == "clear":
            conn.execute("DELETE FROM decisions WHERE candidate_id = ?", (candidate_id,))
        elif action in ("confirm", "exclude"):
            conn.execute(
                """INSERT INTO decisions (candidate_id, comparison_id, action, decided_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(candidate_id) DO UPDATE SET
                       action = excluded.action,
                       decided_at = excluded.decided_at""",
                (candidate_id, comparison_id, action, _now()),
            )
        else:
            raise DecisionConflict("未知的操作")

        decisions = {
            int(r["candidate_id"]): r["action"]
            for r in conn.execute(
                "SELECT candidate_id, action FROM decisions WHERE comparison_id = ?",
                (comparison_id,),
            )
        }
        candidates = [
            Candidate(
                id=int(r["id"]),
                a_row_id=int(r["a_row_id"]),
                b_row_id=int(r["b_row_id"]),
                gap_days=int(r["gap_days"]),
            )
            for r in conn.execute(
                "SELECT id, a_row_id, b_row_id, gap_days FROM candidates WHERE comparison_id = ? order by id",
                (comparison_id,),
            )
        ]
        all_ids = {c.id for c in candidates}
        excluded_ids = {cid for cid, action in decisions.items() if action == "exclude"}
        active = [c for c in candidates if c.id not in excluded_ids]
        solution = solve_catalog(active, decisions, all_candidate_ids=all_ids)
        _persist_solution(conn, comparison_id, candidates, solution)
        conn.commit()
    except Exception as exc:
        conn.rollback()
        conn.execute(
            "UPDATE comparisons SET last_error = ? WHERE id = ?",
            (str(exc), comparison_id),
        )
        conn.commit()
        raise
    finally:
        conn.close()


def recompute(db_path: str | Path, comparison_id: int) -> None:
    conn = connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute(
            "SELECT id FROM comparisons WHERE id = ?", (comparison_id,)
        ).fetchone() is None:
            raise DecisionConflict("对照任务不存在")
        candidates = [
            Candidate(
                id=int(r["id"]),
                a_row_id=int(r["a_row_id"]),
                b_row_id=int(r["b_row_id"]),
                gap_days=int(r["gap_days"]),
            )
            for r in conn.execute(
                "SELECT id, a_row_id, b_row_id, gap_days FROM candidates WHERE comparison_id = ? order by id",
                (comparison_id,),
            )
        ]
        decisions = {
            int(r["candidate_id"]): r["action"]
            for r in conn.execute(
                "SELECT candidate_id, action FROM decisions WHERE comparison_id = ?",
                (comparison_id,),
            )
        }
        excluded_ids = {cid for cid, action in decisions.items() if action == "exclude"}
        all_ids = {c.id for c in candidates}
        active = [c for c in candidates if c.id not in excluded_ids]
        solution = solve_catalog(active, decisions, all_candidate_ids=all_ids)
        _persist_solution(conn, comparison_id, candidates, solution)
        conn.commit()
    except Exception as exc:
        conn.rollback()
        conn.execute(
            "UPDATE comparisons SET last_error = ? WHERE id = ?",
            (str(exc), comparison_id),
        )
        conn.commit()
        raise
    finally:
        conn.close()


def confirmed_pairs(db_path: str | Path, comparison_id: int) -> list[sqlite3.Row]:
    conn = connect(db_path)
    try:
        return conn.execute(
            """
            SELECT a.collection_number AS a_collection, a.line_no AS a_line,
                   b.collection_number AS b_collection, b.line_no AS b_line,
                   a.field_number, a.county_code,
                   a.collected_on AS a_date, b.collected_on AS b_date,
                   c.gap_days, COALESCE(d.action, '') AS decision
            FROM candidates c
            JOIN results r ON r.candidate_id = c.id
            JOIN cat_rows a ON a.id = c.a_row_id
            JOIN cat_rows b ON b.id = c.b_row_id
            LEFT JOIN decisions d ON d.candidate_id = c.id
            WHERE c.comparison_id = ? AND r.selected = 1 AND r.status = 'certain'
            ORDER BY a.line_no, b.line_no
            """,
            (comparison_id,),
        ).fetchall()
    finally:
        conn.close()


def source_lines(
    db_path: str | Path, comparison_id: int, museum: str
) -> list[tuple[int, list[str]]]:
    """Return the uploaded CSV as (physical line number, cells) pairs."""
    import csv
    import io

    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT content FROM uploads WHERE comparison_id = ? AND museum = ?",
            (comparison_id, museum),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return []
    try:
        text = row["content"].decode("utf-8-sig")
    except UnicodeDecodeError:
        text = row["content"].decode("gb18030")
    reader = csv.reader(io.StringIO(text))
    records: list[tuple[int, list[str]]] = []
    previous_end = 0
    for record in reader:
        start_line = previous_end + 1
        previous_end = max(start_line, reader.line_num)
        records.append((start_line, record))
    return records

