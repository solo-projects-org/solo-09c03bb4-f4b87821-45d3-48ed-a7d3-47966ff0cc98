"""Detail-page snapshot consistency, concurrency interleaving and pagination."""
from __future__ import annotations

import csv
import io
import re
import threading

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import PAGE_SIZE, app


def make_csv(rows) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["馆藏号", "采集号", "县代码", "采集日期"])
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


# A1-B1 and A2-B2 are unambiguous; A3 and A4 both compete for B3 (待核).
A_ROWS = [
    ("A1", "F1", "530100", "2024-01-01"),
    ("A2", "F2", "530200", "2024-01-05"),
    ("A3", "F3", "530300", "2024-02-01"),
    ("A4", "F3", "530300", "2024-02-01"),
]
B_ROWS = [
    ("B1", "F1", "530100", "2024-01-01"),
    ("B2", "F2", "530200", "2024-01-06"),
    ("B3", "F3", "530300", "2024-02-01"),
]


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "catalog.db")
    monkeypatch.setenv("CATALOG_DB", path)
    db.init_db(path)
    return path


@pytest.fixture()
def client(db_path):
    with TestClient(app) as test_client:
        yield test_client


def create(db_path, a_rows=A_ROWS, b_rows=B_ROWS) -> int:
    return db.create_comparison(db_path, "A.csv", make_csv(a_rows), "B.csv", make_csv(b_rows))


def upload(client, a_rows, b_rows):
    return client.post(
        "/comparisons",
        files={
            "a_file": ("A.csv", make_csv(a_rows), "text/csv"),
            "b_file": ("B.csv", make_csv(b_rows), "text/csv"),
        },
        follow_redirects=False,
    )


def candidate_ids(db_path, comparison_id) -> dict[tuple[int, int], int]:
    rows, _, _ = db.fetch_result_page(db_path, comparison_id, page_size=1000)
    return {(row["a_line"], row["b_line"]): row["candidate_id"] for row in rows}


def db_path_for(client) -> str:
    import os

    return os.environ["CATALOG_DB"]


def assert_snapshot_consistent(detail) -> None:
    """Every number on the page must describe the same committed result."""
    rows = detail.rows
    assert detail.total == len(rows)  # valid because page_size covers all rows
    assert detail.counts["all"] == len(rows)
    for status in ("certain", "uncertain", "unmatched", "excluded"):
        assert detail.counts[status] == sum(1 for r in rows if r["status"] == status)
    assert detail.comparison["selected_count"] == sum(r["selected"] for r in rows)
    assert detail.comparison["gap_sum"] == sum(
        r["gap_days"] for r in rows if r["selected"]
    )
    for row in rows:
        if row["decision"] == "confirm":
            assert row["status"] == "certain" and row["selected"] == 1
        elif row["decision"] == "exclude":
            assert row["status"] == "excluded" and row["selected"] == 0


def raw_counts(conn, comparison_id) -> dict[str, int]:
    return dict(
        conn.execute(
            "SELECT status, COUNT(*) FROM results WHERE comparison_id = ? GROUP BY status",
            (comparison_id,),
        ).fetchall()
    )


def test_open_snapshot_is_stable_and_does_not_block_writes(db_path):
    """A read transaction keeps one committed state; writers are not blocked."""
    comparison_id = create(db_path)
    target = candidate_ids(db_path, comparison_id)[(4, 4)]  # A3-B3, initially 待核

    reader = db.connect(db_path)
    try:
        reader.execute("BEGIN")
        summary_before = dict(
            reader.execute(
                "SELECT selected_count, gap_sum FROM comparisons WHERE id = ?",
                (comparison_id,),
            ).fetchone()
        )
        counts_before = raw_counts(reader, comparison_id)
        status_before = reader.execute(
            "SELECT status FROM results WHERE candidate_id = ?", (target,)
        ).fetchone()["status"]
        assert status_before == "uncertain"

        # Another instance commits a decision while this read is still open.
        db.apply_decision(db_path, comparison_id, target, "confirm")

        # The open snapshot still reports the previously committed result.
        assert (
            dict(
                reader.execute(
                    "SELECT selected_count, gap_sum FROM comparisons WHERE id = ?",
                    (comparison_id,),
                ).fetchone()
            )
            == summary_before
        )
        assert raw_counts(reader, comparison_id) == counts_before
        assert (
            reader.execute(
                "SELECT status FROM results WHERE candidate_id = ?", (target,)
            ).fetchone()["status"]
            == status_before
        )
        reader.rollback()
    finally:
        reader.close()

    # A fresh read sees the newly committed result.
    detail = db.fetch_detail_page(db_path, comparison_id, page_size=1000)
    assert detail.counts["certain"] == counts_before["certain"] + 1
    assert detail.counts["uncertain"] == 0
    assert_snapshot_consistent(detail)


def test_detail_snapshots_stay_consistent_under_concurrent_decisions(db_path):
    """Readers never see a mix of old and new results while another instance writes."""
    comparison_id = create(db_path)
    ids = candidate_ids(db_path, comparison_id)
    first = ids[(4, 4)]  # A3-B3
    second = ids[(5, 4)]  # A4-B3
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer():
        cycle = [
            (first, "confirm"),
            (second, "exclude"),
            (first, "clear"),
            (second, "clear"),
        ]
        try:
            for _ in range(50):
                for candidate_id, action in cycle:
                    db.apply_decision(db_path, comparison_id, candidate_id, action)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)
        finally:
            stop.set()

    def reader():
        rounds = 0
        while not stop.is_set() and rounds < 5000:
            detail = db.fetch_detail_page(db_path, comparison_id, page_size=1000)
            try:
                assert_snapshot_consistent(detail)
            except AssertionError as exc:
                errors.append(exc)
                stop.set()
                return
            rounds += 1

    thread = threading.Thread(target=writer)
    thread.start()
    reader()
    thread.join(timeout=120)
    stop.set()
    assert not errors


def test_pagination_partitions_candidates_and_clamps(db_path):
    a_rows = [(f"A{i:03d}", f"F{i}", "530100", "2024-01-01") for i in range(1, 121)]
    b_rows = [(f"B{i:03d}", f"F{i}", "530100", "2024-01-01") for i in range(1, 121)]
    comparison_id = create(db_path, a_rows, b_rows)

    seen: list[int] = []
    for page in (1, 2, 3):
        detail = db.fetch_detail_page(db_path, comparison_id, page=page, page_size=50)
        assert detail.page == page
        assert detail.pages == 3
        assert detail.total == 120
        seen.extend(row["candidate_id"] for row in detail.rows)
    assert len(seen) == len(set(seen)) == 120

    clamped = db.fetch_detail_page(db_path, comparison_id, page=99, page_size=50)
    assert clamped.page == 3
    assert clamped.pages == 3
    assert len(clamped.rows) == 20

    low = db.fetch_detail_page(db_path, comparison_id, page=0, page_size=50)
    assert low.page == 1


def paged_rows(n):
    a_rows = [(f"A{i:03d}", f"F{i}", "530100", "2024-01-01") for i in range(1, n + 1)]
    b_rows = [(f"B{i:03d}", f"F{i}", "530100", "2024-01-01") for i in range(1, n + 1)]
    return a_rows, b_rows


def stat_value(html: str, label: str) -> int:
    match = re.search(
        re.escape(label) + r'</span><span class="stat-value">(\d+)', html
    )
    assert match, f"stat {label!r} not found"
    return int(match.group(1))


def test_detail_page_paginates_and_clamps_out_of_range_pages(client):
    a_rows, b_rows = paged_rows(PAGE_SIZE + 10)
    response = upload(client, a_rows, b_rows)
    assert response.status_code == 303

    page1 = client.get("/comparisons/1")
    assert page1.status_code == 200
    assert "A001" in page1.text and "A060" not in page1.text
    assert "第 1 / 2 页" in page1.text
    assert stat_value(page1.text, "确定配对") == PAGE_SIZE + 10

    page2 = client.get("/comparisons/1?page=2")
    assert "A060" in page2.text and "A051" in page2.text
    assert "A001" not in page2.text
    assert "第 2 / 2 页" in page2.text
    # Summary and tab counts are identical on every page.
    assert stat_value(page2.text, "确定配对") == PAGE_SIZE + 10

    clamped = client.get("/comparisons/1?page=99")
    assert clamped.status_code == 200
    assert "第 2 / 2 页" in clamped.text
    assert "A060" in clamped.text and "A001" not in clamped.text

    low = client.get("/comparisons/1?page=0")
    assert "第 1 / 2 页" in low.text
    assert "A001" in low.text

    negative = client.get("/comparisons/1?page=-3")
    assert "第 1 / 2 页" in negative.text


def test_filtered_page_clamps_to_filtered_total(client):
    a_rows, b_rows = paged_rows(PAGE_SIZE + 10)
    upload(client, a_rows, b_rows)

    # No excluded candidates: any page beyond 1 falls back to page 1.
    page = client.get("/comparisons/1?status=excluded&page=5")
    assert page.status_code == 200
    assert "此分类下没有候选配对" in page.text
    assert stat_value(page.text, "确定配对") == PAGE_SIZE + 10

    # The certain filter still has two pages.
    page = client.get("/comparisons/1?status=certain&page=99")
    assert "第 2 / 2 页" in page.text
    assert "A060" in page.text


def test_detail_route_reads_exactly_one_snapshot(client, monkeypatch):
    upload(client, A_ROWS, B_ROWS)
    calls = []
    original = db.fetch_detail_page

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("详情页必须只读取一份快照")

    monkeypatch.setattr(db, "fetch_detail_page", spy)
    monkeypatch.setattr(db, "fetch_result_page", forbidden)
    monkeypatch.setattr(db, "get_comparison", forbidden)
    response = client.get("/comparisons/1?status=certain&page=99")
    assert response.status_code == 200
    assert len(calls) == 1


def test_decision_failure_keeps_previous_snapshot_and_message(client):
    upload(client, A_ROWS, B_ROWS)
    ids = candidate_ids(db_path_for(client), 1)
    first = ids[(4, 4)]
    second = ids[(5, 4)]
    client.post(
        "/comparisons/1/decide",
        data={"candidate_id": first, "action": "confirm"},
        follow_redirects=False,
    )
    before = db.fetch_detail_page(db_path_for(client), 1, page_size=1000)

    response = client.post(
        "/comparisons/1/decide",
        data={"candidate_id": second, "action": "confirm"},
        follow_redirects=False,
    )
    assert "error=" in response.headers["location"]
    after = db.fetch_detail_page(db_path_for(client), 1, page_size=1000)
    assert [(r["candidate_id"], r["status"], r["selected"], r["decision"]) for r in before.rows] == [
        (r["candidate_id"], r["status"], r["selected"], r["decision"]) for r in after.rows
    ]
    assert after.counts == before.counts
    assert after.comparison["selected_count"] == before.comparison["selected_count"]
    page = client.get(response.headers["location"])
    assert "矛盾决定" in page.text
