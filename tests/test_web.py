from __future__ import annotations

import csv
import io

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app
from app.matching import RecomputeError


def make_csv(rows) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["馆藏号", "采集号", "县代码", "采集日期"])
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


# A1-B1 and A2-B2 are unambiguous (one option per specimen);
# A3 and A4 both compete for B3, so those two pairs are "待核".
BASE_A = [
    ("A1", "F1", "530100", "2024-01-01"),
    ("A2", "F2", "530200", "2024-01-05"),
    ("A3", "F3", "530300", "2024-02-01"),
    ("A4", "F3", "530300", "2024-02-01"),
]
BASE_B = [
    ("B1", "F1", "530100", "2024-01-01"),
    ("B2", "F2", "530200", "2024-01-06"),
    ("B3", "F3", "530300", "2024-02-01"),
]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CATALOG_DB", str(tmp_path / "catalog.db"))
    with TestClient(app) as test_client:
        yield test_client


def upload(client, a_rows, b_rows, a_name="A.csv", b_name="B.csv"):
    return client.post(
        "/comparisons",
        files={
            "a_file": (a_name, make_csv(a_rows), "text/csv"),
            "b_file": (b_name, make_csv(b_rows), "text/csv"),
        },
        follow_redirects=False,
    )


def candidate_id(client, comparison_id, a_line, b_line):
    rows, _, _ = db.fetch_result_page(
        db_path(client), comparison_id, page_size=1000
    )
    for row in rows:
        if row["a_line"] == a_line and row["b_line"] == b_line:
            return row["candidate_id"]
    raise AssertionError("candidate not found")


def db_path(client):
    import os

    return os.environ["CATALOG_DB"]


def test_upload_creates_comparison_and_shows_results(client):
    response = upload(client, BASE_A, BASE_B)
    assert response.status_code == 303

    page = client.get("/comparisons/1")
    assert page.status_code == 200
    assert "确定" in page.text and "待核" in page.text
    assert "A1" in page.text and "B1" in page.text

    index = client.get("/")
    assert 'href="/comparisons/1"' in index.text


def test_upload_rejects_duplicate_collection_numbers_without_writing(client):
    bad = BASE_A + [("A1", "F9", "530900", "2024-02-02")]
    response = upload(client, bad, BASE_B)
    assert response.status_code == 400
    assert "重复" in response.text
    assert db.list_comparisons(db_path(client)) == []


def test_upload_rejects_missing_columns(client):
    raw = "馆藏号,采集号\nA1,F1\n".encode()
    response = client.post(
        "/comparisons",
        files={
            "a_file": ("A.csv", raw, "text/csv"),
            "b_file": ("B.csv", make_csv(BASE_B), "text/csv"),
        },
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert db.list_comparisons(db_path(client)) == []


def test_decisions_persist_and_recompute(client):
    upload(client, BASE_A, BASE_B)
    cid = candidate_id(client, 1, 4, 4)  # A3-B3
    response = client.post(
        "/comparisons/1/decide",
        data={"candidate_id": cid, "action": "confirm", "status": "all", "page": "1"},
        follow_redirects=False,
    )
    assert response.status_code == 303

    assert db.get_decisions(db_path(client), 1) == {cid: "confirm"}
    page = client.get(response.headers["location"])
    assert "已确认该配对" in page.text

    # Confirmed pair stays certain and blocks both rows in every later result.
    rows, _, counts = db.fetch_result_page(db_path(client), 1, page_size=1000)
    confirmed_row = next(r for r in rows if r["candidate_id"] == cid)
    assert confirmed_row["status"] == "certain"
    assert confirmed_row["selected"] == 1
    assert confirmed_row["decision"] == "confirm"

    # A page refresh still sees the decision and the recomputed result.
    refreshed = client.get("/comparisons/1?status=excluded")
    assert refreshed.status_code == 200


def test_excluding_candidate_updates_results(client):
    upload(client, BASE_A, BASE_B)
    cid = candidate_id(client, 1, 4, 4)  # A3-B3
    other = candidate_id(client, 1, 5, 4)  # A4-B3
    client.post(
        "/comparisons/1/decide",
        data={"candidate_id": cid, "action": "exclude"},
        follow_redirects=False,
    )
    rows, _, _ = db.fetch_result_page(db_path(client), 1, page_size=1000)
    target = next(r for r in rows if r["candidate_id"] == cid)
    assert target["status"] == "excluded"
    assert target["selected"] == 0
    assert db.get_decisions(db_path(client), 1) == {cid: "exclude"}
    rows, _, _ = db.fetch_result_page(db_path(client), 1, page_size=1000)
    remaining = next(r for r in rows if r["candidate_id"] == other)
    assert remaining["status"] == "certain"
    assert remaining["selected"] == 1


def test_conflicting_decision_is_rejected_and_previous_state_kept(client):
    upload(client, BASE_A, BASE_B)
    first = candidate_id(client, 1, 4, 4)  # A3-B3
    second = candidate_id(client, 1, 5, 4)  # A4-B3
    client.post(
        "/comparisons/1/decide",
        data={"candidate_id": first, "action": "confirm"},
        follow_redirects=False,
    )
    before = db.fetch_result_page(db_path(client), 1, page_size=1000)[0]

    response = client.post(
        "/comparisons/1/decide",
        data={"candidate_id": second, "action": "confirm", "status": "all"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "error=" in response.headers["location"]

    assert db.get_decisions(db_path(client), 1) == {first: "confirm"}
    after = db.fetch_result_page(db_path(client), 1, page_size=1000)[0]
    assert [(r["candidate_id"], r["status"], r["selected"]) for r in before] == [
        (r["candidate_id"], r["status"], r["selected"]) for r in after
    ]

    page = client.get(response.headers["location"])
    assert "矛盾决定" in page.text


def test_recompute_failure_keeps_old_state(client, monkeypatch):
    upload(client, BASE_A, BASE_B)
    before = db.fetch_result_page(db_path(client), 1, page_size=1000)[0]

    def boom(*args, **kwargs):
        raise RecomputeError("模拟重算失败")

    monkeypatch.setattr(db, "solve_catalog", boom)
    response = client.post(
        "/comparisons/1/recompute", data={}, follow_redirects=False
    )
    assert "error=" in response.headers["location"]
    page = client.get(response.headers["location"])
    assert "模拟重算失败" in page.text

    after = db.fetch_result_page(db_path(client), 1, page_size=1000)[0]
    assert [(r["candidate_id"], r["status"], r["selected"]) for r in before] == [
        (r["candidate_id"], r["status"], r["selected"]) for r in after
    ]

    comparison = db.get_comparison(db_path(client), 1)
    assert "模拟重算失败" in comparison["last_error"]


def test_recompute_restores_results_after_transient_failure(client, monkeypatch):
    upload(client, BASE_A, BASE_B)
    before = db.fetch_result_page(db_path(client), 1, page_size=1000)[0]
    original = db.solve_catalog
    monkeypatch.setattr(db, "solve_catalog", lambda *a, **k: (_ for _ in ()).throw(RecomputeError("x")))
    client.post("/comparisons/1/recompute", data={}, follow_redirects=False)
    monkeypatch.setattr(db, "solve_catalog", original)

    response = client.post("/comparisons/1/recompute", data={}, follow_redirects=False)
    assert "error=" not in response.headers["location"]
    after = db.fetch_result_page(db_path(client), 1, page_size=1000)[0]
    assert [(r["candidate_id"], r["status"], r["selected"]) for r in before] == [
        (r["candidate_id"], r["status"], r["selected"]) for r in after
    ]
    assert db.get_comparison(db_path(client), 1)["last_error"] is None


def test_export_contains_only_certain_pairs_with_source_lines(client):
    upload(client, BASE_A, BASE_B)
    response = client.get("/comparisons/1/export.csv")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    text = response.content.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert {row["甲馆馆藏号"] for row in rows} == {"A1", "A2"}
    assert all(row["判定来源"] == "自动确定" for row in rows)
    assert {row["甲馆原文件行号"] for row in rows} == {"2", "3"}

    # Uncertain candidates are not exported until a curator confirms them.
    _, _, counts = db.fetch_result_page(db_path(client), 1, page_size=1000)
    assert counts["uncertain"] == 2
    assert len(rows) == 2


def test_export_marks_confirmed_pairs(client):
    upload(client, BASE_A, BASE_B)
    cid = candidate_id(client, 1, 4, 4)  # A3-B3
    client.post(
        "/comparisons/1/decide",
        data={"candidate_id": cid, "action": "confirm"},
        follow_redirects=False,
    )
    response = client.get("/comparisons/1/export.csv")
    rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))
    assert len(rows) == 3
    confirmed = [row for row in rows if row["判定来源"] == "馆员确认"]
    assert len(confirmed) == 1
    assert confirmed[0]["甲馆馆藏号"] == "A3"
    assert confirmed[0]["乙馆馆藏号"] == "B3"


def test_source_view_shows_original_line_numbers(client):
    upload(client, BASE_A, BASE_B)
    page = client.get("/comparisons/1/source/A")
    assert page.status_code == 200
    for line in (2, 3, 4, 5):
        assert f'id="line-{line}"' in page.text
    assert "A2" in page.text and "A4" in page.text


def test_detail_page_reports_original_file_line_numbers(client):
    upload(client, BASE_A, BASE_B)
    page = client.get("/comparisons/1")
    assert 'href="/comparisons/1/source/A#line-2"' in page.text
    assert 'href="/comparisons/1/source/B#line-4"' in page.text


def test_missing_comparison_returns_404(client):
    page = client.get("/comparisons/999")
    assert page.status_code == 404
    assert "对照任务不存在" in page.text
    assert client.get("/comparisons/999/export.csv").status_code == 404


def test_gap_sum_and_selection_match_expected_objective(client):
    upload(client, BASE_A, BASE_B)
    comparison = db.get_comparison(db_path(client), 1)
    assert comparison["selected_count"] == 3
    assert comparison["gap_sum"] == 1
