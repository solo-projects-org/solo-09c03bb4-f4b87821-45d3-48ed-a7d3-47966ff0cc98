from __future__ import annotations

import csv
import io

import pytest

from app import domain
from app.matching import (
    CERTAIN,
    EXCLUDED,
    UNMATCHED,
    UNCERTAIN,
    Candidate,
    DecisionConflict,
    solve_catalog,
)

from app.matching import validate_decisions


HEADER = "馆藏号,采集号,县代码,采集日期\n"


def csv_bytes(rows, encoding="utf-8-sig"):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["馆藏号", "采集号", "县代码", "采集日期"])
    writer.writerows(rows)
    return buffer.getvalue().encode(encoding)


def test_parse_csv_accepts_aliases_and_encodings():
    raw = (
        "Collection Number,Field Number,County Code,Collection Date\n"
        "A1,F1,530100,2024/03/01\n"
    ).encode()
    rows, errors = domain.parse_csv(raw)
    assert not errors
    assert rows == [
        {
            "line_no": 2,
            "collection_number": "A1",
            "field_number": "F1",
            "county_code": "530100",
            "collected_on": "2024-03-01",
        }
    ]

    gbk = (
        "馆藏号,采集号,县代码,采集日期\nA2,F2,530100,20240302\n"
    ).encode("gb18030")
    rows, errors = domain.parse_csv(gbk)
    assert not errors and rows[0]["collection_number"] == "A2"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2024-3-1", "2024-03-01"),
        ("2024/03/01", "2024-03-01"),
        ("2024年3月1日", "2024-03-01"),
        ("2024-03-01 12:30", "2024-03-01"),
        ("2024.03.01", "2024-03-01"),
    ],
)
def test_parse_date_formats(text, expected):
    assert domain.parse_date(text).isoformat() == expected


def test_semicolon_delimiter_is_detected():
    raw = "馆藏号;采集号;县代码;采集日期\nA1;F1;530100;2024-01-01\n".encode()
    rows, errors = domain.parse_csv(raw)
    assert not errors
    assert rows[0]["field_number"] == "F1"


def test_parse_csv_reports_duplicate_collection_numbers():
    raw = (HEADER + "A1,F1,530100,2024-03-01\nA1,F2,530100,2024-03-02\n").encode()
    rows, errors = domain.parse_csv(raw)
    assert len(rows) == 1
    assert any("重复" in message and "第 3 行" in message for message in errors)


def test_parse_csv_reports_bad_dates_and_missing_columns():
    raw = (HEADER + "A1,F1,530100,not-a-date\n").encode()
    rows, errors = domain.parse_csv(raw)
    assert not rows
    assert any("第 2 行" in message for message in errors)

    with pytest.raises(domain.RowError):
        domain.parse_csv("馆藏号,采集号\nA1,F1\n".encode())


def test_parse_csv_line_numbers_skip_blank_lines():
    raw = (HEADER + "A1,F1,530100,2024-03-01\n\nA2,F2,530100,2024-03-02\n").encode()
    rows, errors = domain.parse_csv(raw)
    assert not errors
    assert [row["line_no"] for row in rows] == [2, 4]


def test_candidate_rule_requires_same_field_county_and_one_day():
    a = {"field_number": "F1", "county_code": "530100", "collected_on": "2024-03-01"}
    b = {"field_number": "F1", "county_code": "530100", "collected_on": "2024-03-02"}
    assert domain.is_candidate(a, b)
    c = dict(b, collected_on="2024-03-03")
    assert not domain.is_candidate(a, c)
    d = dict(b, county_code="530200")
    assert not domain.is_candidate(a, d)
    e = dict(b, field_number="F9")
    assert not domain.is_candidate(a, e)


def test_single_possible_pair_is_certain():
    solution = solve_catalog([Candidate(1, 1, 1, 0)])
    assert solution.status[1] == CERTAIN
    assert solution.selected[1]
    assert solution.selected_count == 1
    assert solution.gap_sum == 0


def test_tie_between_two_equal_pairs_marks_both_uncertain():
    # A1-B1 and A2-B2 both cost 1, but they share no specimen: both fit.
    candidates = [Candidate(1, 10, 20, 1), Candidate(2, 11, 21, 1)]
    solution = solve_catalog(candidates)
    assert solution.selected_count == 2
    assert solution.status[1] == CERTAIN and solution.status[2] == CERTAIN

    # A1 can take B1 or B2 at the same cost; only one pair fits.
    tied = [Candidate(1, 10, 20, 1), Candidate(2, 10, 21, 1)]
    solution = solve_catalog(tied)
    assert solution.selected_count == 1
    assert solution.gap_sum == 1
    assert {solution.status[1], solution.status[2]} == {UNCERTAIN}


def test_prefers_more_pairs_even_with_larger_gap():
    # A1-B1 cost 0 blocks A2-B2; choosing A1-B2 (cost 1) allows A2-B1 (cost 1).
    candidates = [
        Candidate(1, 10, 21, 1),
        Candidate(2, 11, 20, 1),
        Candidate(3, 10, 20, 0),
    ]
    solution = solve_catalog(candidates)
    assert solution.selected_count == 2
    assert solution.gap_sum == 2
    assert solution.status[3] == UNMATCHED
    assert solution.status[1] == CERTAIN
    assert solution.status[2] == CERTAIN


def test_minimises_gap_sum_among_maximum_matchings():
    # Two ways to match 2 pairs: cost 0+1 versus 1+1.
    candidates = [
        Candidate(1, 10, 20, 0),
        Candidate(2, 11, 21, 1),
        Candidate(3, 10, 21, 1),
        Candidate(4, 11, 20, 1),
    ]
    solution = solve_catalog(candidates)
    assert solution.selected_count == 2
    assert solution.gap_sum == 1
    assert solution.selected[1] and solution.selected[2]
    assert solution.status[1] == CERTAIN and solution.status[2] == CERTAIN


def test_diamond_marks_every_edge_uncertain():
    candidates = [
        Candidate(1, 10, 20, 1),
        Candidate(2, 10, 21, 1),
        Candidate(3, 11, 20, 1),
        Candidate(4, 11, 21, 1),
    ]
    solution = solve_catalog(candidates)
    assert solution.selected_count == 2
    assert all(solution.status[i] == UNCERTAIN for i in (1, 2, 3, 4))


def test_confirmed_pair_is_forced_and_blocks_neighbours():
    confirmed = Candidate(1, 10, 20, 1)
    candidates = [
        confirmed,
        Candidate(2, 10, 21, 1),
        Candidate(3, 11, 20, 1),
    ]
    solution = solve_catalog(candidates, {1: "confirm"})
    assert solution.selected[1]
    assert solution.status[1] == CERTAIN
    assert not solution.selected[2]
    assert not solution.selected[3]
    assert solution.status[2] == UNMATCHED
    assert solution.status[3] == UNMATCHED


def test_excluded_candidate_is_reported_as_excluded():
    candidates = [Candidate(1, 10, 20, 0), Candidate(2, 11, 21, 1)]
    solution = solve_catalog(candidates, {1: "exclude"})
    assert solution.status[1] == EXCLUDED
    assert not solution.selected[1]
    assert solution.selected[2]


def test_conflicting_confirmations_are_rejected():
    candidates = [
        Candidate(1, 10, 20, 0),
        Candidate(2, 10, 21, 1),
    ]
    with pytest.raises(DecisionConflict):
        validate_decisions(candidates, {1: "confirm", 2: "confirm"})


def test_unknown_candidate_decision_is_rejected():
    with pytest.raises(DecisionConflict):
        solve_catalog([Candidate(1, 10, 20, 0)], {99: "confirm"})


def test_matching_scales_to_a_long_chain():
    n = 200
    candidates = []
    identifier = 1
    for a in range(n):
        for delta in (0, 1):
            if a + delta < n:
                candidates.append(Candidate(identifier, a, a + delta, delta))
                identifier += 1
    solution = solve_catalog(candidates)
    assert solution.selected_count == n
    assert solution.gap_sum >= 0
