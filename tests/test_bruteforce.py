"""Brute-force cross-checks for decisions and for the whole graph."""

from __future__ import annotations

import itertools

from app.matching import (
    CERTAIN,
    EXCLUDED,
    UNMATCHED,
    UNCERTAIN,
    Candidate,
    DecisionConflict,
    solve_catalog,
)


def brute_force(edges, decisions):
    """Enumerate every subset honouring the decisions, then apply the objective."""
    count = len(edges)
    solutions: list[frozenset[int]] = []
    best_count = -1
    best_cost = -1
    for mask in range(1 << count):
        chosen = {i for i in range(count) if mask >> i & 1}
        if any(action == "confirm" and i not in chosen for i, action in decisions.items()):
            continue
        if any(action == "exclude" and i in chosen for i, action in decisions.items()):
            continue
        used_a: set[int] = set()
        used_b: set[int] = set()
        cost = 0
        feasible = True
        for i in chosen:
            a, b, gap = edges[i]
            if a in used_a or b in used_b:
                feasible = False
                break
            used_a.add(a)
            used_b.add(b)
            cost += gap
        if not feasible:
            continue
        n = len(chosen)
        if n > best_count or (n == best_count and cost < best_cost):
            best_count, best_cost = n, cost
            solutions = [frozenset(chosen)]
        elif n == best_count and cost == best_cost:
            solutions.append(frozenset(chosen))
    if not solutions:
        return None
    status = {}
    for i in range(count):
        if decisions.get(i) == "exclude":
            status[i] = EXCLUDED
            continue
        everywhere = all(i in solution for solution in solutions)
        somewhere = any(i in solution for solution in solutions)
        status[i] = (
            CERTAIN if everywhere else (UNCERTAIN if somewhere else UNMATCHED)
        )
    return best_count, best_cost, status


def test_matches_brute_force_on_all_small_graphs():
    """Exhaustively compare against enumeration for every small 2-side graph."""
    checked = 0
    for na, nb in ((1, 1), (2, 2), (2, 3), (3, 2), (3, 3)):
        positions = [(a, b) for a in range(na) for b in range(nb)]
        for size in range(1, len(positions) + 1):
            for subset in itertools.combinations(positions, size):
                for costs in itertools.product((0, 1), repeat=len(subset)):
                    edges = [(a, b, c) for (a, b), c in zip(subset, costs)]
                    candidates = [
                        Candidate(i + 1, a + 100, b + 200, c)
                        for i, (a, b, c) in enumerate(edges)
                    ]
                    expected = brute_force(edges, {})
                    solution = solve_catalog(candidates)
                    assert (solution.selected_count, solution.gap_sum) == expected[:2]
                    assert solution.status == {
                        i + 1: expected[2][i] for i in range(len(edges))
                    }
                    checked += 1
    assert checked > 500


def test_matches_brute_force_with_decisions():
    """All 3x3 graphs with zero/one gap and a smattering of curator decisions."""
    import random

    rng = random.Random(23)
    checked = 0
    for _ in range(1500):
        na = rng.randint(1, 3)
        nb = rng.randint(1, 4)
        positions = [(a, b) for a in range(na) for b in range(nb)]
        rng.shuffle(positions)
        size = rng.randint(1, min(len(positions), 8))
        edges = [
            (a, b, rng.choice((0, 0, 1))) for a, b in positions[:size]
        ]
        decisions = {}
        for i in range(size):
            if rng.random() < 0.25:
                decisions[i] = rng.choice(("confirm", "exclude"))
        candidates = [
            Candidate(i + 1, a + 100, b + 200, c)
            for i, (a, b, c) in enumerate(edges)
        ]
        db_decisions = {i + 1: action for i, action in decisions.items()}
        expected = brute_force(edges, decisions)
        if expected is None:
            try:
                solve_catalog(
                    candidates, db_decisions, all_candidate_ids=set(range(1, size + 1))
                )
            except DecisionConflict:
                continue
            raise AssertionError("expected a decision conflict")
        solution = solve_catalog(
            candidates, db_decisions, all_candidate_ids=set(range(1, size + 1))
        )
        assert (solution.selected_count, solution.gap_sum) == expected[:2]
        assert solution.status == {i + 1: expected[2][i] for i in range(size)}
        checked += 1
    assert checked > 1000
