"""Optimal bipartite matching and uniqueness classification.

Candidate pairs form a bipartite graph between the two museum catalogues.
An edge is a candidate when 采集号 and 县代码 are equal and the collection
dates differ by at most one day.  Its cost is the absolute date difference
in days (0 or 1).

The objective is lexicographic:

1. maximise the number of pairs;
2. among those, minimise the sum of date differences.

Among all optimal solutions an edge is

* ``certain``   - present in every optimal solution;
* ``uncertain`` - present in at least one, but not every, optimal solution;
* ``unmatched`` - present in no optimal solution.

The classification is derived from an optimal min-cost flow.  In its residual
graph a zero-reduced-cost directed cycle corresponds to a cost-neutral
reassignment, so an edge belongs to some optimal solution exactly when it can
join such a cycle.  Strongly connected components of the zero-reduced-cost
residual graph answer that question for every edge at once.

Potentials with non-negative reduced costs on every residual arc are obtained
by shortest paths from a virtual source connected to every node with a
zero-cost arc (Bellman-Ford initialised with all distances at zero).
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Hashable

CERTAIN = "certain"
UNCERTAIN = "uncertain"
UNMATCHED = "unmatched"
EXCLUDED = "excluded"

STATUS_LABELS = {
    CERTAIN: "确定",
    UNCERTAIN: "待核",
    UNMATCHED: "未匹配",
    EXCLUDED: "已排除",
}

DECISION_CONFIRM = "confirm"
DECISION_EXCLUDE = "exclude"


class DecisionConflict(ValueError):
    """The requested set of decisions cannot be satisfied together."""


class RecomputeError(RuntimeError):
    """The matching engine could not produce a result."""


@dataclass(frozen=True)
class Candidate:
    id: int
    a_row_id: int
    b_row_id: int
    gap_days: int


@dataclass
class Solution:
    status: dict[int, str] = field(default_factory=dict)
    selected: dict[int, bool] = field(default_factory=dict)
    selected_count: int = 0
    gap_sum: int = 0
    certain_count: int = 0
    uncertain_count: int = 0
    unmatched_count: int = 0
    excluded_count: int = 0


class _MinCostFlow:
    """Unit-capacity primal-dual min-cost max-flow.

    Each phase runs Dijkstra on reduced costs, then saturates a blocking flow
    through the zero-reduced-cost admissible graph instead of augmenting one
    path at a time.  This keeps large, mostly zero-cost components (a long
    chain of identical dates) fast while remaining exact.
    """

    def __init__(self, node_count: int):
        self.n = node_count
        # Each edge: [to, reverse_index, residual_capacity, cost]
        self.graph: list[list[list]] = [[] for _ in range(node_count)]

    def add_edge(self, u: int, v: int, cost: int) -> tuple[int, int]:
        forward = [v, len(self.graph[v]), 1, cost]
        backward = [u, len(self.graph[u]), 0, -cost]
        self.graph[u].append(forward)
        self.graph[v].append(backward)
        return u, len(self.graph[u]) - 1

    def _shortest_paths(self, source: int, potential: list[int]):
        inf = 10**18
        dist = [inf] * self.n
        dist[source] = 0
        heap = [(0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d != dist[u]:
                continue
            for edge in self.graph[u]:
                v, _, cap, cost = edge
                if cap <= 0:
                    continue
                nd = d + cost + potential[u] - potential[v]
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def _blocking_flow(self, source: int, sink: int, potential: list[int]) -> int:
        """Augment a blocking flow through zero-reduced-cost residual arcs."""
        n = self.n
        pushed = 0
        while True:
            level = [-1] * n
            level[source] = 0
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for edge in self.graph[u]:
                    v, _, cap, cost = edge
                    if (
                        cap > 0
                        and level[v] < 0
                        and cost + potential[u] - potential[v] == 0
                    ):
                        level[v] = level[u] + 1
                        queue.append(v)
            if level[sink] < 0:
                return pushed

            cursor = [0] * n
            while True:
                stack = [source]
                path: list[tuple[int, int]] = []
                while stack:
                    u = stack[-1]
                    if u == sink:
                        break
                    advanced = False
                    while cursor[u] < len(self.graph[u]):
                        idx = cursor[u]
                        v, _, cap, cost = self.graph[u][idx]
                        if (
                            cap > 0
                            and level[v] == level[u] + 1
                            and cost + potential[u] - potential[v] == 0
                        ):
                            stack.append(v)
                            path.append((u, idx))
                            advanced = True
                            break
                        cursor[u] += 1
                    if not advanced:
                        level[u] = -1  # dead end for this phase
                        stack.pop()
                        if path:
                            path.pop()
                if not stack:
                    break
                for u, idx in path:
                    edge = self.graph[u][idx]
                    v = edge[0]
                    edge[2] -= 1
                    self.graph[v][edge[1]][2] += 1
                pushed += 1

    def solve(self, source: int, sink: int) -> int:
        """Return the cost of a maximum flow of minimum cost."""
        potential = [0] * self.n
        total_cost = 0
        while True:
            dist = self._shortest_paths(source, potential)
            if dist[sink] == 10**18:
                break
            for v in range(self.n):
                if dist[v] < 10**18:
                    potential[v] += dist[v]
            pushed = self._blocking_flow(source, sink, potential)
            if pushed == 0:
                raise RecomputeError("增广路径不完整")
            total_cost += pushed * potential[sink]
        return total_cost

    def residual_arcs(self):
        for u, edges in enumerate(self.graph):
            for edge in edges:
                if edge[2] > 0:
                    yield u, edge[0], edge[3]

    def residual_potentials(self) -> list[int]:
        """Shortest distances from a virtual source joined to all nodes."""
        n = self.n
        dist = [0] * n
        in_queue = [True] * n
        queue = deque(range(n))
        relaxations = 0
        limit = max(64, n * n + 8 * sum(len(e) for e in self.graph))
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            du = dist[u]
            for v, _, cap, cost in self.graph[u]:
                if cap <= 0:
                    continue
                if du + cost < dist[v]:
                    dist[v] = du + cost
                    relaxations += 1
                    if relaxations > limit:
                        raise RecomputeError("残量图存在负环，无法判定最优方案")
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
        return dist


def _strongly_connected_components(n: int, adjacency: list[list[int]]) -> list[int]:
    """Iterative Kosaraju; returns component id per node."""
    visited = [False] * n
    order: list[int] = []
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        stack = [(start, 0)]
        while stack:
            node, idx = stack[-1]
            if idx < len(adjacency[node]):
                nxt = adjacency[node][idx]
                stack[-1] = (node, idx + 1)
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append((nxt, 0))
            else:
                order.append(node)
                stack.pop()

    reverse = [[] for _ in range(n)]
    for u in range(n):
        for v in adjacency[u]:
            reverse[v].append(u)
    component = [-1] * n
    current = 0
    for start in reversed(order):
        if component[start] != -1:
            continue
        component[start] = current
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in reverse[node]:
                if component[nxt] == -1:
                    component[nxt] = current
                    stack.append(nxt)
        current += 1
    return component


def _solve_component(edges: list[Candidate]) -> tuple[set[int], dict[int, str], int]:
    """Solve one connected component; return selected ids, statuses, gap sum."""
    left_ids: list[int] = []
    right_ids: list[int] = []
    left_index: dict[int, int] = {}
    right_index: dict[int, int] = {}
    for edge in edges:
        if edge.a_row_id not in left_index:
            left_index[edge.a_row_id] = len(left_ids)
            left_ids.append(edge.a_row_id)
        if edge.b_row_id not in right_index:
            right_index[edge.b_row_id] = len(right_ids)
            right_ids.append(edge.b_row_id)

    source = 0
    left_base = 1
    right_base = left_base + len(left_ids)
    sink = right_base + len(right_ids)
    flow = _MinCostFlow(sink + 1)
    for i in range(len(left_ids)):
        flow.add_edge(source, left_base + i, 0)
    for j in range(len(right_ids)):
        flow.add_edge(right_base + j, sink, 0)

    handles: dict[int, tuple[int, int]] = {}
    for edge in edges:
        u = left_base + left_index[edge.a_row_id]
        v = right_base + right_index[edge.b_row_id]
        handles[edge.id] = flow.add_edge(u, v, edge.gap_days)

    total_cost = flow.solve(source, sink)

    pot = flow.residual_potentials()
    adjacency: list[list[int]] = [[] for _ in range(flow.n)]
    for u, v, cost in flow.residual_arcs():
        reduced = cost + pot[u] - pot[v]
        if reduced < 0:
            raise RecomputeError("残量图存在负约化代价，无法判定最优方案")
        if reduced == 0:
            adjacency[u].append(v)
    component = _strongly_connected_components(flow.n, adjacency)

    selected: set[int] = set()
    status: dict[int, str] = {}
    for edge in edges:
        u, idx = handles[edge.id]
        forward = flow.graph[u][idx]
        v = forward[0]
        used = forward[2] == 0
        reduced = edge.gap_days + pot[u] - pot[v]
        same_component = component[u] == component[v]
        if used:
            selected.add(edge.id)
            status[edge.id] = UNCERTAIN if (reduced == 0 and same_component) else CERTAIN
        else:
            if forward[2] <= 0:
                raise RecomputeError("候选边状态异常")
            status[edge.id] = UNCERTAIN if (reduced == 0 and same_component) else UNMATCHED
    return selected, status, total_cost


class _DSU:
    def __init__(self) -> None:
        self.parent: dict[Hashable, Hashable] = {}
        self.rank: dict[Hashable, int] = {}

    def find(self, item: Hashable) -> Hashable:
        parent = self.parent.setdefault(item, item)
        root = item
        while parent != root:
            root = parent
            parent = self.parent.setdefault(root, root)
        # path compression
        node = item
        while node != root:
            nxt = self.parent[node]
            self.parent[node] = root
            node = nxt
        return root

    def union(self, a: Hashable, b: Hashable) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        rank_a = self.rank.get(ra, 0)
        rank_b = self.rank.get(rb, 0)
        if rank_a < rank_b:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if rank_a == rank_b:
            self.rank[ra] = rank_a + 1


def validate_decisions(
    candidates: list[Candidate],
    decisions: dict[int, str],
    known_ids: set[int] | None = None,
) -> list[Candidate]:
    """Check decisions and return the confirmed candidates.

    Raises DecisionConflict when a decision names an unknown candidate, uses
    an unknown action, or when two confirmed pairs share a specimen row.
    """
    by_id = {c.id: c for c in candidates}
    valid_ids = set(by_id) if known_ids is None else set(known_ids)
    confirmed: list[Candidate] = []
    used_side_a: dict[int, int] = {}
    used_side_b: dict[int, int] = {}
    for candidate_id, decision in decisions.items():
        if decision not in (DECISION_CONFIRM, DECISION_EXCLUDE):
            raise DecisionConflict(f"未知决定: {decision}")
        if candidate_id not in valid_ids:
            raise DecisionConflict("决定引用了不存在的候选配对")
        if decision != DECISION_CONFIRM:
            continue
        candidate = by_id.get(candidate_id)
        if candidate is None:
            raise DecisionConflict("确认的请求不属于任何候选配对")
        checks = (
            (candidate.a_row_id, used_side_a, "甲馆"),
            (candidate.b_row_id, used_side_b, "乙馆"),
        )
        for row_id, used, museum in checks:
            if row_id in used:
                raise DecisionConflict(
                    f"矛盾决定：{museum}同一件标本被两条确认配对同时占用"
                )
            used[row_id] = candidate_id
        confirmed.append(candidate)
    return confirmed


def solve_catalog(
    candidates: list[Candidate],
    decisions: dict[int, str] | None = None,
    all_candidate_ids: set[int] | None = None,
) -> Solution:
    """Compute the optimal matching under the given decisions.

    Confirmed pairs are forced into the solution, excluded pairs are forbidden,
    and the remaining candidates are optimised for pair count then gap sum.

    ``candidates`` may omit candidates decided as ``exclude``; pass every
    candidate id in ``all_candidate_ids`` so that they are reported as
    excluded rather than silently ignored.
    """
    decisions = dict(decisions or {})
    known_ids = set(all_candidate_ids) if all_candidate_ids is not None else set()
    known_ids.update(c.id for c in candidates)
    confirmed = validate_decisions(candidates, decisions, known_ids)
    confirmed_ids = {c.id for c in confirmed}
    excluded_ids = {cid for cid, d in decisions.items() if d == DECISION_EXCLUDE}
    forced_rows: set[tuple[str, int]] = set()
    for c in confirmed:
        forced_rows.add(("a", c.a_row_id))
        forced_rows.add(("b", c.b_row_id))

    active = [
        c
        for c in candidates
        if c.id not in confirmed_ids
        and c.id not in excluded_ids
        and ("a", c.a_row_id) not in forced_rows
        and ("b", c.b_row_id) not in forced_rows
    ]

    dsu = _DSU()
    for c in active:
        dsu.union(("a", c.a_row_id), ("b", c.b_row_id))
    groups: dict[Hashable, list[Candidate]] = {}
    for c in active:
        groups.setdefault(dsu.find(("a", c.a_row_id)), []).append(c)

    solution = Solution()
    for candidate_id in known_ids:
        solution.status[candidate_id] = UNMATCHED
        solution.selected[candidate_id] = False
    for c in confirmed:
        solution.status[c.id] = CERTAIN
        solution.selected[c.id] = True
    for candidate_id in excluded_ids:
        solution.status[candidate_id] = EXCLUDED
        solution.selected[candidate_id] = False

    selected_count = len(confirmed)
    gap_sum = sum(c.gap_days for c in confirmed)
    for group in groups.values():
        selected, status, cost = _solve_component(group)
        selected_count += len(selected)
        gap_sum += cost
        solution.status.update(status)
        for candidate_id in selected:
            solution.selected[candidate_id] = True

    solution.selected_count = selected_count
    solution.gap_sum = gap_sum
    solution.certain_count = sum(1 for s in solution.status.values() if s == CERTAIN)
    solution.uncertain_count = sum(
        1 for s in solution.status.values() if s == UNCERTAIN
    )
    solution.unmatched_count = sum(
        1 for s in solution.status.values() if s == UNMATCHED
    )
    solution.excluded_count = sum(
        1 for s in solution.status.values() if s == EXCLUDED
    )
    return solution
