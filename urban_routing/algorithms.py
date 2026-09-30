"""Optimal path finding (Design Decision Area 2).

Four algorithms are implemented so the choice can be argued with numbers:

    dijkstra       lazy-deletion binary heap     O((V + E) log V)   chosen
    a_star         Dijkstra + admissible bound   O((V + E) log V) worst case,
                                                 far fewer expansions in practice
    bellman_ford   V-1 relaxation rounds         O(V * E)           baseline
    bfs_min_hops   queue, unweighted             O(V + E)           hop counting

Why Dijkstra is the engine's default
------------------------------------
* Every metric we expose (distance, travel time, balanced) is non-negative, so
  Dijkstra's greedy invariant holds; negative-cycle handling would be dead
  weight.
* On a sparse road network E ~= 3V, so O((V+E) log V) ~= O(V log V).  Bellman-
  Ford's O(V*E) ~= O(V^2) is three orders of magnitude worse at V = 10,000.
* It answers "one-to-all" for free, which is exactly what the reachability
  query needs -- one implementation, two features.

Why A* is offered alongside it
------------------------------
For a single point-to-point query with coordinates available, A* expands a
fraction of the nodes.  Its guarantee needs an *admissible* heuristic (never
overestimates).  We use straight-line distance, divided by the network's top
speed when optimising for time -- no road can beat a straight line at the
maximum legal speed, so the bound is safe.  A* degrades to Dijkstra exactly
when the heuristic is 0, which is what we pass for metrics that have no
geometric bound.

Heap choice: Python's ``heapq`` is a binary heap with no decrease-key.  We use
*lazy deletion* -- push a new entry and skip stale pops -- which bounds the heap
at O(E) entries and keeps the log factor.  A Fibonacci heap would give
O(E + V log V) in theory but loses badly on constants at these sizes.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .constraints import RouteConstraints
from .graph import TransportNetwork
from .model import Edge, Metric


@dataclass(slots=True)
class RouteResult:
    """Outcome of a routing query."""

    found: bool
    path: list[str] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    cost: float = float("inf")
    nodes_expanded: int = 0
    edges_relaxed: int = 0
    algorithm: str = ""

    @property
    def total_distance_km(self) -> float:
        return sum(e.distance_km for e in self.edges)

    @property
    def total_time_min(self) -> float:
        return sum(e.travel_time_min for e in self.edges)

    @property
    def hops(self) -> int:
        return len(self.edges)

    def describe(self, network: TransportNetwork | None = None) -> str:
        if not self.found:
            return f"[{self.algorithm}] no route (expanded {self.nodes_expanded} nodes)"
        if network is not None:
            names = " -> ".join(network.node(n).name for n in self.path)
        else:
            names = " -> ".join(self.path)
        return (
            f"[{self.algorithm}] cost={self.cost:.3f} "
            f"({self.total_distance_km:.2f} km, {self.total_time_min:.1f} min, "
            f"{self.hops} hops, {self.nodes_expanded} nodes expanded)\n    {names}"
        )


def _reconstruct(
    parent: dict[str, tuple[str, Edge]], source: str, target: str
) -> tuple[list[str], list[Edge]]:
    path: list[str] = [target]
    edges: list[Edge] = []
    cur = target
    while cur != source:
        prev, edge = parent[cur]
        edges.append(edge)
        path.append(prev)
        cur = prev
    path.reverse()
    edges.reverse()
    return path, edges


# ---------------------------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------------------------


def dijkstra(
    network: TransportNetwork,
    source: str,
    target: str | None = None,
    metric: Callable[[Edge], float] = Metric.time,
    constraints: RouteConstraints | None = None,
) -> RouteResult:
    """Single-source shortest path with a binary heap and lazy deletion.

    Time  : O((V + E) log V)
    Space : O(V) for the distance/parent maps plus O(E) worst case heap entries.

    With ``target=None`` this is the full one-to-all shortest-path tree; the
    result then carries the cost map via :func:`dijkstra_costs`.
    """
    if source not in network:
        raise KeyError(source)
    constraints = constraints or RouteConstraints()

    dist: dict[str, float] = {source: 0.0}
    parent: dict[str, tuple[str, Edge]] = {}
    settled: set[str] = set()
    heap: list[tuple[float, str]] = [(0.0, source)]
    expanded = relaxed = 0

    while heap:
        d, u = heapq.heappop(heap)
        if u in settled:            # stale entry left by lazy deletion
            continue
        settled.add(u)
        expanded += 1
        if target is not None and u == target:
            path, edges = _reconstruct(parent, source, target)
            return RouteResult(True, path, edges, d, expanded, relaxed, "dijkstra")
        for edge in network.neighbours(u):
            if not constraints.allows_edge(edge):
                continue
            v = edge.target
            if v in settled:
                continue
            relaxed += 1
            nd = d + metric(edge)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = (u, edge)
                heapq.heappush(heap, (nd, v))

    if target is None:
        res = RouteResult(True, [], [], 0.0, expanded, relaxed, "dijkstra")
        res.path = list(dist)
        return res
    return RouteResult(False, [], [], float("inf"), expanded, relaxed, "dijkstra")


def dijkstra_costs(
    network: TransportNetwork,
    source: str,
    metric: Callable[[Edge], float] = Metric.time,
    constraints: RouteConstraints | None = None,
    budget: float = float("inf"),
) -> tuple[dict[str, float], dict[str, tuple[str, Edge]]]:
    """One-to-all costs, optionally pruned at ``budget``.

    This is the primitive behind reachability analysis.  Pruning at the budget
    turns the full O((V+E) log V) scan into work proportional only to the
    reachable sub-network, which is what makes "everywhere within 10 minutes"
    cheap on a large city graph.
    """
    constraints = constraints or RouteConstraints()
    dist: dict[str, float] = {source: 0.0}
    parent: dict[str, tuple[str, Edge]] = {}
    settled: set[str] = set()
    heap: list[tuple[float, str]] = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        for edge in network.neighbours(u):
            if not constraints.allows_edge(edge):
                continue
            nd = d + metric(edge)
            if nd > budget:                     # prune: costs only grow
                continue
            v = edge.target
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = (u, edge)
                heapq.heappush(heap, (nd, v))
    return {k: v for k, v in dist.items() if v <= budget}, parent


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------


def _heuristic_factory(
    network: TransportNetwork, target: str, metric: Callable[[Edge], float]
) -> Callable[[str], float]:
    """Build an admissible heuristic for the given metric.

    distance metric  -> straight-line km (never longer than any road path)
    time metrics     -> straight-line km / fastest speed in the network,
                        converted to minutes
    anything else    -> 0 (A* safely degenerates to Dijkstra)
    """
    goal = network.node(target)
    name = getattr(metric, "__name__", "")

    if metric is Metric.distance or name == "distance":
        return lambda nid: network.node(nid).distance_to(goal)

    if metric in (Metric.time, Metric.free_flow_time) or name in ("time", "free_flow_time"):
        top_speed = max((e.speed_kmh for e in network.edges()), default=1.0)
        return lambda nid: 60.0 * network.node(nid).distance_to(goal) / top_speed

    return lambda nid: 0.0


def a_star(
    network: TransportNetwork,
    source: str,
    target: str,
    metric: Callable[[Edge], float] = Metric.time,
    constraints: RouteConstraints | None = None,
    heuristic: Callable[[str], float] | None = None,
) -> RouteResult:
    """Goal-directed shortest path.

    Time  : O((V + E) log V) worst case (heuristic = 0), typically far less.
    Space : O(V).

    Optimality requires an admissible heuristic; ``_heuristic_factory``
    guarantees one for the metrics we ship, and returns 0 for anything it
    cannot bound.
    """
    constraints = constraints or RouteConstraints()
    h = heuristic or _heuristic_factory(network, target, metric)

    g: dict[str, float] = {source: 0.0}
    parent: dict[str, tuple[str, Edge]] = {}
    settled: set[str] = set()
    heap: list[tuple[float, float, str]] = [(h(source), 0.0, source)]
    expanded = relaxed = 0

    while heap:
        _, gu, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        expanded += 1
        if u == target:
            path, edges = _reconstruct(parent, source, target)
            return RouteResult(True, path, edges, gu, expanded, relaxed, "a_star")
        for edge in network.neighbours(u):
            if not constraints.allows_edge(edge):
                continue
            v = edge.target
            if v in settled:
                continue
            relaxed += 1
            ng = gu + metric(edge)
            if ng < g.get(v, float("inf")):
                g[v] = ng
                parent[v] = (u, edge)
                heapq.heappush(heap, (ng + h(v), ng, v))

    return RouteResult(False, [], [], float("inf"), expanded, relaxed, "a_star")


# ---------------------------------------------------------------------------
# Bellman-Ford (comparison baseline)
# ---------------------------------------------------------------------------


def bellman_ford(
    network: TransportNetwork,
    source: str,
    target: str | None = None,
    metric: Callable[[Edge], float] = Metric.time,
    constraints: RouteConstraints | None = None,
) -> RouteResult:
    """Relax every edge V-1 times.

    Time  : O(V * E)   Space: O(V)

    Included as the honest comparison point.  It tolerates negative weights,
    which this domain never produces (a road cannot take negative time), so it
    pays a large asymptotic penalty for a capability we do not need.  The early
    exit when a round changes nothing helps on shallow graphs but does not
    change the worst case.
    """
    constraints = constraints or RouteConstraints()
    dist: dict[str, float] = {nid: float("inf") for nid in network.node_ids()}
    dist[source] = 0.0
    parent: dict[str, tuple[str, Edge]] = {}
    relaxed = rounds = 0

    edges = [e for e in network.edges() if constraints.allows_edge(e)]
    for _ in range(network.node_count - 1):
        rounds += 1
        changed = False
        for edge in edges:
            du = dist[edge.source]
            if du == float("inf"):
                continue
            relaxed += 1
            nd = du + metric(edge)
            if nd < dist[edge.target]:
                dist[edge.target] = nd
                parent[edge.target] = (edge.source, edge)
                changed = True
        if not changed:
            break

    if target is None:
        res = RouteResult(True, list(dist), [], 0.0, rounds, relaxed, "bellman_ford")
        return res
    if dist.get(target, float("inf")) == float("inf"):
        return RouteResult(False, [], [], float("inf"), rounds, relaxed, "bellman_ford")
    path, path_edges = _reconstruct(parent, source, target)
    return RouteResult(
        True, path, path_edges, dist[target], rounds, relaxed, "bellman_ford"
    )


# ---------------------------------------------------------------------------
# BFS (unweighted / minimum transfers)
# ---------------------------------------------------------------------------


def bfs_min_hops(
    network: TransportNetwork,
    source: str,
    target: str,
    constraints: RouteConstraints | None = None,
) -> RouteResult:
    """Fewest connections, ignoring weights.

    Time: O(V + E).  Space: O(V) for the queue and visited set.

    Useful in its own right -- "fewest turns / fewest interchanges" is a real
    commuter preference -- and it is the traversal that connectivity checking
    reuses.  It is *not* a substitute for Dijkstra: minimum hops is not minimum
    time once edges carry weights.
    """
    constraints = constraints or RouteConstraints()
    if source == target:
        return RouteResult(True, [source], [], 0.0, 1, 0, "bfs")

    parent: dict[str, tuple[str, Edge]] = {}
    seen = {source}
    queue: deque[str] = deque([source])
    expanded = relaxed = 0

    while queue:
        u = queue.popleft()
        expanded += 1
        for edge in network.neighbours(u):
            if not constraints.allows_edge(edge):
                continue
            v = edge.target
            if v in seen:
                continue
            relaxed += 1
            seen.add(v)
            parent[v] = (u, edge)
            if v == target:
                path, edges = _reconstruct(parent, source, target)
                return RouteResult(
                    True, path, edges, float(len(edges)), expanded, relaxed, "bfs"
                )
            queue.append(v)

    return RouteResult(False, [], [], float("inf"), expanded, relaxed, "bfs")
