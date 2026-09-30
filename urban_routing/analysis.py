"""Traversal, reachability and bottleneck analysis (Decision Area 3).

Choosing a traversal strategy
-----------------------------
+-------------------------------+-----------------+--------------------------+
| query                         | strategy        | auxiliary structure      |
+-------------------------------+-----------------+--------------------------+
| reachable within a *budget*   | Dijkstra with   | binary heap  O(V)        |
| (weighted: minutes or km)     | early pruning   | + cost map   O(V)        |
| reachable within k *hops*     | BFS             | FIFO queue   O(V)        |
| "is the network connected?"   | BFS (iterative) | FIFO queue   O(V)        |
| cut vertices / cut edges      | DFS (Tarjan)    | explicit stack + two     |
|                               |                 | int arrays   O(V)        |
+-------------------------------+-----------------+--------------------------+

Why not BFS for the budget query?  BFS visits in order of *hop count*, which
says nothing about minutes travelled: a single 8 km motorway link is one hop and
five minutes, while eight residential turns are eight hops and four minutes.
Only a cost-ordered frontier -- Dijkstra's heap -- can settle nodes in order of
travel budget consumed, and that ordering is what makes the prune sound: once a
popped node exceeds the budget every node behind it does too.

Why BFS and not DFS for connectivity?  Either is O(V + E) and either answers the
question.  BFS is preferred because its queue depth is bounded by the width of
the frontier, whereas recursive DFS on a 50,000-node network can exhaust the
Python call stack.  Where DFS *is* required -- articulation points need DFS
discovery/low-link times, which BFS cannot produce -- we implement it
iteratively with an explicit stack for the same reason.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .algorithms import dijkstra_costs
from .constraints import RouteConstraints
from .graph import TransportNetwork
from .model import Edge, Metric


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReachabilityResult:
    source: str
    budget: float
    metric_name: str
    costs: dict[str, float] = field(default_factory=dict)
    parent: dict[str, tuple[str, Edge]] = field(default_factory=dict)

    @property
    def reached(self) -> int:
        return len(self.costs)

    def coverage(self, network: TransportNetwork) -> float:
        return self.reached / network.node_count if network.node_count else 0.0

    def within(self, limit: float) -> list[str]:
        return sorted((n for n, c in self.costs.items() if c <= limit),
                      key=lambda n: self.costs[n])

    def farthest(self, k: int = 5) -> list[tuple[str, float]]:
        return sorted(self.costs.items(), key=lambda kv: -kv[1])[:k]


def reachable_within(
    network: TransportNetwork,
    source: str,
    budget: float,
    metric: Callable[[Edge], float] = Metric.time,
    constraints: RouteConstraints | None = None,
) -> ReachabilityResult:
    """All nodes reachable from ``source`` within ``budget``.

    Implemented as a Dijkstra scan that discards any relaxation exceeding the
    budget.  Because edge costs are non-negative the cost of a node only grows
    along a path, so pruning is safe -- nothing beyond the budget can lead back
    inside it.

    Time  : O((V' + E') log V') where V'/E' is the *reachable* sub-network, so
            a small budget on a big city costs far less than a full scan.
    Space : O(V') for the cost map, parent map and heap.
    """
    costs, parent = dijkstra_costs(network, source, metric, constraints, budget)
    return ReachabilityResult(
        source, budget, getattr(metric, "__name__", str(metric)), costs, parent
    )


def reachable_within_hops(
    network: TransportNetwork,
    source: str,
    max_hops: int,
    constraints: RouteConstraints | None = None,
) -> dict[str, int]:
    """Nodes within ``max_hops`` connections.  BFS, O(V + E)."""
    constraints = constraints or RouteConstraints()
    depth = {source: 0}
    q: deque[str] = deque([source])
    while q:
        u = q.popleft()
        if depth[u] >= max_hops:
            continue
        for edge in network.neighbours(u):
            if not constraints.allows_edge(edge):
                continue
            if edge.target not in depth:
                depth[edge.target] = depth[u] + 1
                q.append(edge.target)
    return depth


def isochrone_bands(
    result: ReachabilityResult, bands: list[float]
) -> dict[float, int]:
    """Count how many nodes fall inside each successive budget band.

    This is the data behind an isochrone map ("what can a paramedic reach in
    5 / 10 / 15 minutes?").
    """
    out: dict[float, int] = {}
    for b in sorted(bands):
        out[b] = sum(1 for c in result.costs.values() if c <= b)
    return out


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConnectivityReport:
    connected: bool
    component_count: int
    components: list[list[str]]
    largest_component: int
    isolated: list[str]

    def describe(self) -> str:
        if self.connected:
            return f"Network is connected: 1 component of {self.largest_component} nodes."
        sizes = ", ".join(str(len(c)) for c in self.components[:6])
        return (
            f"Network is NOT connected: {self.component_count} components "
            f"(sizes {sizes}{'...' if self.component_count > 6 else ''}), "
            f"{len(self.isolated)} isolated node(s)."
        )


def connected_components(
    network: TransportNetwork, constraints: RouteConstraints | None = None
) -> ConnectivityReport:
    """Weakly connected components via iterative BFS.  O(V + E).

    "Weakly" = direction ignored.  For a city that is the right question: a
    one-way street does not make a district unreachable, it only changes how you
    get in and out.  Strong connectivity is checked separately.
    """
    constraints = constraints or RouteConstraints()
    seen: set[str] = set()
    components: list[list[str]] = []

    for start in network.node_ids():
        if start in seen:
            continue
        comp: list[str] = []
        q: deque[str] = deque([start])
        seen.add(start)
        while q:
            u = q.popleft()
            comp.append(u)
            for e in network.neighbours(u):
                if constraints.allows_edge(e) and e.target not in seen:
                    seen.add(e.target)
                    q.append(e.target)
            for e in network.incoming(u):
                if constraints.allows_edge(e) and e.source not in seen:
                    seen.add(e.source)
                    q.append(e.source)
        components.append(comp)

    components.sort(key=len, reverse=True)
    return ConnectivityReport(
        connected=len(components) == 1,
        component_count=len(components),
        components=components,
        largest_component=len(components[0]) if components else 0,
        isolated=[c[0] for c in components if len(c) == 1],
    )


def is_strongly_connected(
    network: TransportNetwork, constraints: RouteConstraints | None = None
) -> bool:
    """Can every node reach every other *respecting direction*?

    Kosaraju's single-source test: BFS forward from one node, then BFS backward.
    If both reach all nodes the digraph is strongly connected.  O(V + E).
    """
    constraints = constraints or RouteConstraints()
    ids = list(network.node_ids())
    if not ids:
        return True
    start = ids[0]

    def _bfs(forward: bool) -> int:
        seen = {start}
        q: deque[str] = deque([start])
        while q:
            u = q.popleft()
            edges = network.neighbours(u) if forward else network.incoming(u)
            for e in edges:
                if not constraints.allows_edge(e):
                    continue
                v = e.target if forward else e.source
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return len(seen)

    return _bfs(True) == len(ids) and _bfs(False) == len(ids)


# ---------------------------------------------------------------------------
# Bottleneck detection: articulation points and bridges
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BottleneckReport:
    articulation_points: list[str]
    bridges: list[tuple[str, str]]
    #: node -> how many components the network falls into if it is removed
    fragmentation: dict[str, int] = field(default_factory=dict)

    def describe(self, network: TransportNetwork, limit: int = 10) -> str:
        lines = [
            f"Critical nodes (articulation points): {len(self.articulation_points)}"
        ]
        for nid in self.articulation_points[:limit]:
            frag = self.fragmentation.get(nid)
            frag_txt = f" -> splits network into {frag} components" if frag else ""
            lines.append(f"  - {nid} {network.node(nid).name}{frag_txt}")
        lines.append(f"Critical connections (bridges): {len(self.bridges)}")
        for u, v in self.bridges[:limit]:
            edge = network.edge_between(u, v) or network.edge_between(v, u)
            label = f" [{edge.name}]" if edge and edge.name else ""
            lines.append(
                f"  - {u} {network.node(u).name} <-> {v} {network.node(v).name}{label}"
            )
        return "\n".join(lines)


def find_bottlenecks(
    network: TransportNetwork, with_fragmentation: bool = True
) -> BottleneckReport:
    """Tarjan's articulation-point / bridge algorithm, iteratively.

    Runs one DFS over the *undirected* view of the network, recording for each
    node its discovery time ``disc[u]`` and the earliest discovery time
    reachable from its subtree without using the tree edge back to its parent,
    ``low[u]``.

        u is an articulation point  <=>  u is the DFS root with >1 child, or
                                         u has a child c with low[c] >= disc[u]
        (u,v) is a bridge           <=>  low[v] > disc[u] for tree edge u->v

    Time : O(V + E) -- one DFS.  Space : O(V) for disc/low/stack.

    Compare with the naive approach: remove each node, re-run a connectivity
    check, put it back.  That is O(V * (V + E)) -- on the 100-node study set,
    100 full traversals instead of one.  The naive version is implemented below
    as ``verify_critical_nodes`` and used to *validate* this one in the tests.

    The undirected view is the right model here: a bridge collapsing removes
    both directions of travel, and an intersection closed for works is closed to
    everyone.

    ``with_fragmentation`` additionally reports, for each critical node, how
    many pieces the network falls into without it.  That is a *reporting*
    extra, not part of Tarjan: it costs one extra traversal per articulation
    point, i.e. O(A * (V + E)).  It is worth it for a human-readable report on a
    city-sized network and must be switched off when timing the algorithm
    itself.
    """
    disc: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    articulation: set[str] = set()
    bridges: list[tuple[str, str]] = []
    timer = 0

    adjacency = {n: sorted(network.undirected_neighbours(n)) for n in network.node_ids()}

    for root in network.node_ids():
        if root in disc:
            continue
        parent[root] = None
        root_children = 0
        # stack entries: (node, iterator position into its neighbour list)
        stack: list[tuple[str, int]] = [(root, 0)]
        disc[root] = low[root] = timer
        timer += 1

        while stack:
            u, i = stack[-1]
            neigh = adjacency[u]
            if i < len(neigh):
                stack[-1] = (u, i + 1)
                v = neigh[i]
                if v == parent.get(u):
                    continue
                if v in disc:                      # back edge
                    low[u] = min(low[u], disc[v])
                else:                              # tree edge
                    parent[v] = u
                    disc[v] = low[v] = timer
                    timer += 1
                    stack.append((v, 0))
                    if u == root:
                        root_children += 1
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[u])
                    if p != root and low[u] >= disc[p]:
                        articulation.add(p)
                    if low[u] > disc[p]:
                        bridges.append((p, u))
        if root_children > 1:
            articulation.add(root)

    ordered = sorted(articulation)
    return BottleneckReport(
        articulation_points=ordered,
        bridges=sorted(bridges),
        fragmentation=(
            {nid: _components_without(network, nid) for nid in ordered}
            if with_fragmentation else {}
        ),
    )


def _components_without(network: TransportNetwork, removed: str) -> int:
    """Component count of the undirected view with ``removed`` deleted."""
    seen = {removed}
    count = 0
    for start in network.node_ids():
        if start in seen:
            continue
        count += 1
        q: deque[str] = deque([start])
        seen.add(start)
        while q:
            u = q.popleft()
            for v in network.undirected_neighbours(u):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
    return count


def verify_critical_nodes(network: TransportNetwork) -> list[str]:
    """Brute-force articulation points: O(V * (V + E)).

    Only used to cross-check Tarjan's result in the test suite and to quantify
    the speed-up in the report.  Never called on the hot path.
    """
    baseline = _components_without(network, "\0none\0")
    return sorted(
        nid for nid in network.node_ids()
        if _components_without(network, nid) > baseline
    )


def verify_critical_edges(network: TransportNetwork) -> list[tuple[str, str]]:
    """Brute-force bridges over the undirected view: O(E * (V + E))."""
    seen_pairs: set[tuple[str, str]] = set()
    for e in network.edges():
        seen_pairs.add((min(e.source, e.target), max(e.source, e.target)))

    def components_without_pair(a: str, b: str) -> int:
        seen: set[str] = set()
        count = 0
        for start in network.node_ids():
            if start in seen:
                continue
            count += 1
            q: deque[str] = deque([start])
            seen.add(start)
            while q:
                u = q.popleft()
                for v in network.undirected_neighbours(u):
                    if {u, v} == {a, b}:
                        continue
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
        return count

    baseline = _components_without(network, "\0none\0")
    return sorted(p for p in seen_pairs if components_without_pair(*p) > baseline)


def edge_betweenness_top(
    network: TransportNetwork,
    metric: Callable[[Edge], float] = Metric.time,
    sample: int | None = None,
    top: int = 10,
    seed: int = 7,
) -> list[tuple[tuple[str, str], int]]:
    """Rank connections by how many shortest paths use them.

    A link need not be a bridge to be critical: a bridge carrying 40% of all
    optimal routes is a bottleneck even if a detour exists.  We approximate edge
    betweenness by building the shortest-path tree from every source (or a
    random sample of ``sample`` sources) and counting tree-edge usage.

    Time: O(S * (V + E) log V) for S sources.
    """
    import random as _random

    sources = list(network.node_ids())
    if sample is not None and sample < len(sources):
        sources = _random.Random(seed).sample(sources, sample)

    tally: dict[tuple[str, str], int] = {}
    for s in sources:
        _, parent = dijkstra_costs(network, s, metric)
        for _, (pu, edge) in parent.items():
            key = (edge.source, edge.target)
            tally[key] = tally.get(key, 0) + 1
    return sorted(tally.items(), key=lambda kv: -kv[1])[:top]
