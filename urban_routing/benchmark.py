"""Performance evaluation.

Everything here answers one question: how does the engine behave as the network
grows?  Results are written as CSV (raw numbers) and SVG (charts) into
``results/`` so the report can cite them and a marker can re-derive them.

Method
------
* Each measurement repeats a query over ``repeats`` random source/target pairs
  drawn from a fixed seed, and reports the mean wall-clock time per query using
  ``time.perf_counter``.
* A warm-up query runs first and is discarded, so the first-call overhead of
  building the heuristic closure does not land in the mean.
* Memory is measured with ``tracemalloc`` peak allocation, which counts Python
  objects rather than resident set size -- the right measure for comparing two
  data structures.
* Bellman-Ford is capped at a smaller maximum size; its O(V*E) cost makes the
  large sizes take minutes and the trend is already unambiguous by then.
"""

from __future__ import annotations

import csv
import gc
import random
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .algorithms import a_star, bellman_ford, bfs_min_hops, dijkstra, dijkstra_costs
from .analysis import find_bottlenecks, reachable_within, verify_critical_nodes
from .charts import Series, bar_chart, line_chart
from .constraints import RouteConstraints
from .dataset import generate_synthetic_network
from .graph import MatrixNetwork, TransportNetwork
from .model import Metric

DEFAULT_SIZES = [25, 30, 100, 250, 500, 1000, 2000, 4000, 8000]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pairs(network: TransportNetwork, count: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    ids = list(network.node_ids())
    out = []
    while len(out) < count:
        s, t = rng.choice(ids), rng.choice(ids)
        if s != t:
            out.append((s, t))
    return out


def _time_query(fn: Callable[[], object], repeats: int) -> tuple[float, object]:
    """Return (mean seconds per call, last result)."""
    fn()  # warm-up, discarded
    gc.disable()
    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = fn()
    elapsed = time.perf_counter() - start
    gc.enable()
    return elapsed / repeats, result


def _write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# 1. Routing algorithms vs network size
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ScalingRow:
    nodes: int
    edges: int
    algorithm: str
    mean_ms: float
    mean_nodes_expanded: float


def algorithm_scaling(
    sizes: list[int] | None = None,
    queries: int = 25,
    seed: int = 11,
    bellman_ford_max: int = 1000,
) -> list[ScalingRow]:
    """Mean per-query time for each algorithm across increasing network sizes."""
    sizes = sizes or DEFAULT_SIZES
    rows: list[ScalingRow] = []

    for n in sizes:
        net = generate_synthetic_network(n, avg_degree=4.0, seed=seed + n)
        pairs = _pairs(net, queries, seed)

        candidates: list[tuple[str, Callable]] = [
            ("dijkstra", lambda s, t, g=net: dijkstra(g, s, t, Metric.time)),
            ("a_star", lambda s, t, g=net: a_star(g, s, t, Metric.time)),
            ("bfs_min_hops", lambda s, t, g=net: bfs_min_hops(g, s, t)),
        ]
        if n <= bellman_ford_max:
            candidates.append(
                ("bellman_ford", lambda s, t, g=net: bellman_ford(g, s, t, Metric.time))
            )

        for label, fn in candidates:
            fn(*pairs[0])  # warm-up
            gc.disable()
            start = time.perf_counter()
            expanded = 0
            for s, t in pairs:
                res = fn(s, t)
                expanded += res.nodes_expanded
            elapsed = time.perf_counter() - start
            gc.enable()
            rows.append(
                ScalingRow(
                    nodes=net.node_count,
                    edges=net.edge_count,
                    algorithm=label,
                    mean_ms=1000.0 * elapsed / len(pairs),
                    mean_nodes_expanded=expanded / len(pairs),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# 2. Representation comparison: adjacency list vs adjacency matrix
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RepresentationRow:
    nodes: int
    edges: int
    list_build_ms: float
    matrix_build_ms: float
    list_peak_kb: float
    matrix_peak_kb: float
    list_scan_ms: float
    matrix_scan_ms: float
    matrix_cells: int


def representation_comparison(
    sizes: list[int] | None = None, seed: int = 5
) -> list[RepresentationRow]:
    """Measure the O(V+E) vs O(V^2) claim directly.

    ``*_scan_ms`` times a full neighbour sweep of every node -- the operation
    Dijkstra's inner loop performs -- which is where the matrix loses.
    """
    sizes = sizes or [25, 30, 100, 250, 500, 1000, 2000]
    rows: list[RepresentationRow] = []

    for n in sizes:
        tracemalloc.start()
        t0 = time.perf_counter()
        net = generate_synthetic_network(n, avg_degree=4.0, seed=seed + n)
        list_build = (time.perf_counter() - t0) * 1000
        _, list_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        t0 = time.perf_counter()
        mat = MatrixNetwork(net, Metric.time)
        matrix_build = (time.perf_counter() - t0) * 1000
        _, matrix_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        t0 = time.perf_counter()
        total = 0
        for nid in net.node_ids():
            for _ in net.neighbours(nid):
                total += 1
        list_scan = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        total_m = 0
        for i in range(mat.size):
            for _ in mat.neighbours(i):
                total_m += 1
        matrix_scan = (time.perf_counter() - t0) * 1000

        assert total == total_m == net.edge_count, "representations disagree"

        rows.append(
            RepresentationRow(
                nodes=net.node_count, edges=net.edge_count,
                list_build_ms=list_build, matrix_build_ms=matrix_build,
                list_peak_kb=list_peak / 1024, matrix_peak_kb=matrix_peak / 1024,
                list_scan_ms=list_scan, matrix_scan_ms=matrix_scan,
                matrix_cells=mat.cell_count(),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 3. Reachability cost vs budget
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReachabilityRow:
    nodes: int
    budget_min: float
    reached: int
    coverage_pct: float
    mean_ms: float


def reachability_scaling(
    size: int = 4000,
    budgets: list[float] | None = None,
    repeats: int = 5,
    seed: int = 21,
) -> list[ReachabilityRow]:
    """Show that a bounded scan costs in proportion to what it reaches.

    Budgets are derived from the network itself rather than hard-coded: a
    fixed "10 minutes" means something completely different on a 25-node
    neighbourhood and a 4,000-node city.  We take the maximum travel time from
    the source over the whole network and sample fractions of it, so the same
    experiment is meaningful at any size.
    """
    net = generate_synthetic_network(size, avg_degree=4.0, seed=seed)
    source = list(net.node_ids())[size // 2]

    if budgets is None:
        full, _ = dijkstra_costs(net, source, Metric.time)
        horizon = max(full.values())
        budgets = [round(horizon * f, 1)
                   for f in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)]
        budgets.append(float("inf"))

    rows: list[ReachabilityRow] = []
    for b in budgets:
        mean_s, result = _time_query(
            lambda b=b: reachable_within(net, source, b, Metric.time), repeats
        )
        rows.append(
            ReachabilityRow(
                nodes=net.node_count, budget_min=b, reached=result.reached,
                coverage_pct=100.0 * result.coverage(net), mean_ms=mean_s * 1000,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 4. Bottleneck detection: Tarjan vs brute force
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BottleneckRow:
    nodes: int
    edges: int
    tarjan_ms: float
    brute_force_ms: float | None
    speedup: float | None
    articulation_points: int
    bridges: int


def bottleneck_scaling(
    sizes: list[int] | None = None, brute_force_max: int = 1000, seed: int = 31
) -> list[BottleneckRow]:
    sizes = sizes or [25, 30, 100, 250, 500, 1000, 2000, 4000]
    rows: list[BottleneckRow] = []
    for n in sizes:
        net = generate_synthetic_network(n, avg_degree=4.0, seed=seed + n)
        # time Tarjan alone: the fragmentation counts are a reporting extra
        # that costs one extra traversal per articulation point
        t0 = time.perf_counter()
        report = find_bottlenecks(net, with_fragmentation=False)
        tarjan = (time.perf_counter() - t0) * 1000

        brute = speedup = None
        if n <= brute_force_max:
            t0 = time.perf_counter()
            naive = verify_critical_nodes(net)
            brute = (time.perf_counter() - t0) * 1000
            assert naive == report.articulation_points, "Tarjan disagrees with brute force"
            speedup = brute / tarjan if tarjan else None

        rows.append(
            BottleneckRow(
                nodes=net.node_count, edges=net.edge_count, tarjan_ms=tarjan,
                brute_force_ms=brute, speedup=speedup,
                articulation_points=len(report.articulation_points),
                bridges=len(report.bridges),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 5. Constrained vs unconstrained routing cost
# ---------------------------------------------------------------------------


def constraint_overhead(
    size: int = 4000, queries: int = 40, seed: int = 41
) -> list[list]:
    """Does filtering inside the loop cost anything?  Rows for the report."""
    net = generate_synthetic_network(size, avg_degree=4.0, seed=seed)
    pairs = _pairs(net, queries, seed)
    profiles = {
        "unconstrained": RouteConstraints(),
        "road vehicle": RouteConstraints.road_vehicle(),
        "avoid congestion >0.6": RouteConstraints.avoiding_congestion(0.6),
        "heavy goods vehicle": RouteConstraints.heavy_goods_vehicle(),
    }
    rows = []
    for label, c in profiles.items():
        dijkstra(net, *pairs[0], Metric.time, c)  # warm-up
        gc.disable()
        t0 = time.perf_counter()
        found = 0
        expanded = 0
        for s, t in pairs:
            r = dijkstra(net, s, t, Metric.time, c)
            found += int(r.found)
            expanded += r.nodes_expanded
        elapsed = time.perf_counter() - t0
        gc.enable()
        rows.append([
            label, size, queries,
            round(1000 * elapsed / queries, 4),
            round(expanded / queries, 1),
            f"{found}/{queries}",
        ])
    return rows


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all(results_dir: Path, sizes: list[int] | None = None, quick: bool = False) -> None:
    """Run every experiment, writing CSVs and SVG charts into ``results_dir``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    sizes = sizes or ([25, 30, 100, 250, 500] if quick else DEFAULT_SIZES)
    rep_sizes = [s for s in sizes if s <= (500 if quick else 2000)]

    # --- 1. algorithms ---------------------------------------------------
    print("  [1/5] routing algorithms vs network size ...")
    scaling = algorithm_scaling(sizes, queries=10 if quick else 25,
                                bellman_ford_max=250 if quick else 1000)
    _write_csv(
        results_dir / "algorithm_scaling.csv",
        ["nodes", "edges", "algorithm", "mean_ms_per_query", "mean_nodes_expanded"],
        [[r.nodes, r.edges, r.algorithm, round(r.mean_ms, 4),
          round(r.mean_nodes_expanded, 1)] for r in scaling],
    )
    by_algo: dict[str, list[ScalingRow]] = {}
    for r in scaling:
        by_algo.setdefault(r.algorithm, []).append(r)
    line_chart(
        results_dir / "chart_algorithm_time.svg",
        [Series(a, [r.nodes for r in rs], [r.mean_ms for r in rs])
         for a, rs in by_algo.items()],
        "Routing query time vs network size",
        "nodes in network", "mean time per query (ms)", log_y=True,
        subtitle="Aurora City synthetic networks, avg degree 4, mean of random "
                 "source-target pairs",
    )
    line_chart(
        results_dir / "chart_nodes_expanded.svg",
        [Series(a, [r.nodes for r in rs], [max(r.mean_nodes_expanded, 0.1) for r in rs])
         for a, rs in by_algo.items()],
        "Search effort: nodes expanded per query",
        "nodes in network", "mean nodes expanded", log_y=True,
        subtitle="Lower is better; A*'s geometric heuristic prunes the frontier",
    )

    # --- 2. representation ------------------------------------------------
    print("  [2/5] adjacency list vs adjacency matrix ...")
    reps = representation_comparison(rep_sizes)
    _write_csv(
        results_dir / "representation_comparison.csv",
        ["nodes", "edges", "list_build_ms", "matrix_build_ms", "list_peak_kb",
         "matrix_peak_kb", "list_full_scan_ms", "matrix_full_scan_ms", "matrix_cells"],
        [[r.nodes, r.edges, round(r.list_build_ms, 3), round(r.matrix_build_ms, 3),
          round(r.list_peak_kb, 1), round(r.matrix_peak_kb, 1),
          round(r.list_scan_ms, 4), round(r.matrix_scan_ms, 4), r.matrix_cells]
         for r in reps],
    )
    line_chart(
        results_dir / "chart_representation_memory.svg",
        [Series("adjacency list O(V+E)", [r.nodes for r in reps],
                [r.list_peak_kb for r in reps]),
         Series("adjacency matrix O(V^2)", [r.nodes for r in reps],
                [r.matrix_peak_kb for r in reps])],
        "Memory: adjacency list vs adjacency matrix",
        "nodes in network", "peak allocation (KB)", log_y=True,
        subtitle="tracemalloc peak while building each representation",
    )
    line_chart(
        results_dir / "chart_representation_scan.svg",
        [Series("adjacency list O(deg)", [r.nodes for r in reps],
                [r.list_scan_ms for r in reps]),
         Series("adjacency matrix O(V)", [r.nodes for r in reps],
                [r.matrix_scan_ms for r in reps])],
        "Full neighbour sweep (Dijkstra's inner loop)",
        "nodes in network", "time for one full sweep (ms)", log_y=True,
        subtitle="Visiting every neighbour of every node, once",
    )

    # --- 3. reachability --------------------------------------------------
    print("  [3/5] reachability vs travel budget ...")
    reach = reachability_scaling(size=min(4000, max(sizes)),
                                 repeats=3 if quick else 5)
    _write_csv(
        results_dir / "reachability_budget.csv",
        ["nodes", "budget_min", "nodes_reached", "coverage_pct", "mean_ms"],
        [[r.nodes, r.budget_min, r.reached, round(r.coverage_pct, 2),
          round(r.mean_ms, 4)] for r in reach],
    )
    finite = [r for r in reach if r.budget_min != float("inf")]
    line_chart(
        results_dir / "chart_reachability.svg",
        [Series("nodes reached", [r.budget_min for r in finite],
                [r.reached for r in finite])],
        "Reachability: nodes within a travel-time budget",
        "budget (minutes)", "nodes reached",
        subtitle=f"Single source, {finite[0].nodes if finite else 0}-node network",
    )
    line_chart(
        results_dir / "chart_reachability_cost.svg",
        [Series("query time", [r.budget_min for r in finite],
                [max(r.mean_ms, 1e-4) for r in finite])],
        "Bounded search costs in proportion to what it reaches",
        "budget (minutes)", "mean query time (ms)",
        subtitle="Pruning at the budget avoids scanning the whole network",
    )

    # --- 4. bottlenecks ---------------------------------------------------
    print("  [4/5] bottleneck detection: Tarjan vs brute force ...")
    bott = bottleneck_scaling([s for s in sizes if s <= 8000],
                              brute_force_max=250 if quick else 1000)
    _write_csv(
        results_dir / "bottleneck_scaling.csv",
        ["nodes", "edges", "tarjan_ms", "brute_force_ms", "speedup",
         "articulation_points", "bridges"],
        [[r.nodes, r.edges, round(r.tarjan_ms, 3),
          round(r.brute_force_ms, 3) if r.brute_force_ms else "",
          round(r.speedup, 1) if r.speedup else "",
          r.articulation_points, r.bridges] for r in bott],
    )
    with_bf = [r for r in bott if r.brute_force_ms is not None]
    bar_chart(
        results_dir / "chart_bottleneck.svg",
        [str(r.nodes) for r in with_bf],
        [("Tarjan O(V+E)", [r.tarjan_ms for r in with_bf]),
         ("brute force O(V(V+E))", [r.brute_force_ms for r in with_bf])],
        "Critical-node detection: one DFS vs remove-and-retest",
        "time (ms)", log_y=True,
        subtitle="Both produce identical results; only the cost differs",
    )

    # --- 5. constraint overhead -------------------------------------------
    print("  [5/5] constrained routing overhead ...")
    crows = constraint_overhead(size=min(4000, max(sizes)),
                                queries=15 if quick else 40)
    _write_csv(
        results_dir / "constraint_overhead.csv",
        ["profile", "nodes", "queries", "mean_ms_per_query",
         "mean_nodes_expanded", "routes_found"],
        crows,
    )
    bar_chart(
        results_dir / "chart_constraint_overhead.svg",
        [r[0] for r in crows],
        [("mean ms per query", [r[3] for r in crows])],
        "Constrained routing costs no more than unconstrained routing",
        "mean time per query (ms)",
        subtitle="Filtering happens inside the relaxation loop at O(1) per edge",
    )
    print(f"  results written to {results_dir}")
