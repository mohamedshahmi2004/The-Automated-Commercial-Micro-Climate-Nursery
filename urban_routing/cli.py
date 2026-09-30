"""Command-line driver: `python -m urban_routing <command>`.

Commands
--------
demo        run the full worked demonstration on Aurora City
datasets    build and export the 25 / 30 / 100-node study sets (+ Aurora City)
route       a single point-to-point query
reach       reachability from one node within a budget
analyse     connectivity and bottleneck report
benchmark   run the performance experiments and write CSV + SVG charts
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .algorithms import a_star, bellman_ford, bfs_min_hops, dijkstra
from .analysis import (
    connected_components,
    edge_betweenness_top,
    find_bottlenecks,
    is_strongly_connected,
    isochrone_bands,
    reachable_within,
    verify_critical_edges,
    verify_critical_nodes,
)
from .constraints import RouteConstraints
from .dataset import (
    STUDY_SETS,
    build_aurora_city,
    build_study_set,
    export_csv,
    export_json,
)
from .graph import subnetwork
from .model import Metric, RoadType

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

METRICS = {
    "time": Metric.time,
    "distance": Metric.distance,
    "free_flow": Metric.free_flow_time,
    "balanced": Metric.balanced(0.5),
}

PROFILES = {
    "any": RouteConstraints.unrestricted,
    "car": RouteConstraints.road_vehicle,
    "hgv": RouteConstraints.heavy_goods_vehicle,
    "emergency": RouteConstraints.emergency,
    "clear": lambda: RouteConstraints.avoiding_congestion(0.6),
}


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _net(name: str):
    if name == "aurora":
        return build_aurora_city()
    return build_study_set(name)


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


def cmd_demo(args) -> None:
    net = build_aurora_city()

    _rule("1. NETWORK REPRESENTATION")
    print(f"Aurora City: {net.summary()}")
    print(
        "Stored as an adjacency list (dict of out-edge lists) plus an O(1) "
        "(u,v) edge index.\nA dense matrix for the same network would hold "
        f"{net.node_count ** 2:,} cells for {net.edge_count} real edges "
        f"({100 * net.edge_count / net.node_count ** 2:.1f}% occupied)."
    )
    print("\nSample connections:")
    for e in list(net.edges())[:6]:
        print(
            f"  {e.source}->{e.target:<4} {e.name:<20} {e.distance_km:>5.2f} km  "
            f"{e.speed_kmh:>3.0f} km/h  {e.road_type.value:<12} "
            f"congestion {e.congestion:.2f}  =>  {e.travel_time_min:5.2f} min"
        )

    _rule("2. OPTIMAL PATH FINDING -- same query, four metrics")
    src, dst = "N17", "N10"          # Container Terminal -> Science Park
    print(f"Query: {net.node(src).name} -> {net.node(dst).name}\n")
    for label, metric in METRICS.items():
        res = dijkstra(net, src, dst, metric, RouteConstraints.road_vehicle())
        print(f"  metric={label}")
        print("  " + res.describe(net).replace("\n", "\n  "))
    print(
        "\nThe fastest route and the shortest route differ: the time-optimal "
        "path\ndetours onto the motorway ring, trading kilometres for minutes."
    )

    _rule("3. ALGORITHM COMPARISON -- identical answers, different effort")
    car = RouteConstraints.road_vehicle()
    for fn, label in (
        (lambda: dijkstra(net, src, dst, Metric.time, car), "Dijkstra"),
        (lambda: a_star(net, src, dst, Metric.time, car), "A*"),
        (lambda: bellman_ford(net, src, dst, Metric.time, car), "Bellman-Ford"),
    ):
        r = fn()
        print(f"  {label:<14} cost={r.cost:7.3f} min   "
              f"expanded={r.nodes_expanded:<4} relaxed={r.edges_relaxed}")
    hops = bfs_min_hops(net, src, dst, car)
    print(f"  {'BFS (min hops)':<14} hops={int(hops.cost):<7} "
          f"expanded={hops.nodes_expanded:<4} "
          f"(travel time {hops.total_time_min:.2f} min -- not optimal)")
    print(
        "\nDijkstra, A* and Bellman-Ford agree on cost, as they must. "
        "BFS does not:\nfewest connections is a different objective from "
        "least travel time."
    )

    _rule("4. CONSTRAINED ROUTING")
    print(f"Query: {net.node('N17').name} -> {net.node('N26').name} "
          "(Container Terminal -> Airport)\n")
    for label, factory in PROFILES.items():
        c = factory()
        r = dijkstra(net, "N17", "N26", Metric.time, c)
        if r.found:
            types = {e.road_type.value for e in r.edges}
            print(f"  profile={label:<10} {r.total_time_min:6.2f} min, "
                  f"{r.total_distance_km:5.2f} km, {r.hops} hops, "
                  f"uses {{{', '.join(sorted(types))}}}")
        else:
            print(f"  profile={label:<10} NO ROUTE -- restrictions disconnect the pair")

    print(f"\n  Restriction test: {net.node('N17').name} -> "
          f"{net.node('N04').name} (freight into the old town)\n")
    for label in ("car", "hgv"):
        r = dijkstra(net, "N17", "N04", Metric.time, PROFILES[label]())
        if r.found:
            print(f"    profile={label:<5} {r.total_time_min:6.2f} min via "
                  f"{' -> '.join(r.path)}")
        else:
            print(f"    profile={label:<5} NO ROUTE -- the old town is reachable "
                  "only by residential\n                  streets, which are "
                  "closed to heavy goods vehicles.")

    print("\n  Incident test: North Bridge (N07<->N08) closed by an accident.")
    base = dijkstra(net, "N01", "N09", Metric.time, RouteConstraints.road_vehicle())
    blocked = RouteConstraints.road_vehicle().with_blocked(
        edges={("N07", "N08"), ("N08", "N07")}
    )
    after = dijkstra(net, "N01", "N09", Metric.time, blocked)
    print(f"    before closure: {base.total_time_min:6.2f} min via "
          f"{' -> '.join(base.path)}")
    print(f"    after  closure: {after.total_time_min:6.2f} min via "
          f"{' -> '.join(after.path)}")
    print(f"    delay imposed : {after.total_time_min - base.total_time_min:+.2f} min "
          "(traffic reroutes through the Hilltop Tunnel)")

    print("\n  Zone test: avoid every link above 50% congestion.")
    r = dijkstra(net, "N17", "N10", Metric.time,
                 RouteConstraints.avoiding_congestion(0.5))
    print(f"    {r.describe(net)}" if r.found else "    no compliant route exists")

    _rule("5. REACHABILITY ANALYSIS (isochrones)")
    depot = "N22"   # Hospital -- ambulance dispatch point
    print(f"Source: {net.node(depot).name} (emergency dispatch)\n")
    print(f"  {'budget':>8}  {'reached':>7}  {'coverage':>8}   farthest node reached")
    for budget in (3, 5, 8, 12, 20, 40):
        res = reachable_within(net, depot, budget, Metric.time,
                               RouteConstraints.emergency())
        far = res.farthest(1)
        far_txt = (f"{net.node(far[0][0]).name} ({far[0][1]:.1f} min)"
                   if far else "-")
        print(f"  {budget:>6} min  {res.reached:>7}  "
              f"{100 * res.coverage(net):>7.1f}%   {far_txt}")
    full = reachable_within(net, depot, 12, Metric.time, RouteConstraints.emergency())
    unreachable = set(net.node_ids()) - set(full.costs)
    print(f"\n  Outside the 12-minute envelope: "
          f"{', '.join(sorted(net.node(n).name for n in unreachable)) or 'none'}")
    print("  These are the locations a second ambulance station would cover.")

    _rule("6. CONNECTIVITY AND BOTTLENECK DETECTION")
    conn = connected_components(net)
    print(f"  Full network      : {conn.describe()}")
    print(f"  Strongly connected: {is_strongly_connected(net)} "
          "(every location reaches every other, respecting one-way streets)")

    road = subnetwork(net, RouteConstraints.road_vehicle())
    print(f"\n  Road-vehicle view : {road.summary()}")
    report = find_bottlenecks(road)
    print("  " + report.describe(road).replace("\n", "\n  "))

    print("\n  Cross-check against brute-force removal "
          "(remove each node, recount components):")
    print(f"    Tarjan      : {report.articulation_points}")
    print(f"    Brute force : {verify_critical_nodes(road)}")
    print(f"    match       : {report.articulation_points == verify_critical_nodes(road)}")

    print("\n  Why these are critical:")
    for nid in report.articulation_points:
        frag = report.fragmentation[nid]
        print(f"    - {nid} {net.node(nid).name}: removing it breaks the road "
              f"network into {frag} pieces.")
    print("    The airport spur (N12-N25-N26) is a single chain: one closure "
          "isolates\n    the airport from every road approach. Only the "
          "Airport Express rail link\n    keeps the airport connected in the "
          "full multi-modal network -- which is\n    exactly the kind of "
          "single-point-of-failure this analysis exists to find.")

    hgv = subnetwork(net, RouteConstraints.heavy_goods_vehicle())
    hgv_conn = connected_components(hgv)
    print(f"\n  Heavy-goods view  : {hgv.summary()}")
    print(f"    {hgv_conn.describe()}")
    stranded = [c[0] for c in hgv_conn.components if len(c) == 1]
    if stranded:
        print("    Unreachable by HGV: "
              + ", ".join(f"{n} {net.node(n).name}" for n in stranded))
        print("    Freight cannot legally reach these via the modelled network.")

    print("\n  Busiest connections by shortest-path usage (edge betweenness):")
    for (u, v), count in edge_betweenness_top(road, Metric.time, top=5):
        e = road.edge_between(u, v)
        print(f"    {u}->{v}  {e.name:<22} on {count} optimal routes")
    print("    A link can be heavily loaded without being a bridge -- these are "
          "the\n    congestion bottlenecks as opposed to the connectivity ones.")

    _rule("7. NEXT STEPS")
    print("Run `python -m urban_routing datasets`  to export the 25/30/100-node sets.")
    print("Run `python -m urban_routing benchmark` to reproduce the performance charts.")
    print("See REPORT.md for the full design justification and results.\n")


# ---------------------------------------------------------------------------
# other commands
# ---------------------------------------------------------------------------


def cmd_datasets(args) -> None:
    out = Path(args.out or DATA)
    _rule("DATASET EXPORT")
    networks = {"aurora_city": build_aurora_city()}
    for name in STUDY_SETS:
        networks[name] = build_study_set(name)

    print(f"{'dataset':<14} {'nodes':>6} {'directed edges':>15} "
          f"{'undirected links':>17}  description")
    for name, net in networks.items():
        desc = (STUDY_SETS[name][3] if name in STUDY_SETS
                else "Hand-authored realistic baseline network.")
        links = len({(min(e.source, e.target), max(e.source, e.target))
                     for e in net.edges()})
        print(f"{name:<14} {net.node_count:>6} {net.edge_count:>15} "
              f"{links:>17}  {desc}")
        export_csv(net, out, name)
        export_json(net, out / f"{name}.json")
    print(f"\nWritten to {out}/ as <name>_nodes.csv, <name>_edges.csv and <name>.json")
    print("Every set meets the brief's minimum of 25 nodes and 50 edges.")


def cmd_route(args) -> None:
    net = _net(args.network)
    metric = METRICS[args.metric]
    constraints = PROFILES[args.profile]()
    if args.block:
        constraints = constraints.with_blocked(nodes=set(args.block))
    algo = {"dijkstra": dijkstra, "a_star": a_star, "bellman_ford": bellman_ford}[args.algorithm]
    res = algo(net, args.source, args.target, metric, constraints)
    print(res.describe(net))


def cmd_reach(args) -> None:
    net = _net(args.network)
    res = reachable_within(net, args.source, args.budget, METRICS[args.metric],
                           PROFILES[args.profile]())
    print(f"From {args.source} ({net.node(args.source).name}) within "
          f"{args.budget} ({args.metric}): {res.reached}/{net.node_count} nodes "
          f"= {100 * res.coverage(net):.1f}% coverage")
    bands = isochrone_bands(res, [args.budget * f for f in (0.25, 0.5, 0.75, 1.0)])
    for b, count in bands.items():
        print(f"  <= {b:8.2f} : {count:>4} nodes")
    for nid in res.within(args.budget)[: args.limit]:
        print(f"    {nid:<8} {net.node(nid).name:<24} {res.costs[nid]:7.2f}")


def cmd_analyse(args) -> None:
    net = _net(args.network)
    if args.profile != "any":
        net = subnetwork(net, PROFILES[args.profile]())
    _rule(f"CONNECTIVITY AND BOTTLENECKS -- {args.network} (profile={args.profile})")
    print(net.summary())
    print(connected_components(net).describe())
    print(f"Strongly connected: {is_strongly_connected(net)}")
    report = find_bottlenecks(net)
    print(report.describe(net, limit=args.limit))
    if args.verify:
        print(f"\nBrute-force cross-check nodes match: "
              f"{report.articulation_points == verify_critical_nodes(net)}")
        print(f"Brute-force cross-check edges match: "
              f"{sorted(tuple(sorted(p)) for p in report.bridges) == verify_critical_edges(net)}")


def cmd_benchmark(args) -> None:
    from .benchmark import run_all

    _rule("PERFORMANCE EVALUATION")
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else None
    run_all(Path(args.out or RESULTS), sizes=sizes, quick=args.quick)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m urban_routing",
        description="Aurora City intelligent urban routing engine",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="full worked demonstration").set_defaults(func=cmd_demo)

    d = sub.add_parser("datasets", help="build and export the study datasets")
    d.add_argument("--out", help="output directory (default: data/)")
    d.set_defaults(func=cmd_datasets)

    r = sub.add_parser("route", help="single routing query")
    r.add_argument("source")
    r.add_argument("target")
    r.add_argument("--network", default="aurora",
                   choices=["aurora", *STUDY_SETS])
    r.add_argument("--metric", default="time", choices=list(METRICS))
    r.add_argument("--profile", default="car", choices=list(PROFILES))
    r.add_argument("--algorithm", default="dijkstra",
                   choices=["dijkstra", "a_star", "bellman_ford"])
    r.add_argument("--block", nargs="*", default=[], help="node ids to avoid")
    r.set_defaults(func=cmd_route)

    rc = sub.add_parser("reach", help="reachability within a budget")
    rc.add_argument("source")
    rc.add_argument("budget", type=float)
    rc.add_argument("--network", default="aurora", choices=["aurora", *STUDY_SETS])
    rc.add_argument("--metric", default="time", choices=list(METRICS))
    rc.add_argument("--profile", default="car", choices=list(PROFILES))
    rc.add_argument("--limit", type=int, default=15)
    rc.set_defaults(func=cmd_reach)

    a = sub.add_parser("analyse", help="connectivity and bottleneck report")
    a.add_argument("--network", default="aurora", choices=["aurora", *STUDY_SETS])
    a.add_argument("--profile", default="car", choices=list(PROFILES))
    a.add_argument("--limit", type=int, default=10)
    a.add_argument("--verify", action="store_true",
                   help="cross-check against the brute-force method")
    a.set_defaults(func=cmd_analyse)

    b = sub.add_parser("benchmark", help="run performance experiments")
    b.add_argument("--out", help="output directory (default: results/)")
    b.add_argument("--sizes", help="comma-separated node counts")
    b.add_argument("--quick", action="store_true", help="smaller, faster sweep")
    b.set_defaults(func=cmd_benchmark)

    args = p.parse_args(argv)
    args.func(args)
    return 0
