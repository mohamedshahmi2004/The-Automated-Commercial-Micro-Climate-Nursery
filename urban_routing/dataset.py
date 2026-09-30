"""Datasets for the Aurora City routing engine.

Two sources of network data
---------------------------
1. ``build_aurora_city()`` -- a hand-authored 30-node / 78-directed-edge network
   modelled on a mid-sized river city.  It is deliberately *irregular*: it has a
   motorway ring, an old town of narrow residential streets, two river bridges,
   one tunnel, a metro line, and an airport spur hanging off a single road.
   Those features exist so the analysis has something to find -- the spur gives
   a guaranteed cut vertex, the bridges give guaranteed cut edges.

2. ``generate_synthetic_network(n, ...)`` -- a reproducible generator used for
   the fixed 25 / 30 / 100-node study sets and for the scaling benchmark.

Generator model (documented assumptions)
----------------------------------------
* Nodes are laid out on a perturbed grid over a ``span`` x ``span`` km area.
  Real intersections are grid-ish but not exact; the jitter stops every edge
  from having identical length.
* A spanning path is laid first (node i connected to a random earlier node), so
  the network is guaranteed connected -- an unreachable destination would be a
  data artefact, not an algorithmic result.
* Remaining edges connect each node to its nearest unconnected neighbours.
  Real roads join nearby places, so distance-ranked attachment reproduces the
  bounded degree of a road network (avg out-degree stays ~4-6 regardless of n).
* Road class is assigned by edge length: long links become motorway/arterial,
  short ones residential.  Speed limit follows the class.
* Congestion is drawn per edge from a distribution biased towards the city
  centre, since load concentrates on central arterials.
* Everything is driven by ``random.Random(seed)`` so every figure in the report
  is reproducible.

What a node and an edge mean
----------------------------
node  = a junction, interchange or transit stop.  Fields: id, display name,
        x/y position in km from the city origin, zone label.
edge  = one *direction* of travel along a connection.  Fields: source, target,
        length in km, posted speed limit in km/h, road class, live congestion
        (0 = free flow, 1 = gridlock).  Travel time is derived, not stored.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

from .graph import TransportNetwork
from .model import Edge, Node, RoadType

# ---------------------------------------------------------------------------
# 1. Hand-authored baseline network: "Aurora City"
# ---------------------------------------------------------------------------

#: (id, display name, x km, y km, zone)
AURORA_NODES: list[tuple[str, str, float, float, str]] = [
    ("N01", "Central Station",        0.0,  0.0, "core"),
    ("N02", "Market Square",          0.8,  0.4, "core"),
    ("N03", "Cathedral Gate",         1.5,  0.1, "oldtown"),
    ("N04", "Old Town Cross",         2.1,  0.7, "oldtown"),
    ("N05", "Guild Street",           2.4, -0.2, "oldtown"),
    ("N06", "Riverside North",       -0.6,  1.4, "riverside"),
    ("N07", "North Bridge",          -0.4,  2.3, "riverside"),
    ("N08", "Northbank Junction",    -0.2,  3.2, "north"),
    ("N09", "University Campus",      1.3,  3.6, "north"),
    ("N10", "Science Park",           2.6,  4.1, "north"),
    ("N11", "Ring North",             0.2,  5.0, "ring"),
    ("N12", "Ring East",              5.4,  3.0, "ring"),
    ("N13", "Ring South",             4.6, -2.8, "ring"),
    ("N14", "Ring West",             -3.9, -0.6, "ring"),
    ("N15", "West Gate",             -2.4, -0.3, "west"),
    ("N16", "Harbour Road",          -2.8, -1.9, "harbour"),
    ("N17", "Container Terminal",    -3.6, -3.1, "harbour"),
    ("N18", "South Quay",            -1.4, -2.4, "harbour"),
    ("N19", "South Bridge",          -0.3, -2.0, "riverside"),
    ("N20", "Stadium",                1.2, -2.6, "south"),
    ("N21", "Retail Park",            2.9, -2.1, "south"),
    ("N22", "Hospital",               3.3, -0.4, "east"),
    ("N23", "East Interchange",       4.2,  0.9, "east"),
    ("N24", "Tech Quarter",           3.8,  2.2, "east"),
    ("N25", "Airport Spur",           6.4,  4.4, "airport"),
    ("N26", "Aurora Airport",         7.6,  5.1, "airport"),
    ("N27", "Metro Depot",            1.9,  1.6, "core"),
    ("N28", "Civic Centre",           0.4,  1.1, "core"),
    ("N29", "Hilltop Tunnel West",   -1.9,  2.6, "north"),
    ("N30", "Hilltop Tunnel East",    0.6,  2.9, "north"),
]

#: (source, target, km, km/h, road class, congestion, street name, two-way)
AURORA_LINKS: list[tuple[str, str, float, float, RoadType, float, str, bool]] = [
    # --- city core ---------------------------------------------------------
    ("N01", "N02", 0.9, 40, RoadType.ARTERIAL,    0.55, "Station Approach",  True),
    ("N01", "N28", 1.2, 40, RoadType.ARTERIAL,    0.45, "Civic Way",         True),
    ("N01", "N06", 1.6, 50, RoadType.ARTERIAL,    0.30, "Riverside Drive",   True),
    ("N02", "N03", 0.8, 30, RoadType.RESIDENTIAL, 0.40, "Cathedral Lane",    True),
    ("N02", "N27", 1.7, 30, RoadType.RESIDENTIAL, 0.35, "Depot Road",        True),
    ("N02", "N28", 0.84, 30, RoadType.PEDESTRIAN,  0.00, "Market Walk",       True),
    ("N03", "N04", 0.9, 30, RoadType.RESIDENTIAL, 0.25, "Old Town Wynd",     True),
    ("N03", "N05", 0.98, 30, RoadType.RESIDENTIAL, 0.20, "Guild Close",       True),
    ("N04", "N05", 0.98, 20, RoadType.RESIDENTIAL, 0.15, "Weavers Row",       True),
    ("N04", "N27", 0.95, 30, RoadType.RESIDENTIAL, 0.30, "Chapel Street",     True),
    ("N05", "N22", 0.95, 50, RoadType.ARTERIAL,    0.50, "Infirmary Road",    True),
    ("N27", "N28", 1.6, 30, RoadType.RESIDENTIAL, 0.25, "Museum Street",     True),
    ("N28", "N06", 1.08, 40, RoadType.ARTERIAL,    0.30, "North Parade",      True),
    # --- river crossings (the only two road links across the river) --------
    ("N06", "N07", 0.95, 50, RoadType.ARTERIAL,    0.35, "Riverside North",   True),
    ("N07", "N08", 0.95, 50, RoadType.BRIDGE,      0.60, "North Bridge",      True),
    ("N19", "N18", 1.2, 50, RoadType.BRIDGE,      0.50, "South Bridge",      True),
    ("N01", "N19", 2.09, 50, RoadType.ARTERIAL,    0.45, "Southern Approach", True),
    # --- tunnel under the hill --------------------------------------------
    ("N29", "N30", 2.6, 60, RoadType.TUNNEL,      0.20, "Hilltop Tunnel",    True),
    ("N08", "N29", 1.86, 50, RoadType.ARTERIAL,    0.15, "Hill Road West",    True),
    ("N30", "N09", 1.0, 50, RoadType.ARTERIAL,    0.20, "Hill Road East",    True),
    ("N06", "N29", 1.9, 40, RoadType.RESIDENTIAL, 0.10, "Quarry Lane",       True),
    # --- north side --------------------------------------------------------
    ("N08", "N09", 1.6, 50, RoadType.ARTERIAL,    0.30, "Campus Road",       True),
    ("N09", "N10", 1.4, 50, RoadType.ARTERIAL,    0.25, "Innovation Way",    True),
    ("N08", "N11", 1.90, 90, RoadType.MOTORWAY,    0.20, "A1 Ring North",     True),
    ("N11", "N10", 2.65, 90, RoadType.MOTORWAY,    0.20, "A1 Ring NE",        True),
    ("N10", "N12", 3.10, 90, RoadType.MOTORWAY,    0.25, "A1 Ring East",      True),
    ("N12", "N24", 1.8, 70, RoadType.ARTERIAL,    0.35, "Tech Approach",     True),
    ("N12", "N23", 2.50, 90, RoadType.MOTORWAY,    0.30, "A1 Ring SE",        True),
    ("N23", "N13", 3.8, 90, RoadType.MOTORWAY,    0.25, "A1 Ring South",     True),
    ("N13", "N20", 3.51, 70, RoadType.ARTERIAL,    0.30, "Stadium Way",       True),
    ("N13", "N14", 9.05, 90, RoadType.MOTORWAY,    0.20, "A1 Ring SW",        True),
    ("N14", "N15", 1.6, 70, RoadType.ARTERIAL,    0.30, "West Gate Road",    True),
    ("N14", "N11", 7.15, 90, RoadType.MOTORWAY,    0.15, "A1 Ring NW",        True),
    # --- east --------------------------------------------------------------
    ("N22", "N23", 1.6, 60, RoadType.ARTERIAL,    0.45, "Hospital Link",     True),
    ("N23", "N24", 1.4, 60, RoadType.ARTERIAL,    0.35, "Eastfield Road",    True),
    ("N24", "N10", 2.32, 60, RoadType.ARTERIAL,    0.30, "Park Avenue",       True),
    ("N22", "N21", 1.8, 50, RoadType.ARTERIAL,    0.40, "Retail Approach",   True),
    ("N21", "N20", 1.8, 50, RoadType.ARTERIAL,    0.35, "South Circular",    True),
    ("N21", "N13", 1.9, 70, RoadType.ARTERIAL,    0.25, "Ring Slip South",   True),
    # --- south and harbour -------------------------------------------------
    ("N20", "N19", 1.67, 50, RoadType.ARTERIAL,    0.40, "Stadium Approach",  True),
    ("N18", "N16", 1.6, 50, RoadType.ARTERIAL,    0.30, "Harbour Road",      True),
    ("N16", "N17", 1.49, 50, RoadType.ARTERIAL,    0.25, "Terminal Road",     True),
    ("N16", "N15", 1.7, 50, RoadType.ARTERIAL,    0.25, "Dock Street",       True),
    ("N15", "N01", 2.50, 50, RoadType.ARTERIAL,    0.45, "West Approach",     True),
    # --- airport spur: a single road in and out ----------------------------
    ("N12", "N25", 1.9, 90, RoadType.MOTORWAY,    0.20, "Airport Spur",      True),
    ("N25", "N26", 1.4, 90, RoadType.MOTORWAY,    0.15, "Terminal Approach", True),
    # --- transit (rail) links: not usable by road vehicles ------------------
    ("N01", "N27", 2.56, 60, RoadType.TRANSIT,     0.10, "Metro Line 1",      True),
    ("N27", "N09", 2.2, 60, RoadType.TRANSIT,     0.10, "Metro Line 1 N",    True),
    ("N01", "N26", 9.43, 80, RoadType.TRANSIT,     0.10, "Airport Express",   True),
    # --- one genuine one-way street ----------------------------------------
    ("N05", "N02", 1.77, 30, RoadType.RESIDENTIAL, 0.30, "Guild One-Way",     False),
]


def build_aurora_city() -> TransportNetwork:
    """The 30-node hand-authored baseline network."""
    net = TransportNetwork()
    for nid, name, x, y, zone in AURORA_NODES:
        net.add_node(Node(nid, name, x, y, zone))
    for src, dst, km, kmh, rtype, cong, name, two_way in AURORA_LINKS:
        net.add_link(src, dst, km, kmh, rtype, cong, name, two_way)
    return net


# ---------------------------------------------------------------------------
# 2. Synthetic generator
# ---------------------------------------------------------------------------

#: road class -> (speed limit km/h, share of congestion sensitivity)
_CLASS_SPEEDS: dict[RoadType, float] = {
    RoadType.MOTORWAY: 90.0,
    RoadType.ARTERIAL: 60.0,
    RoadType.RESIDENTIAL: 30.0,
    RoadType.BRIDGE: 50.0,
    RoadType.TUNNEL: 60.0,
}


def _classify(length_km: float, rng: random.Random) -> RoadType:
    """Assign a road class from the link's length.

    Assumption: long links between distant junctions are trunk roads; short
    links inside a neighbourhood are residential streets.  A small share of
    links become bridges or tunnels so constrained routing has something to
    exclude.
    """
    roll = rng.random()
    if length_km > 2.2:
        return RoadType.BRIDGE if roll < 0.06 else RoadType.MOTORWAY
    if length_km > 1.0:
        return RoadType.TUNNEL if roll < 0.05 else RoadType.ARTERIAL
    return RoadType.RESIDENTIAL


def generate_synthetic_network(
    n_nodes: int,
    avg_degree: float = 4.0,
    seed: int = 42,
    span: float | None = None,
    name_prefix: str = "S",
    cul_de_sac_fraction: float = 0.04,
) -> TransportNetwork:
    """Generate a connected, road-like network of ``n_nodes`` nodes.

    ``avg_degree`` sets how many nearest neighbours each node proposes a link
    to (``k = round(avg_degree)``).  Because two nodes often propose each other
    and the duplicate is dropped, the *realised* average degree lands roughly
    0.5-0.9 above ``avg_degree``; the exact figure is reported by
    ``TransportNetwork.summary()`` and recorded in the exported datasets.  What
    matters for this project is the property it guarantees: degree stays bounded
    and independent of ``n_nodes``, so |E| grows as O(|V|), not O(|V|^2).  A
    real city behaves the same way -- junctions do not sprout new arms just
    because the suburbs expand -- and that sparsity is exactly why the adjacency
    list keeps winning.

    Construction
    ------------
    1. Lay nodes on a perturbed grid and index them into a spatial hash so
       neighbour candidates can be found in O(1) expected time rather than by
       scanning all n nodes (which would make generation O(n^2)).
    2. Connect each node to its ``k`` nearest neighbours, k chosen to hit the
       requested average degree.  Distance-ranked attachment is what gives the
       result a road-like structure: bounded degree, mostly short links, and
       local clustering.
    3. Reserve a fraction of nodes as cul-de-sacs: dead ends with exactly one
       connection.  Every real residential area has them, and structurally they
       are what create critical infrastructure -- the single link into a
       cul-de-sac is a bridge, and its one neighbour is an articulation point.
       Without them a k-nearest-neighbour graph is so uniformly well-connected
       that it has no bottlenecks at all, and the bottleneck analysis would
       have nothing to find.
    4. Stitch any leftover components together by their closest pair of nodes,
       so the network is guaranteed connected.  An unreachable destination in
       the results must be an effect of a *constraint*, never a data artefact.
    """
    if n_nodes < 2:
        raise ValueError("need at least 2 nodes")
    rng = random.Random(seed)
    net = TransportNetwork()

    # -- 1. layout: perturbed grid over a square whose area grows with n -----
    if span is None:
        span = max(4.0, math.sqrt(n_nodes) * 1.1)
    side = math.ceil(math.sqrt(n_nodes))
    cell = span / side
    zones = ("core", "north", "east", "south", "west")
    for i in range(n_nodes):
        gx, gy = i % side, i // side
        x = gx * cell + rng.uniform(-0.35, 0.35) * cell
        y = gy * cell + rng.uniform(-0.35, 0.35) * cell
        zone = (zones[0]
                if abs(x - span / 2) < span / 5 and abs(y - span / 2) < span / 5
                else zones[1 + (i % 4)])
        net.add_node(Node(f"{name_prefix}{i:05d}", f"Junction {i}", x, y, zone))

    ids = list(net.node_ids())
    nodes = [net.node(i) for i in ids]
    centre = (span / 2, span / 2)

    # -- spatial hash: bucket -> node indices --------------------------------
    bucket_size = max(cell, 1e-6)

    def bucket_of(node: Node) -> tuple[int, int]:
        return (int(node.x // bucket_size), int(node.y // bucket_size))

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, node in enumerate(nodes):
        buckets.setdefault(bucket_of(node), []).append(idx)

    def candidates(idx: int, rings: int = 2) -> list[int]:
        """Node indices in the bucket of ``idx`` and its surrounding rings."""
        bx, by = bucket_of(nodes[idx])
        out: list[int] = []
        for dx in range(-rings, rings + 1):
            for dy in range(-rings, rings + 1):
                out.extend(buckets.get((bx + dx, by + dy), ()))
        return [j for j in out if j != idx]

    def congestion(a: Node, b: Node) -> float:
        """Load is higher near the centre; jittered so no two links match."""
        mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
        d = math.hypot(mx - centre[0], my - centre[1]) / (span / 2 + 1e-9)
        base = max(0.05, 0.75 - 0.55 * min(1.0, d))
        return round(min(0.95, max(0.0, rng.gauss(base, 0.12))), 3)

    def link(i: int, j: int) -> bool:
        a, b = nodes[i], nodes[j]
        if net.edge_between(a.id, b.id) is not None:
            return False
        # round *up* to 3 dp: rounding down could make a link shorter than the
        # straight line between its endpoints, which would break A*'s heuristic
        length = math.ceil(max(0.15, a.distance_to(b)) * 1000) / 1000
        rtype = _classify(length, rng)
        net.add_link(a.id, b.id, length, _CLASS_SPEEDS[rtype], rtype,
                     congestion(a, b), f"Link {i}-{j}", two_way=True)
        return True

    # -- 2. k-nearest-neighbour attachment, applied to every node ------------
    # Each node proposes k links; many are proposed twice (i->j and j->i), so
    # the realised average degree lands near avg_degree for k ~= avg_degree.
    k = max(1, round(avg_degree))
    order = list(range(n_nodes))
    rng.shuffle(order)

    # cul-de-sacs: dead ends with exactly one connection (see step 3 above)
    n_dead = int(n_nodes * max(0.0, min(0.5, cul_de_sac_fraction)))
    cul_de_sacs = set(order[:n_dead]) if n_dead else set()

    def degree_of(i: int) -> int:
        return net.degree(nodes[i].id)

    def may_link(i: int, j: int) -> bool:
        """A cul-de-sac accepts exactly one connection, from either side."""
        if i in cul_de_sacs and degree_of(i) >= 1:
            return False
        if j in cul_de_sacs and degree_of(j) >= 1:
            return False
        return True

    for i in order:
        cand = candidates(i)
        if not cand:
            cand = [j for j in range(max(0, i - 40), min(n_nodes, i + 40)) if j != i]
        limit = 1 if i in cul_de_sacs else k
        nearest = sorted(cand, key=lambda j: nodes[i].distance_to(nodes[j]))
        made = 0
        for j in nearest:
            if made >= limit or degree_of(i) >= limit:
                break
            if may_link(i, j) and link(i, j):
                made += 1

    # -- 3. stitch components so the network is connected --------------------
    _connect_components(net, nodes, ids, link, cul_de_sacs)
    return net


def _component_map(net: TransportNetwork) -> list[list[str]]:
    """Weakly connected components of ``net`` (BFS, O(V + E))."""
    from collections import deque

    seen: set[str] = set()
    comps: list[list[str]] = []
    for start in net.node_ids():
        if start in seen:
            continue
        comp = [start]
        seen.add(start)
        q = deque([start])
        while q:
            u = q.popleft()
            for v in net.undirected_neighbours(u):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
                    comp.append(v)
        comps.append(comp)
    return comps


def _connect_components(net, nodes, ids, link, cul_de_sacs=frozenset()) -> None:
    """Join every component to the largest one by its closest pair of nodes.

    Guarantees a connected network without destroying the geography: the new
    link is the shortest one that could bridge the gap.
    """
    index = {nid: i for i, nid in enumerate(ids)}
    comps = _component_map(net)
    if len(comps) <= 1:
        return
    comps.sort(key=len, reverse=True)
    main = [index[n] for n in comps[0]]
    for comp in comps[1:]:
        members = [index[n] for n in comp]
        best = None
        # never stitch through a cul-de-sac: that would give it a second
        # connection and quietly destroy the dead end
        members_ok = [i for i in members if i not in cul_de_sacs] or members
        main_ok = [j for j in main if j not in cul_de_sacs] or main
        for i in members_ok:
            for j in main_ok:
                d = nodes[i].distance_to(nodes[j])
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is not None:
            link(best[1], best[2])
            main.extend(members)


# ---------------------------------------------------------------------------
# 3. The named study datasets (25 / 30 / 100 nodes)
# ---------------------------------------------------------------------------

#: name -> (node count, target undirected degree, seed, description)
STUDY_SETS: dict[str, tuple[int, float, int, str]] = {
    "small25":  (25,  4.0, 2025, "Baseline set: 25 junctions, dense enough to "
                                 "exceed the 50-edge minimum."),
    "medium30": (30,  4.4, 3030, "Matches the hand-authored Aurora City in size "
                                 "so synthetic and realistic results compare."),
    "large100": (100, 4.6, 1100, "Borough-scale set used for the constrained "
                                 "routing and reachability demonstrations."),
}


def build_study_set(name: str) -> TransportNetwork:
    """Build one of the fixed, reproducible study networks."""
    if name not in STUDY_SETS:
        raise KeyError(f"unknown study set {name!r}; have {sorted(STUDY_SETS)}")
    n, deg, seed, _ = STUDY_SETS[name]
    return generate_synthetic_network(n, avg_degree=deg, seed=seed,
                                      name_prefix=name[0].upper())


def all_study_sets() -> dict[str, TransportNetwork]:
    return {name: build_study_set(name) for name in STUDY_SETS}


# ---------------------------------------------------------------------------
# 4. Export / import
# ---------------------------------------------------------------------------


def export_csv(network: TransportNetwork, directory: Path, stem: str) -> tuple[Path, Path]:
    """Write ``<stem>_nodes.csv`` and ``<stem>_edges.csv``."""
    directory.mkdir(parents=True, exist_ok=True)
    npath = directory / f"{stem}_nodes.csv"
    epath = directory / f"{stem}_edges.csv"

    with npath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "x_km", "y_km", "zone"])
        for node in network.nodes():
            w.writerow([node.id, node.name, f"{node.x:.4f}", f"{node.y:.4f}", node.zone])

    with epath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "source", "target", "distance_km", "speed_kmh", "road_type",
            "congestion", "travel_time_min", "name",
        ])
        for e in network.edges():
            w.writerow([
                e.source, e.target, f"{e.distance_km:.3f}", f"{e.speed_kmh:.0f}",
                e.road_type.value, f"{e.congestion:.3f}",
                f"{e.travel_time_min:.3f}", e.name,
            ])
    return npath, epath


def export_json(network: TransportNetwork, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": network.summary(),
        "nodes": [
            {"id": n.id, "name": n.name, "x_km": n.x, "y_km": n.y, "zone": n.zone}
            for n in network.nodes()
        ],
        "edges": [
            {
                "source": e.source, "target": e.target,
                "distance_km": e.distance_km, "speed_kmh": e.speed_kmh,
                "road_type": e.road_type.value, "congestion": e.congestion,
                "travel_time_min": round(e.travel_time_min, 3), "name": e.name,
            }
            for e in network.edges()
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_csv(nodes_csv: Path, edges_csv: Path) -> TransportNetwork:
    """Read a network back from the exported CSV pair.

    Edges are stored one row per *direction*, so a two-way street appears twice
    and is re-added here as two directed edges.
    """
    net = TransportNetwork()
    with Path(nodes_csv).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            net.add_node(Node(row["id"], row["name"], float(row["x_km"]),
                              float(row["y_km"]), row["zone"]))
    with Path(edges_csv).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            net.add_edge(Edge(
                row["source"], row["target"], float(row["distance_km"]),
                float(row["speed_kmh"]), RoadType(row["road_type"]),
                float(row["congestion"]), row.get("name", ""),
            ))
    return net


def validate_geometry(network: TransportNetwork, tolerance: float = 1e-9) -> list[str]:
    """Check that no connection is shorter than the straight line between its ends.

    This is a physical invariant -- a road cannot beat a straight line -- and it
    is also the precondition for A*'s heuristic being *admissible*.  If a
    dataset declares a 2.5 km link between points 2.56 km apart, the
    straight-line heuristic overestimates the remaining cost and A* can return a
    sub-optimal route.  That bug is silent: A* still returns *a* path.

    Returns a list of human-readable violations; empty means the dataset is
    sound.  The test suite asserts it is empty for every shipped dataset.
    """
    problems: list[str] = []
    for edge in network.edges():
        straight = network.node(edge.source).distance_to(network.node(edge.target))
        if edge.distance_km < straight - tolerance:
            problems.append(
                f"{edge.source}->{edge.target} ({edge.name}): declared "
                f"{edge.distance_km:.3f} km < straight-line {straight:.3f} km"
            )
    return problems
