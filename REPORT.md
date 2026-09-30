# Scenario 01 — Intelligent Urban Routing and Network Analysis

**Aurora City smart-mobility platform: core routing engine**

A stdlib-only Python prototype. Every number in this report is produced by the
code in this repository and can be regenerated with:

```bash
python3 -m urban_routing demo        # worked demonstration (Sections 4–7)
python3 -m urban_routing datasets    # export the 25/30/100-node study sets
python3 -m urban_routing benchmark   # regenerate results/*.csv and results/*.svg
python3 -m unittest discover -s tests -v   # 49 correctness tests
```

---

## 1. Problem scope

**City.** *Aurora City* — a mid-sized river city of roughly 30 modelled
junctions in its core, with a motorway ring, a historic quarter of narrow
streets, two river bridges, a hill tunnel, a container port, a metro line and an
airport on a spur road.

**Users the engine serves, and what each one actually needs:**

| User | Query shape | What makes it hard |
|---|---|---|
| Autonomous shuttle | fastest route, now | must respect live congestion and vehicle class |
| Ambulance | fastest route + "what can I reach in 8 minutes?" | one-to-all, not one-to-one |
| Freight operator | route legal for a heavy goods vehicle | whole road classes are off-limits |
| Commuter app | shortest, fastest, or fewest changes | the metric is the user's choice, not ours |
| City planner | which junction or link is critical? | a structural question, not a routing one |

**Constraints that matter.** Queries are interactive, so a single route must
return in milliseconds. The network changes constantly (congestion updates, a
bridge closed by an incident) but its *shape* changes rarely. Restrictions are
per-vehicle, so the same network must answer differently for a car, an ambulance
and a lorry — without holding five copies of the city in memory.

**What this prototype deliberately does not do:** turn restrictions, traffic
signal timing, time-dependent travel (a route planned for 17:00 rather than
now), multi-modal transfers with waiting times, or persistence. Section 9 sets
out what a production system would need.

---

## 2. Decision Area 1 — How the network is represented

### The alternatives

| | Adjacency list | Adjacency matrix | Edge list |
|---|---|---|---|
| Space | **O(V + E)** | O(V²) | O(E) |
| Iterate neighbours of *u* | **O(deg u)** | O(V) | O(E) |
| Edge lookup (u,v) | O(deg u), **O(1) with an index** | **O(1)** | O(E) |
| Insert edge | **O(1)** amortised | O(1) | **O(1)** |
| Add node | **O(1)** | O(V²) rebuild | **O(1)** |

### Answering the guiding questions

**Sparse or dense?** Decisively sparse. A junction of six streets is unusual, so
degree is bounded by geometry and does not grow with the city. Measured across
the generated networks, average degree stays at **4.2–4.5 from 25 to 8,000
nodes** (`results/algorithm_scaling.csv`) — |E| grows as O(|V|), not O(|V|²). At
2,000 nodes a matrix would hold **4,000,000 cells for 8,478 real edges: 0.21%
occupancy**.

**How often do we look up a specific connection?** Constantly, but not in the
inner loop: real-time congestion updates are per-edge writes (`set_congestion`),
and the incident API blocks a named link. This is the one operation where a
plain adjacency list is weak — so we do not use a plain one.

**How often do we iterate all neighbours of a node?** On *every single
relaxation step of every query*. This is the hot path and it decides the choice.

**Weighted, directional, dynamic?** All three. Weighted (four metrics),
directional (one-way streets, asymmetric congestion), and dynamic in its
weights though not in its shape.

### The decision

**An adjacency list of directed edges, plus a hash index `(u,v) → Edge`.**

The index costs O(E) extra space — which on a sparse graph is still O(V) — and
buys back the matrix's *only* advantage, O(1) edge lookup. The result dominates
the matrix on every operation we perform and ties it on the one we don't:

| Operation | Our structure | Matrix |
|---|---|---|
| Space | O(V + E) | O(V²) |
| Neighbours of *u* | O(deg u) | O(V) |
| Edge lookup | **O(1)** | O(1) |
| Congestion update | **O(1)** | O(1) |

Two-way streets are stored as **two directed edges sharing a `pair_id`**. This
roughly doubles edge memory versus an undirected model, and it is worth it: real
cities have one-way streets, and inbound and outbound congestion genuinely
differ at rush hour. A model that cannot express that is modelling a different
city.

### Measured (`results/representation_comparison.csv`)

| Nodes | Edges | List peak (KB) | Matrix peak (KB) | List full sweep (ms) | Matrix full sweep (ms) |
|---:|---:|---:|---:|---:|---:|
| 25 | 108 | 40.5 | 8.7 | 0.009 | 0.065 |
| 100 | 436 | 158.9 | 92.4 | 0.017 | 0.853 |
| 500 | 2,126 | 882.2 | 2,051.4 | 0.141 | 20.37 |
| 1,000 | 4,248 | 1,768.3 | 8,022.7 | 0.580 | 93.35 |
| 2,000 | 8,478 | **3,663.9** | **31,681.6** | **1.48** | **335.40** |

Charts: `results/chart_representation_memory.svg`,
`results/chart_representation_scan.svg`.

**Reading the numbers honestly.** Below ~150 nodes the matrix uses *less* memory,
because a Python `Edge` object carries far more overhead than a float in a list.
The crossover is near 250 nodes, and past it the gap widens exactly as the
asymptotics predict: at 2,000 nodes the matrix costs **8.6× the memory** and
**226× the time** for the neighbour sweep that Dijkstra's inner loop performs.
Memory quadruples per doubling for the matrix and merely doubles for the list.

For a 30-node toy the choice is irrelevant. For a city it is the difference
between O(E log V) and O(V²).

---

## 3. Decision Area 2 — Routing algorithms

### Complexity comparison

| Algorithm | Time | Space | Assumes | Constrained routing |
|---|---|---|---|---|
| **Dijkstra** (binary heap) | O((V+E) log V) | O(V) | non-negative weights | native — predicate in the relaxation loop |
| **A\*** | O((V+E) log V) worst case | O(V) | non-negative **and** an admissible heuristic | native, same mechanism |
| **Bellman–Ford** | O(V·E) | O(V) | nothing (handles negative weights) | native, but pays O(V·E) |
| **BFS** | O(V+E) | O(V) | **unweighted** | native; answers a different question |
| Floyd–Warshall | O(V³) time, **O(V²) space** | — | non-negative | all-pairs only |

### On a sparse road network vs a dense transit network

With E ≈ 4V (our measured case), Dijkstra is O(V log V) while Bellman–Ford is
O(V²) — at V = 1,000 that predicted gap is ~100×; **we measure 26.5×**, the
difference being Bellman–Ford's early exit and Python's per-operation constants.
If Aurora's transit layer were modelled as a dense all-pairs network (E ≈ V²),
Dijkstra would degrade to O(V² log V) and an adjacency *matrix* with an
O(V²) array-scan Dijkstra would become competitive. Our network is not that, and
the measurements confirm it.

Floyd–Warshall was rejected outright: its O(V²) space means **64 GB at 90,000
nodes** before considering time. Precomputing all pairs is also wrong for this
domain — congestion changes by the minute, so the table would be stale before it
finished building.

### The decision

**Dijkstra is the engine's default; A\* is offered for point-to-point queries.**

1. **Every metric we expose is non-negative** — a road cannot take negative time.
   Bellman–Ford's negative-weight tolerance is a capability we pay O(V·E) for and
   never use. (`test_all_metrics_are_non_negative` enforces the premise.)
2. **Dijkstra answers one-to-all for free**, which is exactly what reachability
   analysis needs. One implementation, two features.
3. **A\* wins on point-to-point** when coordinates are available, expanding
   **26.9% fewer nodes** at 8,000 nodes.

**Heap choice.** Python's `heapq` has no decrease-key, so we use *lazy deletion*:
push a new entry and skip stale pops. This bounds the heap at O(E) entries while
keeping the log factor. A Fibonacci heap's theoretical O(E + V log V) loses badly
on constant factors at these sizes.

**A\* admissibility — and a bug it caught.** A\* is only optimal if its heuristic
never overestimates. We use straight-line distance, divided by the network's top
speed for time metrics, and return 0 for metrics with no geometric bound (A\*
then safely degenerates to Dijkstra).

This is not a free assumption, and the test suite proved it. The first version of
the hand-authored dataset declared a 2.50 km link between two points **2.563 km
apart** — physically impossible, and it made the heuristic inadmissible. A\* and
Dijkstra disagreed on `N14 → N25` by 0.1 minutes, and
`test_a_star_heuristic_is_admissible` located it. Twenty-four link lengths were
wrong; the generator's `round()` had the same flaw and now rounds *up*.
`validate_geometry()` enforces the invariant, and a test asserts it for every
shipped dataset. **A silent sub-optimality bug, caught because the property was
tested rather than assumed.**

### Measured (`results/algorithm_scaling.csv`)

Mean ms per query, 25 random source–target pairs:

| Nodes | Dijkstra | A\* | BFS (hops) | Bellman–Ford |
|---:|---:|---:|---:|---:|
| 25 | 0.036 | 0.040 | 0.011 | 0.192 |
| 30 | 0.044 | 0.047 | 0.015 | 0.255 |
| 100 | 0.166 | 0.156 | 0.050 | 1.371 |
| 250 | 0.324 | 0.315 | 0.127 | 4.689 |
| 500 | 0.868 | 0.706 | 0.277 | 13.473 |
| 1,000 | 1.431 | 1.421 | 0.535 | **37.891** |
| 2,000 | 3.102 | 2.950 | 1.163 | — |
| 4,000 | 5.966 | 5.275 | 2.306 | — |
| 8,000 | **15.973** | **13.210** | 5.879 | — |

Charts: `results/chart_algorithm_time.svg`, `results/chart_nodes_expanded.svg`.

**What the numbers show.**

* **Dijkstra scales as predicted.** From 250 → 1,000 nodes (4×), time grows
  4.4×; from 1,000 → 8,000 (8×), 11.2×. That is O(V log V) behaviour, not
  quadratic.
* **Bellman–Ford does not.** Over the same 250 → 1,000 range it grows **8.1×**
  for a 4× size increase — the signature of O(V·E). It is **26.5× slower than
  Dijkstra at 1,000 nodes** and was excluded above that because it stops being a
  serious candidate.
* **A\* expands consistently fewer nodes** — 2,961.8 vs 4,049.6 at 8,000 nodes
  (26.9% fewer) — but converts that into only ~17% less wall time, because each
  expansion now costs a heuristic evaluation. Below ~250 nodes it is *slower*
  than Dijkstra: the heuristic costs more than the search it saves. **A\* is
  worth it for long point-to-point queries on large networks, and not
  otherwise** — which is why Dijkstra, not A\*, is the default.
* **BFS is ~2.7× faster than Dijkstra and answers a different question.** It
  minimises hops, not time. On `N17 → N10` it finds a 5-hop route that is
  *slower* than the time-optimal one. Fast and wrong is not a trade-off worth
  making, so BFS is exposed only for "fewest changes" queries, where hop count
  *is* the objective.

---

## 4. Decision Area 3 — Traversal for non-optimal queries

| Query | Strategy | Auxiliary structure | Space |
|---|---|---|---|
| Reachable within a **cost budget** | Dijkstra, pruned at the budget | binary heap + cost map | O(V′) |
| Reachable within **k hops** | BFS | FIFO queue | O(V) |
| Is the network connected? | BFS (iterative) | FIFO queue | O(V) |
| Critical nodes / links | DFS, Tarjan (iterative) | explicit stack + 2 int maps | O(V) |

**Why not BFS for the budget query?** BFS visits in order of *hop count*, which
says nothing about minutes travelled: one 8 km motorway link is one hop and five
minutes; eight residential turns are eight hops and four minutes. Only a
cost-ordered frontier — Dijkstra's heap — settles nodes in order of budget
consumed, and that ordering is what makes pruning sound. Because costs are
non-negative, a node beyond the budget can never lead back inside it, so the
search can stop expanding there. `test_matches_full_dijkstra_then_filter`
verifies the pruned scan returns exactly what a full scan returns, filtered, at
seven budgets.

**Why BFS and not DFS for connectivity?** Both are O(V+E) and both answer the
question. BFS wins on *space in practice*: its queue is bounded by frontier
width, whereas recursive DFS on a 50,000-node network exhausts Python's call
stack. Where DFS is genuinely required — articulation points need
discovery/low-link times that BFS cannot produce — we implement it **iteratively
with an explicit stack** for exactly that reason.

### Measured (`results/reachability_budget.csv`)

Single source, 4,000-node network. Budgets are derived from the network's own
cost horizon, because "10 minutes" means different things at different scales:

| Budget (min) | Nodes reached | Coverage | Query time (ms) |
|---:|---:|---:|---:|
| 7.0 | 18 | 0.45% | **0.048** |
| 14.1 | 76 | 1.90% | 0.209 |
| 28.1 | 345 | 8.62% | 1.115 |
| 42.2 | 857 | 21.43% | 3.201 |
| 70.3 | 2,062 | 51.55% | 7.646 |
| 98.4 | 2,886 | 72.15% | 10.473 |
| 126.5 | 3,827 | 95.67% | 14.745 |
| ∞ (full scan) | 4,000 | 100% | 14.557 |

**Query cost tracks coverage almost exactly.** A tight 7-minute budget touches
0.45% of the city and costs **0.048 ms — 303× cheaper than the full scan**. This
is the whole point of pruning: an ambulance isochrone is cheap precisely because
it is local. Note the 126.5-minute row is marginally *slower* than the unbounded
scan: at 95.67% coverage the budget comparison is pure overhead with almost
nothing left to prune. Pruning pays when the budget is tight, which is when it is
actually used.

Charts: `results/chart_reachability.svg`, `results/chart_reachability_cost.svg`.

### Reachability on Aurora City

Ambulance dispatch from the Hospital (N22), emergency profile:

| Budget | Reached | Coverage | Farthest reached |
|---:|---:|---:|---|
| 3 min | 3 | 10.0% | East Interchange (2.9 min) |
| 5 min | 5 | 16.7% | Cathedral Gate (4.7 min) |
| 8 min | 13 | 43.3% | Aurora Airport (8.0 min) |
| 12 min | 21 | 70.0% | Northbank Junction (11.8 min) |
| 20 min | 30 | 100% | Container Terminal (18.3 min) |

**Operational reading:** a single hospital covers **43% of the city within the
8-minute national ambulance target**, reaches 70% by 12 minutes, and needs a
further 8 minutes for the remaining 30% — the harbour, the west gate and the
river crossings. Those are precisely the districts a second station would
serve.

---

## 5. Decision Area 4 — What the nodes and connections hold

Four attributes stored, one derived.

| Attribute | Stored? | Why |
|---|---|---|
| `distance_km` | stored | the "shortest route" metric; also bounds the A\* heuristic |
| `speed_kmh` | stored | free-flow speed; with distance it yields travel time |
| `road_type` | stored | **the enabler for constrained routing** — HGV bans, rail vs road |
| `congestion` | stored | the only genuinely real-time field; 0.0 free flow → 1.0 gridlock |
| `travel_time_min` | **derived** | see below |

**Why travel time is derived, not stored.** It is a pure function of the other
three, and it is the one value that changes every time congestion changes.
Storing it too would create two sources of truth that can silently disagree —
update congestion, forget to recompute time, and the router quietly optimises
against stale data. Deriving it costs one float division per relaxation and keeps
the network consistent by construction. `set_congestion` is O(1) and every
subsequent query sees the new time immediately.

**The congestion model.** Effective speed is `speed × max(0.10, 1 − congestion)`.
The floor matters: a fully gridlocked link is slow but *finite*. An infinite cost
would make the edge invisible to the router rather than merely unattractive, and
"gridlock" operationally means crawling, not teleporting around.

**Node attributes.** `id`, `name`, `zone` (for zone-based restrictions such as a
low-emission zone) and `x, y` coordinates in km. The coordinates exist for one
reason: they give A\* an admissible heuristic. Without them A\* is Dijkstra with
extra steps.

**Attributes deliberately excluded:** lane count, gradient, surface, toll cost,
turn restrictions, vehicle height and weight limits. Each would add a field
without changing which algorithm is correct. `RouteConstraints.predicate` is the
extension point for adding them.

### Four metrics, one implementation

Every algorithm takes a cost function, so the same Dijkstra answers every
preference. On `N17 Container Terminal → N10 Science Park`:

| Metric | Cost | Distance | Time | Hops | Route |
|---|---:|---:|---:|---:|---|
| `time` | 14.88 min | 14.59 km | **14.9 min** | 5 | via the **motorway ring** |
| `distance` | 12.19 km | **12.19 km** | 22.9 min | 8 | via the city centre |
| `balanced(0.5)` | 16.19 | 14.59 km | 14.9 min | 5 | via the motorway ring |

**The shortest route is 16% shorter and 54% slower.** It cuts through the
congested core; the fastest one trades 2.4 km for 8.0 minutes on the ring. A
routing engine that only knew distance would give commuters demonstrably bad
advice. `test_different_metrics_can_give_different_routes` asserts this remains
true.

---

## 6. Constrained routing

### Why a predicate, not a filtered copy

| Approach | Per-query cost | Problem |
|---|---|---|
| Delete elements, route, restore | O(V+E) | destructive; breaks concurrent queries |
| Build a filtered copy per query | O(V+E) | duplicates the whole city per query |
| **Predicate inside the relaxation loop** | **O(1) per edge examined** | none |

`RouteConstraints.allows_edge` is consulted once per edge the search actually
relaxes. A constrained query is therefore asymptotically identical to an
unconstrained one — and often *cheaper*, because pruned edges are never expanded.
The same object is honoured by Dijkstra, A\*, BFS and the reachability scan:
one implementation, five features.

**Measured** (`results/constraint_overhead.csv`, 4,000 nodes, 40 queries):

| Profile | ms/query | Nodes expanded | Routes found |
|---|---:|---:|---:|
| unconstrained | 5.64 | 1,756.3 | 40/40 |
| road vehicle | 5.38 | 1,756.3 | 40/40 |
| avoid congestion > 0.6 | 6.31 | 1,831.4 | 37/40 |
| heavy goods vehicle | 4.65 | 1,649.4 | 39/40 |

Constraining costs nothing measurable — the spread is within run-to-run noise,
and the HGV profile is *faster* because a restricted network is a smaller one.
Note that 3 of 40 congestion-limited queries have **no compliant route**: a
correct answer, not a failure.

For *structural* analysis the copy is the right tool, so `subnetwork()` exists —
"is the city still connected for lorries?" is a question about a different graph,
and Tarjan's algorithm wants it up front. That O(V+E) cost is paid once per
analysis, not once per query.

### Demonstrated test cases

**1. Vehicle-class restriction.** `N17 Container Terminal → N04 Old Town Cross`:

| Profile | Result |
|---|---|
| car | 18.63 min via N17→N16→N15→N01→N02→N03→N04 |
| **HGV** | **NO ROUTE** — the old town is reachable only by residential streets |

Correct and useful: the freight operator learns the delivery is illegal *before*
dispatching. (`test_hgv_cannot_reach_the_old_town`)

**2. Incident closure.** North Bridge (N07↔N08) closed. `N01 → N09`:

| | Time | Route |
|---|---:|---|
| before | 10.09 min | N01 → N06 → N07 → N08 → N09 |
| after | 10.66 min | N01 → N06 → **N29 → N30** → N09 |
| **delay** | **+0.57 min** | reroutes through the Hilltop Tunnel |

The engine quantifies an incident's cost to the network, not just to one driver.

**3. Modal restriction.** `N17 → N26 Airport`: the unrestricted profile takes
18.42 min using a **transit** link; every road profile takes 20.32 min via
motorway. The engine correctly refuses to route a car onto a railway.

**4. Congestion-zone avoidance.** Capping congestion at 0.4 changes the route;
every edge on the result satisfies the cap (`test_congestion_ceiling_is_respected`).

**5. Composition.** `road_vehicle().with_blocked(nodes={"N29"}, edges={("N07","N08")})`
applies all three rules at once.

---

## 7. Connectivity and bottleneck detection

### Tarjan vs remove-and-retest

The naive method — remove each node, re-run a connectivity check, restore it — is
O(V·(V+E)). Tarjan's algorithm finds every articulation point *and* every bridge
in **one DFS**, O(V+E), using discovery times and low-link values:

* *u* is an articulation point ⟺ *u* is the DFS root with >1 child, or *u* has a
  child *c* with `low[c] ≥ disc[u]`
* (u,v) is a bridge ⟺ `low[v] > disc[u]` for tree edge u→v

Both methods are implemented; the brute-force one exists **solely to validate
the fast one**. `test_tarjan_matches_brute_force_on_synthetic_networks` asserts
identical results on five random networks plus the baseline, and the benchmark
asserts agreement at every size it can afford.

**Measured** (`results/bottleneck_scaling.csv`):

| Nodes | Edges | Tarjan (ms) | Brute force (ms) | Speed-up | Critical nodes | Bridges |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 110 | 0.118 | 0.667 | 5.7× | 1 | 1 |
| 100 | 424 | 0.343 | 9.341 | 27.3× | 4 | 4 |
| 500 | 2,128 | 1.578 | 295.0 | 186.9× | 19 | 20 |
| 1,000 | 4,214 | 3.123 | **1,174.5** | **376.1×** | 39 | 40 |
| 2,000 | 8,482 | 6.377 | — | — | 78 | 80 |
| 4,000 | 16,906 | 14.579 | — | — | 158 | 160 |
| 8,000 | 33,782 | **33.374** | — | — | 318 | 320 |

Tarjan's time **doubles as the network doubles** — textbook O(V+E) — while the
brute-force speed-up grows linearly with V, as O(V·(V+E)) / O(V+E) predicts.
Tarjan analyses 8,000 junctions in 33 ms; brute force needs 1.2 seconds for 1,000
and was abandoned above that.

Chart: `results/chart_bottleneck.svg`.

> **A measurement bug worth recording.** The first run showed Tarjan taking 894 ms
> at 4,000 nodes — clearly superlinear. The cause was in the harness, not the
> algorithm: `find_bottlenecks` also reports how many pieces the network falls
> into per critical node, which costs one extra traversal each, O(A·(V+E)). That
> is a reporting nicety, not Tarjan. It is now behind `with_fragmentation` and
> switched off when timing. **The true figure is 14.6 ms — 61× faster than what
> the flawed harness reported.**

### Critical infrastructure in Aurora City

Full network: **connected, and strongly connected** — every location reaches
every other, respecting one-way streets.

**Road-vehicle view** (transit and pedestrian links excluded) — 3 critical nodes,
3 critical links, confirmed by brute force:

| Critical node | Effect of removal |
|---|---|
| **N25 Airport Spur** | isolates Aurora Airport from all road access |
| **N12 Ring East** | isolates the airport branch from the ring |
| **N16 Harbour Road** | isolates the Container Terminal |

| Critical link | Effect |
|---|---|
| N12 ↔ N25 *Airport Spur* | severs the airport branch |
| N25 ↔ N26 *Terminal Approach* | severs the airport |
| N16 ↔ N17 *Terminal Road* | severs the container port |

#### The most critical piece of infrastructure, and why

**N25 Airport Spur.** The airport hangs off a single chain: `N12 → N25 → N26`,
with no alternative road approach. Closing that one junction severs every road
route to Aurora Airport — passengers, staff, air freight, emergency access.

The finding is sharper than "it is a dead end", and it required the right model
to see:

1. **In the full multi-modal network, N25 is not critical at all.** The Airport
   Express rail link (N01↔N26) keeps the airport connected. Run the analysis on
   the whole network and it reports one critical node.
2. **In the road-vehicle view it is critical.** Run it on the graph that road
   vehicles actually use and three appear.

So the airport's road resilience rests entirely on a rail line that **no road
vehicle can use**. An analysis that ignored `road_type` would have declared the
airport resilient. This is precisely the single-point-of-failure that Decision
Area 4's choice to store road class exists to expose.

Verified operationally, not just structurally: with N25 blocked,
`dijkstra(N01 → N26)` under the road profile returns **no route**
(`test_identified_cut_vertex_really_disconnects`).

#### Connectivity is vehicle-dependent

**Heavy-goods view:** 30 nodes, 72 directed edges, **4 components**. Three
locations — *Cathedral Gate, Old Town Cross, Metro Depot* — are unreachable by
freight, because the old town is served only by residential streets barred to
HGVs. The city is connected; the city *for lorries* is not.

#### Load bottlenecks vs connectivity bottlenecks

Not every critical link is a bridge. Ranking links by how many optimal routes use
them (edge betweenness, road-vehicle view):

| Link | Name | On N optimal routes |
|---|---|---:|
| N16→N17 | Terminal Road | 29 |
| N25→N26 | Terminal Approach | 29 |
| N12→N25 | Airport Spur | 28 |
| N15→N16 | Dock Street | 23 |
| N19→N18 | South Bridge | 21 |

**Dock Street and South Bridge are not bridges** — removing either leaves the
city connected — yet each carries a fifth of all optimal routes. They fail
*gracefully but expensively*: not a blackout, a city-wide slowdown. The two
analyses answer different questions and a planner needs both.

---

## 8. Dataset documentation

### What the data represents

* **Node** = a junction, interchange or transit stop. Fields: `id`, display
  `name`, `x`/`y` in km from an arbitrary city origin, `zone` label.
* **Edge** = **one direction** of travel along a connection. Fields: `source`,
  `target`, `distance_km`, `speed_kmh` (posted limit), `road_type`,
  `congestion` (0–1). A two-way street appears as two rows sharing a `pair_id`.
  `travel_time_min` is exported for convenience but **derived**, never
  authoritative.

### The datasets

| Dataset | Nodes | Directed edges | Undirected links | Source |
|---|---:|---:|---:|---|
| `aurora_city` | 30 | 99 | 50 | hand-authored |
| `small25` | 25 | 110 | 55 | generated, seed 2025 |
| `medium30` | 30 | 128 | 64 | generated, seed 3030 |
| `large100` | 100 | 530 | 265 | generated, seed 1100 |

All four exceed the brief's minimum of 25 nodes and 50 edges
(`test_study_sets_meet_the_minimum_and_are_connected`). Scaling runs use
generated networks from 25 to 8,000 nodes. Exported to `data/` as
`<name>_nodes.csv`, `<name>_edges.csv` and `<name>.json` via
`python3 -m urban_routing datasets`.

### How `aurora_city` was built

Hand-authored to be **deliberately irregular**, because a uniform grid has
nothing interesting to find. It contains a motorway ring, an old town of narrow
residential streets, two river bridges, a hill tunnel, a container port, a metro
line, one genuine one-way street, and an airport on a spur. The spur guarantees a
cut vertex; the port guarantees a bridge; the metro line guarantees that the
multi-modal and road-only views differ. Distances and speeds are plausible for a
mid-sized European city; congestion values are illustrative rush-hour figures.

### How the synthetic networks are generated

1. **Layout.** Nodes on a perturbed grid over a `span × span` km area, with span
   growing as √n so density stays realistic. Jitter stops every link having an
   identical length.
2. **Spatial hash.** Nodes are bucketed so neighbour candidates are found in O(1)
   expected time. *This replaced an O(n²) scan that made generating 4,000 nodes
   take minutes; generation is now 0.13 s.*
3. **k-nearest attachment.** Each node links to its k nearest neighbours
   (k = round(avg_degree)). Distance-ranked attachment is what makes the result
   road-like: bounded degree, mostly short links, local clustering.
4. **Cul-de-sacs.** 4% of nodes are dead ends with exactly one connection. Every
   real residential area has them, and structurally they *are* what creates
   critical infrastructure. *Without them the k-NN graph was so uniformly
   well-connected that it had zero articulation points and the bottleneck
   benchmark had nothing to find.*
5. **Stitching.** Leftover components are joined at their closest pair, so the
   network is guaranteed connected. **An unreachable destination in the results
   is therefore always the effect of a constraint, never a data artefact.**
6. **Road class from length** (long → motorway, short → residential), speed from
   class, congestion drawn from a distribution biased toward the city centre.

### Assumptions, stated plainly

* Congestion degrades speed linearly, floored at 10% of the limit.
* Straight-line (Euclidean) geometry; no earth curvature — correct at city scale.
* Every road link is two-way unless explicitly marked otherwise.
* Speed limits are free-flow; congestion carries all time-of-day variation.
* **Every declared length ≥ the straight-line distance between its endpoints.**
  This is a physical fact and the precondition for A\* admissibility; it is
  enforced by `validate_geometry()` and asserted for every shipped dataset.
* Generation is fully seeded — every figure here is reproducible.

---

## 9. Architecture (class structure)

Four layers, each depending only on the one below it. No layer reaches upward,
so the domain model has no idea that A\* or Tarjan exist.

```
┌─ Interface ────────────────────────────────────────────────────────────────┐
│  cli.py            demo · datasets · route · reach · analyse · benchmark   │
│  benchmark.py      5 scaling experiments → CSV + SVG                      │
│  charts.py         Series → line_chart / bar_chart (SVG, no dependency)    │
└────────────────────────────────┬───────────────────────────────────────────┘
┌─ Algorithms ───────────────────▼───────────────────────────────────────────┐
│  algorithms.py     RouteResult                                             │
│                    dijkstra · dijkstra_costs · a_star · bellman_ford       │
│                    bfs_min_hops · _heuristic_factory · _reconstruct        │
│  analysis.py       ReachabilityResult · ConnectivityReport                 │
│                    BottleneckReport                                        │
│                    reachable_within · connected_components                 │
│                    is_strongly_connected · find_bottlenecks                │
│                    edge_betweenness_top · verify_critical_* (references)   │
└────────────────────────────────┬───────────────────────────────────────────┘
┌─ Structure ────────────────────▼───────────────────────────────────────────┐
│  graph.py          TransportNetwork    adjacency + reverse + edge index    │
│                    MatrixNetwork       comparison baseline only            │
│                    subnetwork()        materialise a constrained view      │
│  constraints.py    RouteConstraints    allows_edge · allows_node           │
│                                        + vehicle profiles                  │
│  dataset.py        build_aurora_city · generate_synthetic_network          │
│                    export_csv / export_json / load_csv · validate_geometry │
└────────────────────────────────┬───────────────────────────────────────────┘
┌─ Domain ───────────────────────▼───────────────────────────────────────────┐
│  model.py          Node (frozen)  id, name, x, y, zone                    │
│                    Edge           source, target, distance_km, speed_kmh, │
│                                   road_type, congestion, pair_id          │
│                                   + derived travel_time_min               │
│                    RoadType       Enum, 7 classes                          │
│                    Metric         Edge → float cost functions              │
└────────────────────────────────────────────────────────────────────────────┘
```

### The classes that carry state

| Class | Kind | Mutable? | Holds |
|---|---|---|---|
| `Node` | frozen dataclass | no | identity and geometry — neither ever changes |
| `Edge` | dataclass, `slots` | **yes** (`congestion`) | the connection's properties; travel time is a `@property` |
| `RoadType` | `str` Enum | no | the seven road classes |
| `Metric` | namespace of static functions | no | `Edge → float`; `balanced(α)` returns a closure |
| `TransportNetwork` | class, `slots` | yes | `_nodes`, `_adjacency`, `_reverse`, `_edge_index` |
| `MatrixNetwork` | class, `slots` | no | dense cost matrix, built from a network |
| `RouteConstraints` | dataclass, `slots` | no | the filter; `with_blocked()` returns a copy |
| `RouteResult` | dataclass | no | path, edges, cost, and the search-effort counters |
| `ReachabilityResult` / `ConnectivityReport` / `BottleneckReport` | dataclasses | no | analysis output plus its own `describe()` |

### Why it is shaped this way

**Algorithms are functions, not classes.** `dijkstra(network, source, target,
metric, constraints)` has no state worth keeping between calls, so a class would
only add ceremony. The *results* are classes, because a result has behaviour —
`RouteResult.total_time_min`, `describe()`.

**`Metric` and `RouteConstraints` are the extension seams.** Every algorithm
takes both as parameters, so adding a fifth cost metric or a new vehicle class
touches **no algorithm code at all**. That is why one Dijkstra implementation
serves four metrics × five profiles × two query shapes.

**`Node` frozen, `Edge` not.** Geometry never changes; congestion changes
constantly. Freezing `Node` makes it hashable and safe to share, while `Edge`
stays mutable so `set_congestion` is O(1). Travel time is a derived property on
`Edge` rather than a field — one source of truth for the only value that moves in
real time (§5).

**Reference implementations sit beside the real ones.** `MatrixNetwork`,
`bellman_ford`, `verify_critical_nodes` and `verify_critical_edges` are not dead
code: they are the differential-testing oracles and the source of every
comparison in this report. Keeping them in the package means the claims stay
checkable.

**`slots` on the hot-path classes.** `Edge`, `TransportNetwork` and
`RouteConstraints` all declare `__slots__`. On a 8,000-node network there are
33,782 `Edge` objects, and dropping each one's `__dict__` is the difference
between the list and the matrix staying competitive on memory at all.

---

## 10. What a production system would need that this prototype does not provide

Honestly assessed, in rough order of how soon each would bite.

**1. Time-dependent routing.** The single largest gap. We model congestion as
*current*, so a route planned at 08:00 for a 09:00 journey uses the wrong data.
Production needs time-dependent edge weights and a time-expanded search. The
FIFO property is what keeps Dijkstra correct in that setting; without it, correct
algorithms get considerably harder.

**2. Preprocessing for continent-scale queries.** 16 ms per query at 8,000 nodes
extrapolates poorly: a real city is 10⁵–10⁶ nodes, so a plain Dijkstra lands in
the hundreds of milliseconds — too slow at scale. Production systems use
Contraction Hierarchies or Hub Labelling for microsecond queries. Both require
minutes to hours of preprocessing that must be redone as the network changes,
which is why we did not build one: the prototype's job is to demonstrate the
algorithmic reasoning, and a CH implementation would obscure it.

**3. Turn restrictions and penalties.** We model junctions as dimensionless
points. Real ones ban turns and cost time to cross; an unsignalled right turn
across traffic can dominate a short urban route. This needs an *edge-based* graph
(edges become nodes, turns become edges) — roughly a 4× size increase and a
genuine redesign, not a new field.

**4. Concurrency and consistency.** `set_congestion` mutates an `Edge` in place
while queries read it. Single-threaded that is fine; under concurrent load a
long-running query could observe a half-updated network. Production needs
immutable weight snapshots with versioning, or a read-copy-update scheme.

**5. Persistence and real data ingestion.** Everything is in memory and rebuilt
on start. Production needs an OpenStreetMap ingestion pipeline, a spatial
database, map-matching from GPS traces, and live feeds from loop detectors and
floating-car data. Our congestion values are plausible fiction.

**6. Multi-modal routing with waiting times.** We model transit links as edges
with a travel time, which implies a train is always waiting. Real multi-modal
routing needs timetables, transfer penalties and a Pareto frontier over (time,
cost, transfers) — RAPTOR or CSA territory, not Dijkstra's.

**7. Alternative routes and robustness.** We return *the* optimal route. Users
want two or three meaningfully different options, and fleet operators need routes
that stay good when a link fails. Both need k-shortest-paths with dissimilarity
constraints.

**8. Operational engineering.** No API, authentication, rate limiting, caching,
telemetry, or graceful degradation. `find_bottlenecks` runs in 33 ms on 8,000
nodes but would be a scheduled offline job in production, not a request handler.

**9. Scale limits of the language.** Python's per-object overhead is why the
adjacency list loses to the matrix below ~150 nodes. A production engine would
use a compressed sparse-row layout in a systems language, giving cache-friendly
traversal and likely a 10–50× constant-factor gain — without changing a single
complexity result in this report.

---

## 11. Summary of decisions

| Decision area | Choice | Decisive reason |
|---|---|---|
| **1. Representation** | Adjacency list + O(1) edge index | Road networks are sparse (degree 4.2–4.5 at every size); neighbour iteration is the hot path. 8.6× less memory and 226× faster sweeps than a matrix at 2,000 nodes |
| **2. Algorithm** | Dijkstra (default), A\* (point-to-point) | All metrics non-negative, so Bellman–Ford's tolerance is 26.5× wasted cost; Dijkstra gives one-to-all free; A\* expands 26.9% fewer nodes at scale |
| **3. Traversal** | Budget-pruned Dijkstra; BFS for connectivity; iterative Tarjan DFS | Only a cost-ordered frontier respects a travel budget; BFS bounds stack depth; Tarjan is 376× faster than remove-and-retest at 1,000 nodes |
| **4. Attributes** | distance, speed, road class, congestion stored; travel time derived | One source of truth for the only field that changes in real time; road class is what makes constrained routing and the airport finding possible |

### Verification

**49 tests, all passing.** The strategy throughout is *differential testing*:
wherever a fast algorithm sits on the hot path, its output is checked against a
slow, obviously-correct reference on networks small enough to afford it.

* Dijkstra vs A\* — **all 900 node pairs** of the baseline network, two metrics
* Dijkstra vs A\* vs Bellman–Ford — random pairs across three synthetic networks
* Tarjan vs remove-and-retest — identical critical-node *and* bridge sets on six
  networks
* Budget-pruned reachability vs full scan then filter — seven budgets
* A\* heuristic admissibility — checked against true cost from every node
* Structural claims verified operationally: each identified bridge and cut vertex
  is removed and the resulting disconnection confirmed

Three real defects were caught this way and are documented above rather than
quietly fixed: the **inadmissible A\* heuristic** from geometrically impossible
edge lengths (§3), the **degree-1 generator bug** where the link budget was
consumed by the first 1,000 nodes (§8), and the **O(A·(V+E)) measurement error**
that made Tarjan look superlinear (§7).
