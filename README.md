# Aurora City — Intelligent Urban Routing and Network Analysis

Core routing engine for a smart-mobility platform: network representation,
optimal path finding, constrained routing, reachability analysis, connectivity
and bottleneck detection, with an empirical performance harness.

**Pure Python 3.11+ standard library — no dependencies to install.**

> **[REPORT.md](REPORT.md) is the main deliverable**: problem scope, the four
> design decisions with complexity tables and measured justification, worked
> test cases, critical-infrastructure findings and a production-readiness
> assessment.

### Presentation plan coverage

The brief's presentation plan maps onto this repository as follows:

| Slides required | Content | Where it comes from |
|---|---|---|
| Problem, scope, assumptions (1) | scope, users, boundary, assumptions | REPORT.md §1, §8 |
| Dataset justification (1) | 4 datasets, generation method, assumptions | REPORT.md §8 |
| Requirements, candidate algorithms, selection, data structures (5) | one slide each: representation, optimal paths, constrained routing, reachability, connectivity/bottlenecks | REPORT.md §2–§7 |
| Testing: performance, stability, complexity (1) | scaling tables, 49 differential tests, defects found | REPORT.md §3, §10 (Verification) |
| Limitations and future modifications (1) | 9 gaps to a shippable product | REPORT.md §10 |
| Architecture (class structure) | 4 layers, the classes that carry state, why | REPORT.md §9 |

---

## Quick start

```bash
python3 -m urban_routing demo                  # full worked demonstration
python3 -m urban_routing datasets              # export the 25/30/100-node sets
python3 -m urban_routing benchmark             # regenerate results/*.csv + *.svg
python3 -m unittest discover -s tests -v       # 49 correctness tests
```

Individual queries:

```bash
# fastest route for a car, Container Terminal -> Science Park
python3 -m urban_routing route N17 N10 --metric time --profile car

# same pair, shortest distance, using A*
python3 -m urban_routing route N17 N10 --metric distance --algorithm a_star

# freight into the old town: correctly finds no legal route
python3 -m urban_routing route N17 N04 --profile hgv

# route around a closed junction
python3 -m urban_routing route N01 N09 --block N08

# everything an ambulance reaches within 8 minutes
python3 -m urban_routing reach N22 8 --profile emergency

# critical infrastructure in the road-vehicle network, cross-checked
python3 -m urban_routing analyse --profile car --verify
python3 -m urban_routing analyse --network large100 --profile car
```

## Layout

```
urban_routing/
  model.py        Node, Edge, RoadType, cost metrics   (Decision Area 4)
  graph.py        TransportNetwork (adjacency list), MatrixNetwork, subnetwork
                                                       (Decision Area 1)
  algorithms.py   dijkstra, a_star, bellman_ford, bfs_min_hops
                                                       (Decision Area 2)
  analysis.py     reachability, connectivity, Tarjan bottlenecks, betweenness
                                                       (Decision Area 3)
  constraints.py  RouteConstraints + vehicle profiles
  dataset.py      Aurora City + synthetic generator + CSV/JSON export
  benchmark.py    the five scaling experiments
  charts.py       dependency-free SVG chart writer
  cli.py          command-line interface
tests/            49 tests, mostly differential against reference implementations
data/             exported datasets (CSV + JSON)
results/          benchmark CSVs and SVG charts
REPORT.md         design decisions, analysis and results
```

## The engine at a glance

| Capability | Implementation | Complexity |
|---|---|---|
| Network representation | adjacency list + O(1) `(u,v)` edge index | O(V+E) space |
| Optimal path | Dijkstra, binary heap with lazy deletion | O((V+E) log V) |
| Goal-directed path | A\*, admissible straight-line heuristic | ≤ Dijkstra in practice |
| Constrained routing | predicate applied in the relaxation loop | O(1) per edge |
| Reachability | budget-pruned Dijkstra | O((V′+E′) log V′) |
| Connectivity | iterative BFS; Kosaraju for strong connectivity | O(V+E) |
| Critical nodes / links | iterative Tarjan (articulation points + bridges) | O(V+E) |

## Headline results

| Measurement | Result |
|---|---|
| Dijkstra, 8,000-node network | 16 ms per query |
| Bellman–Ford vs Dijkstra at 1,000 nodes | **26.5× slower** |
| A\* search effort at 8,000 nodes | **26.9% fewer** nodes expanded |
| Adjacency matrix vs list at 2,000 nodes | **8.6× the memory, 226× the sweep time** |
| Tarjan vs remove-and-retest at 1,000 nodes | **376× faster** |
| 7-minute isochrone vs full network scan | **303× cheaper** |
| Constrained vs unconstrained routing | no measurable overhead |

Full tables, charts and interpretation in [REPORT.md](REPORT.md).

## Datasets

| Dataset | Nodes | Directed edges | Undirected links |
|---|---:|---:|---:|
| `aurora_city` | 30 | 99 | 50 |
| `small25` | 25 | 110 | 55 |
| `medium30` | 30 | 128 | 64 |
| `large100` | 100 | 530 | 265 |

All exceed the brief's 25-node / 50-edge minimum; scaling runs go to 8,000
nodes. Synthetic generation is fully seeded and reproducible. See REPORT.md §8
for what each field means and how the data was produced.
