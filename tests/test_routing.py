"""Correctness tests for the Aurora City routing engine.

Run with:  python -m unittest discover -s tests -v

The strategy throughout is *differential testing*: wherever a fast algorithm is
used on the hot path, its result is checked against a slow, obviously-correct
reference implementation on networks small enough to make that affordable.

    dijkstra / a_star   vs  bellman_ford      (all agree, or all fail)
    Tarjan bottlenecks  vs  remove-and-retest (identical sets)
    bounded reachability vs full Dijkstra then filter
"""

from __future__ import annotations

import random
import unittest

from urban_routing.algorithms import (
    a_star,
    bellman_ford,
    bfs_min_hops,
    dijkstra,
    dijkstra_costs,
)
from urban_routing.analysis import (
    connected_components,
    find_bottlenecks,
    is_strongly_connected,
    reachable_within,
    reachable_within_hops,
    verify_critical_edges,
    verify_critical_nodes,
)
from urban_routing.constraints import RouteConstraints
from urban_routing.dataset import (
    STUDY_SETS,
    build_aurora_city,
    build_study_set,
    generate_synthetic_network,
)
from urban_routing.graph import MatrixNetwork, TransportNetwork, subnetwork
from urban_routing.model import Edge, Metric, Node, RoadType

TOL = 1e-9


class TestModel(unittest.TestCase):
    def test_travel_time_derives_from_speed_and_congestion(self):
        e = Edge("a", "b", 10.0, 60.0)             # 10 km at 60 km/h = 10 min
        self.assertAlmostEqual(e.travel_time_min, 10.0)
        e.congestion = 0.5                          # half speed => double time
        self.assertAlmostEqual(e.travel_time_min, 20.0)
        self.assertAlmostEqual(e.free_flow_time_min, 10.0)

    def test_gridlock_is_slow_but_finite(self):
        e = Edge("a", "b", 1.0, 50.0, congestion=1.0)
        self.assertLess(e.travel_time_min, float("inf"))
        self.assertGreater(e.travel_time_min, 10.0)

    def test_all_metrics_are_non_negative(self):
        """Dijkstra's correctness depends on this."""
        net = build_aurora_city()
        for metric in (Metric.time, Metric.distance, Metric.free_flow_time,
                       Metric.balanced(0.5)):
            for e in net.edges():
                self.assertGreaterEqual(metric(e), 0.0)


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.net = build_aurora_city()

    def test_counts_and_lookup(self):
        self.assertEqual(self.net.node_count, 30)
        self.assertGreaterEqual(self.net.edge_count, 50)
        self.assertIsNotNone(self.net.edge_between("N01", "N02"))
        self.assertIsNone(self.net.edge_between("N01", "N26_missing"))

    def test_two_way_creates_both_directions(self):
        self.assertIsNotNone(self.net.edge_between("N01", "N02"))
        self.assertIsNotNone(self.net.edge_between("N02", "N01"))

    def test_one_way_street_has_no_reverse(self):
        self.assertIsNotNone(self.net.edge_between("N05", "N02"))
        self.assertIsNone(self.net.edge_between("N02", "N05"))

    def test_duplicate_edge_rejected(self):
        with self.assertRaises(ValueError):
            self.net.add_link("N01", "N02", 1.0, 30, two_way=False)

    def test_unknown_endpoint_rejected(self):
        with self.assertRaises(KeyError):
            self.net.add_link("N01", "NOPE", 1.0, 30)

    def test_congestion_update_is_reflected_in_time(self):
        before = self.net.edge_between("N01", "N02").travel_time_min
        self.net.set_congestion("N01", "N02", 0.95)
        after = self.net.edge_between("N01", "N02").travel_time_min
        self.assertGreater(after, before)

    def test_remove_edge(self):
        self.net.remove_edge("N01", "N02")
        self.assertIsNone(self.net.edge_between("N01", "N02"))
        self.assertNotIn("N02", [e.target for e in self.net.neighbours("N01")])

    def test_matrix_and_list_hold_the_same_edges(self):
        mat = MatrixNetwork(self.net, Metric.time)
        for e in self.net.edges():
            self.assertTrue(mat.has_edge(e.source, e.target))
        total = sum(1 for i in range(mat.size) for _ in mat.neighbours(i))
        self.assertEqual(total, self.net.edge_count)

    def test_subnetwork_only_keeps_allowed_edges(self):
        road = subnetwork(self.net, RouteConstraints.road_vehicle())
        self.assertTrue(all(e.road_type is not RoadType.TRANSIT for e in road.edges()))
        self.assertLess(road.edge_count, self.net.edge_count)


class TestShortestPaths(unittest.TestCase):
    def setUp(self):
        self.net = build_aurora_city()

    def test_trivial_and_missing(self):
        r = dijkstra(self.net, "N01", "N01", Metric.time)
        self.assertTrue(r.found)
        self.assertAlmostEqual(r.cost, 0.0)
        with self.assertRaises(KeyError):
            dijkstra(self.net, "NOPE", "N01")

    def test_path_is_contiguous_and_cost_matches_edges(self):
        r = dijkstra(self.net, "N17", "N10", Metric.time)
        self.assertTrue(r.found)
        for i, e in enumerate(r.edges):
            self.assertEqual(e.source, r.path[i])
            self.assertEqual(e.target, r.path[i + 1])
        self.assertAlmostEqual(r.cost, sum(e.travel_time_min for e in r.edges), places=9)

    def test_dijkstra_a_star_bellman_ford_agree(self):
        """Differential test over every pair in the baseline network."""
        ids = list(self.net.node_ids())
        for metric in (Metric.time, Metric.distance):
            for s in ids:
                for t in ids:
                    d = dijkstra(self.net, s, t, metric)
                    a = a_star(self.net, s, t, metric)
                    self.assertEqual(d.found, a.found, f"{s}->{t}")
                    if d.found:
                        self.assertAlmostEqual(d.cost, a.cost, places=9,
                                               msg=f"A* != Dijkstra on {s}->{t}")

    def test_bellman_ford_agrees_on_a_sample(self):
        rng = random.Random(7)
        ids = list(self.net.node_ids())
        for _ in range(40):
            s, t = rng.choice(ids), rng.choice(ids)
            d = dijkstra(self.net, s, t, Metric.time)
            b = bellman_ford(self.net, s, t, Metric.time)
            self.assertEqual(d.found, b.found)
            if d.found:
                self.assertAlmostEqual(d.cost, b.cost, places=9)

    def test_agreement_on_synthetic_networks(self):
        for seed in (1, 2, 3):
            net = generate_synthetic_network(60, avg_degree=4.0, seed=seed)
            ids = list(net.node_ids())
            rng = random.Random(seed)
            for _ in range(25):
                s, t = rng.choice(ids), rng.choice(ids)
                d = dijkstra(net, s, t, Metric.time)
                a = a_star(net, s, t, Metric.time)
                b = bellman_ford(net, s, t, Metric.time)
                self.assertEqual((d.found, a.found), (b.found, b.found))
                if d.found:
                    self.assertAlmostEqual(d.cost, a.cost, places=9)
                    self.assertAlmostEqual(d.cost, b.cost, places=9)

    def test_a_star_heuristic_is_admissible(self):
        """h(n) must never exceed the true remaining cost."""
        from urban_routing.algorithms import _heuristic_factory

        target = "N10"
        for metric in (Metric.distance, Metric.time):
            h = _heuristic_factory(self.net, target, metric)
            for nid in self.net.node_ids():
                true = dijkstra(self.net, nid, target, metric)
                if true.found:
                    self.assertLessEqual(h(nid), true.cost + 1e-9,
                                         f"inadmissible at {nid}")

    def test_a_star_expands_no_more_than_dijkstra_on_average(self):
        ids = list(self.net.node_ids())
        rng = random.Random(3)
        d_total = a_total = 0
        for _ in range(40):
            s, t = rng.choice(ids), rng.choice(ids)
            d_total += dijkstra(self.net, s, t, Metric.distance).nodes_expanded
            a_total += a_star(self.net, s, t, Metric.distance).nodes_expanded
        self.assertLessEqual(a_total, d_total)

    def test_different_metrics_can_give_different_routes(self):
        fast = dijkstra(self.net, "N17", "N10", Metric.time,
                        RouteConstraints.road_vehicle())
        short = dijkstra(self.net, "N17", "N10", Metric.distance,
                         RouteConstraints.road_vehicle())
        self.assertNotEqual(fast.path, short.path)
        self.assertLessEqual(fast.total_time_min, short.total_time_min + TOL)
        self.assertLessEqual(short.total_distance_km, fast.total_distance_km + TOL)

    def test_bfs_minimises_hops_not_time(self):
        b = bfs_min_hops(self.net, "N17", "N10")
        d = dijkstra(self.net, "N17", "N10", Metric.time)
        self.assertTrue(b.found)
        self.assertLessEqual(b.hops, d.hops)
        self.assertGreaterEqual(b.total_time_min, d.total_time_min - TOL)


class TestConstrainedRouting(unittest.TestCase):
    def setUp(self):
        self.net = build_aurora_city()

    def test_blocked_edge_is_never_used(self):
        blocked = RouteConstraints(blocked_edges=frozenset({("N07", "N08")}))
        r = dijkstra(self.net, "N01", "N09", Metric.time, blocked)
        self.assertNotIn(("N07", "N08"), [(e.source, e.target) for e in r.edges])

    def test_blocked_node_is_never_entered(self):
        blocked = RouteConstraints(blocked_nodes=frozenset({"N08"}))
        r = dijkstra(self.net, "N01", "N09", Metric.time, blocked)
        self.assertTrue(r.found)
        self.assertNotIn("N08", r.path)

    def test_constraining_never_makes_a_route_cheaper(self):
        rng = random.Random(11)
        ids = list(self.net.node_ids())
        car = RouteConstraints.road_vehicle()
        for _ in range(40):
            s, t = rng.choice(ids), rng.choice(ids)
            free = dijkstra(self.net, s, t, Metric.time)
            limited = dijkstra(self.net, s, t, Metric.time, car)
            if limited.found:
                self.assertTrue(free.found)
                self.assertGreaterEqual(limited.cost, free.cost - TOL)

    def test_road_profile_excludes_transit_and_pedestrian(self):
        car = RouteConstraints.road_vehicle()
        rng = random.Random(5)
        ids = list(self.net.node_ids())
        for _ in range(30):
            s, t = rng.choice(ids), rng.choice(ids)
            r = dijkstra(self.net, s, t, Metric.time, car)
            for e in r.edges:
                self.assertNotIn(e.road_type, (RoadType.TRANSIT, RoadType.PEDESTRIAN))

    def test_hgv_cannot_reach_the_old_town(self):
        r = dijkstra(self.net, "N17", "N04", Metric.time,
                     RouteConstraints.heavy_goods_vehicle())
        self.assertFalse(r.found)

    def test_congestion_ceiling_is_respected(self):
        c = RouteConstraints.avoiding_congestion(0.4)
        r = dijkstra(self.net, "N17", "N10", Metric.time, c)
        for e in r.edges:
            self.assertLessEqual(e.congestion, 0.4)

    def test_constraints_compose(self):
        c = RouteConstraints.road_vehicle().with_blocked(
            nodes={"N29"}, edges={("N07", "N08")}
        )
        r = dijkstra(self.net, "N01", "N09", Metric.time, c)
        self.assertNotIn("N29", r.path)
        self.assertNotIn(("N07", "N08"), [(e.source, e.target) for e in r.edges])

    def test_a_star_respects_constraints_identically(self):
        c = RouteConstraints.road_vehicle().with_blocked(edges={("N07", "N08")})
        d = dijkstra(self.net, "N01", "N09", Metric.time, c)
        a = a_star(self.net, "N01", "N09", Metric.time, c)
        self.assertAlmostEqual(d.cost, a.cost, places=9)


class TestReachability(unittest.TestCase):
    def setUp(self):
        self.net = build_aurora_city()

    def test_matches_full_dijkstra_then_filter(self):
        """The pruned scan must return exactly what a full scan would, filtered."""
        source = "N22"
        full, _ = dijkstra_costs(self.net, source, Metric.time)
        for budget in (1, 3, 5, 8, 12, 20, 100):
            bounded = reachable_within(self.net, source, budget, Metric.time)
            expected = {k: v for k, v in full.items() if v <= budget}
            self.assertEqual(set(bounded.costs), set(expected), f"budget={budget}")
            for k, v in expected.items():
                self.assertAlmostEqual(bounded.costs[k], v, places=9)

    def test_budget_is_monotonic(self):
        prev = -1
        for budget in (1, 2, 5, 10, 20, 50):
            n = reachable_within(self.net, "N22", budget, Metric.time).reached
            self.assertGreaterEqual(n, prev)
            prev = n

    def test_source_always_reachable_at_zero_cost(self):
        r = reachable_within(self.net, "N01", 0.0, Metric.time)
        self.assertEqual(set(r.costs), {"N01"})

    def test_generous_budget_reaches_everything(self):
        r = reachable_within(self.net, "N01", 10_000.0, Metric.time)
        self.assertEqual(r.reached, self.net.node_count)
        self.assertAlmostEqual(r.coverage(self.net), 1.0)

    def test_hop_reachability_is_bfs_depth(self):
        d = reachable_within_hops(self.net, "N01", 1)
        direct = {e.target for e in self.net.neighbours("N01")} | {"N01"}
        self.assertEqual(set(d), direct)

    def test_reachability_honours_constraints(self):
        wide = reachable_within(self.net, "N17", 30, Metric.time)
        hgv = reachable_within(self.net, "N17", 30, Metric.time,
                               RouteConstraints.heavy_goods_vehicle())
        self.assertLess(hgv.reached, wide.reached)


class TestConnectivityAndBottlenecks(unittest.TestCase):
    def setUp(self):
        self.net = build_aurora_city()
        self.road = subnetwork(self.net, RouteConstraints.road_vehicle())

    def test_baseline_network_is_connected(self):
        report = connected_components(self.net)
        self.assertTrue(report.connected)
        self.assertEqual(report.component_count, 1)
        self.assertTrue(is_strongly_connected(self.net))

    def test_disconnecting_a_node_is_detected(self):
        net = build_aurora_city()
        for u, v in [("N16", "N17"), ("N17", "N16")]:
            net.remove_edge(u, v)
        report = connected_components(net)
        self.assertFalse(report.connected)
        self.assertIn("N17", report.isolated)

    def test_tarjan_matches_brute_force_on_baseline(self):
        report = find_bottlenecks(self.road)
        self.assertEqual(report.articulation_points, verify_critical_nodes(self.road))
        self.assertEqual(
            sorted(tuple(sorted(p)) for p in report.bridges),
            verify_critical_edges(self.road),
        )

    def test_tarjan_matches_brute_force_on_synthetic_networks(self):
        for seed in (1, 2, 3, 4, 5):
            net = generate_synthetic_network(40, avg_degree=3.0, seed=seed)
            report = find_bottlenecks(net)
            self.assertEqual(report.articulation_points, verify_critical_nodes(net),
                             f"seed={seed}")
            self.assertEqual(
                sorted(tuple(sorted(p)) for p in report.bridges),
                verify_critical_edges(net), f"seed={seed}",
            )

    def test_identified_bridge_really_disconnects(self):
        report = find_bottlenecks(self.road)
        self.assertTrue(report.bridges)
        u, v = report.bridges[0]
        net = subnetwork(build_aurora_city(), RouteConstraints.road_vehicle())
        net.remove_edge(u, v)
        net.remove_edge(v, u)
        self.assertFalse(connected_components(net).connected)

    def test_identified_cut_vertex_really_disconnects(self):
        report = find_bottlenecks(self.road)
        self.assertIn("N25", report.articulation_points)
        blocked = RouteConstraints.road_vehicle().with_blocked(nodes={"N25"})
        self.assertFalse(dijkstra(self.road, "N01", "N26", Metric.time, blocked).found)

    def test_a_cycle_has_no_bottlenecks(self):
        net = TransportNetwork()
        for i in range(6):
            net.add_node(Node(f"C{i}", f"C{i}", float(i), 0.0))
        for i in range(6):
            net.add_link(f"C{i}", f"C{(i + 1) % 6}", 1.0, 50, two_way=True)
        report = find_bottlenecks(net)
        self.assertEqual(report.articulation_points, [])
        self.assertEqual(report.bridges, [])

    def test_a_path_has_every_interior_node_critical(self):
        net = TransportNetwork()
        for i in range(5):
            net.add_node(Node(f"P{i}", f"P{i}", float(i), 0.0))
        for i in range(4):
            net.add_link(f"P{i}", f"P{i+1}", 1.0, 50, two_way=True)
        report = find_bottlenecks(net)
        self.assertEqual(report.articulation_points, ["P1", "P2", "P3"])
        self.assertEqual(len(report.bridges), 4)


class TestDatasets(unittest.TestCase):
    def test_baseline_meets_the_brief_minimum(self):
        net = build_aurora_city()
        self.assertGreaterEqual(net.node_count, 25)
        self.assertGreaterEqual(net.edge_count, 50)

    def test_study_sets_meet_the_minimum_and_are_connected(self):
        for name, (n, _, _, _) in STUDY_SETS.items():
            net = build_study_set(name)
            self.assertEqual(net.node_count, n, name)
            self.assertGreaterEqual(net.edge_count, 50, name)
            self.assertTrue(connected_components(net).connected, name)

    def test_no_edge_is_shorter_than_the_straight_line(self):
        """Physical invariant, and the precondition for A* admissibility."""
        from urban_routing.dataset import validate_geometry

        for name, net in [("aurora", build_aurora_city())] + [
            (n, build_study_set(n)) for n in STUDY_SETS
        ]:
            self.assertEqual(validate_geometry(net), [], name)

    def test_generation_is_reproducible(self):
        a = generate_synthetic_network(60, seed=99)
        b = generate_synthetic_network(60, seed=99)
        self.assertEqual(
            [(e.source, e.target, e.distance_km, e.congestion) for e in a.edges()],
            [(e.source, e.target, e.distance_km, e.congestion) for e in b.edges()],
        )

    def test_generated_networks_stay_sparse_as_they_grow(self):
        """The premise of choosing an adjacency list."""
        for n in (100, 400, 1600):
            net = generate_synthetic_network(n, avg_degree=4.0, seed=n)
            avg_degree = net.edge_count / net.node_count
            self.assertLess(avg_degree, 8.0, f"n={n}")
            self.assertLess(net.edge_count, net.node_count ** 2 / 4, f"n={n}")

    def test_csv_round_trip(self):
        import tempfile
        from pathlib import Path

        from urban_routing.dataset import export_csv, load_csv

        net = build_aurora_city()
        with tempfile.TemporaryDirectory() as tmp:
            npath, epath = export_csv(net, Path(tmp), "t")
            back = load_csv(npath, epath)
        self.assertEqual(back.node_count, net.node_count)
        self.assertEqual(back.edge_count, net.edge_count)
        r1 = dijkstra(net, "N17", "N10", Metric.time)
        r2 = dijkstra(back, "N17", "N10", Metric.time)
        self.assertAlmostEqual(r1.cost, r2.cost, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
