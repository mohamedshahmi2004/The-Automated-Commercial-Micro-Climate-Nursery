"""Aurora City intelligent urban routing engine.

A stdlib-only prototype routing engine for a smart-mobility platform:
network representation, optimal path finding, constrained routing,
reachability analysis, connectivity/bottleneck detection and a performance
harness.
"""

from .algorithms import RouteResult, a_star, bellman_ford, bfs_min_hops, dijkstra
from .analysis import (
    ReachabilityResult,
    connected_components,
    find_bottlenecks,
    is_strongly_connected,
    reachable_within,
)
from .constraints import RouteConstraints
from .dataset import build_aurora_city, build_study_set, generate_synthetic_network
from .graph import MatrixNetwork, TransportNetwork, subnetwork
from .model import Edge, Metric, Node, RoadType

__all__ = [
    "RouteResult", "dijkstra", "a_star", "bellman_ford", "bfs_min_hops",
    "ReachabilityResult", "reachable_within", "connected_components",
    "is_strongly_connected", "find_bottlenecks",
    "RouteConstraints", "TransportNetwork", "MatrixNetwork", "subnetwork",
    "build_aurora_city", "build_study_set", "generate_synthetic_network",
    "Edge", "Node", "RoadType", "Metric",
]
__version__ = "1.0.0"
