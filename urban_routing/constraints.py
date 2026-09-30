"""Constrained routing (blocked roads, zones, vehicle classes).

Design note
-----------
There are three ways to implement "route but avoid X":

  1. Physically delete the nodes/edges, route, then restore them.
     O(V + E) per query and destructive -- unusable for concurrent queries.
  2. Build a filtered copy of the network per query.
     Also O(V + E) per query, and it duplicates the whole graph in memory.
  3. Apply a *predicate* inside the relaxation loop.
     O(1) per edge examined, no copying, no mutation, and it composes.

We use (3).  ``RouteConstraints.allows_edge`` is called exactly once per edge
the search actually relaxes, so a constrained query costs the same asymptotic
time as an unconstrained one -- and usually less, because pruned edges are
never expanded.  Crucially this works identically for Dijkstra, A*, BFS and the
reachability scan: they all consult the same object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .model import Edge, RoadType


@dataclass(slots=True)
class RouteConstraints:
    """A filter applied during search.

    Attributes
    ----------
    blocked_nodes       locations that must not be entered (incidents, closures)
    blocked_edges       specific directed connections (u, v) that are closed
    forbidden_road_types road classes this vehicle may not use
    allowed_road_types  if set, the *only* classes this vehicle may use
    max_congestion      refuse links above this live congestion level
    avoid_zones         node zones to route around (e.g. a low-emission zone)
    min_clearance_m     placeholder for height/weight style restrictions
    predicate           escape hatch for an arbitrary extra rule
    """

    blocked_nodes: frozenset[str] = frozenset()
    blocked_edges: frozenset[tuple[str, str]] = frozenset()
    forbidden_road_types: frozenset[RoadType] = frozenset()
    allowed_road_types: frozenset[RoadType] | None = None
    max_congestion: float = 1.0
    avoid_zones: frozenset[str] = frozenset()
    predicate: Callable[[Edge], bool] | None = None

    # ---- the hot path: O(1) ---------------------------------------------

    def allows_edge(self, edge: Edge) -> bool:
        if (edge.source, edge.target) in self.blocked_edges:
            return False
        if edge.target in self.blocked_nodes:
            return False
        if edge.road_type in self.forbidden_road_types:
            return False
        if self.allowed_road_types is not None and edge.road_type not in self.allowed_road_types:
            return False
        if edge.congestion > self.max_congestion:
            return False
        if self.predicate is not None and not self.predicate(edge):
            return False
        return True

    def allows_node(self, node_id: str, zone: str = "") -> bool:
        if node_id in self.blocked_nodes:
            return False
        if zone and zone in self.avoid_zones:
            return False
        return True

    def is_empty(self) -> bool:
        return (
            not self.blocked_nodes
            and not self.blocked_edges
            and not self.forbidden_road_types
            and self.allowed_road_types is None
            and self.max_congestion >= 1.0
            and not self.avoid_zones
            and self.predicate is None
        )

    # ---- ready-made vehicle profiles ------------------------------------

    @classmethod
    def unrestricted(cls) -> "RouteConstraints":
        return cls()

    @classmethod
    def road_vehicle(cls) -> "RouteConstraints":
        """A car / autonomous shuttle: no rail links, no pedestrian ways."""
        from .model import ROAD_VEHICLE_ALLOWED

        return cls(allowed_road_types=frozenset(ROAD_VEHICLE_ALLOWED))

    @classmethod
    def heavy_goods_vehicle(cls) -> "RouteConstraints":
        """An HGV: motorways, arterials and bridges only -- no tunnels, no
        residential streets, no transit or pedestrian links."""
        from .model import HEAVY_VEHICLE_ALLOWED

        return cls(allowed_road_types=frozenset(HEAVY_VEHICLE_ALLOWED))

    @classmethod
    def emergency(cls) -> "RouteConstraints":
        """An ambulance: may use everything a road vehicle can, and ignores
        congestion limits (blue-light priority), but still cannot drive on rail
        or pedestrian links."""
        from .model import ROAD_VEHICLE_ALLOWED

        return cls(allowed_road_types=frozenset(ROAD_VEHICLE_ALLOWED))

    @classmethod
    def avoiding_congestion(cls, threshold: float = 0.6) -> "RouteConstraints":
        from .model import ROAD_VEHICLE_ALLOWED

        return cls(
            allowed_road_types=frozenset(ROAD_VEHICLE_ALLOWED),
            max_congestion=threshold,
        )

    def with_blocked(
        self,
        nodes: frozenset[str] | set[str] = frozenset(),
        edges: frozenset[tuple[str, str]] | set[tuple[str, str]] = frozenset(),
    ) -> "RouteConstraints":
        """Return a copy with extra closures -- constraints compose."""
        return RouteConstraints(
            blocked_nodes=frozenset(self.blocked_nodes | set(nodes)),
            blocked_edges=frozenset(self.blocked_edges | set(edges)),
            forbidden_road_types=self.forbidden_road_types,
            allowed_road_types=self.allowed_road_types,
            max_congestion=self.max_congestion,
            avoid_zones=self.avoid_zones,
            predicate=self.predicate,
        )
