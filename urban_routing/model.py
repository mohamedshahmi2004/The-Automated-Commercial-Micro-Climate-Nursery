"""Core domain model: what a node and an edge in the transport network hold.

Design Decision Area 4 (attributes) lives here.  The scenario lists five
candidate edge attributes -- distance, travel time, speed limit, road type and
congestion.  We model four of them directly and derive the fifth:

    distance_km   stored   -- the physical length of the connection.
    speed_kmh     stored   -- the free-flow (posted) speed limit.
    road_type     stored   -- drives restriction-based (constrained) routing.
    congestion    stored   -- a live 0.0-1.0 load factor, updated at runtime.
    travel_time   derived  -- computed from the three above.

Travel time is derived rather than stored because it is the only attribute that
changes every time congestion changes.  Storing it too would mean two sources of
truth that can silently disagree; deriving it costs one float division per edge
relaxation and keeps the network consistent by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class RoadType(str, Enum):
    """Class of a connection.  Used by constrained routing."""

    MOTORWAY = "motorway"
    ARTERIAL = "arterial"
    RESIDENTIAL = "residential"
    BRIDGE = "bridge"
    TUNNEL = "tunnel"
    TRANSIT = "transit"          # rail / metro link, not usable by road vehicles
    PEDESTRIAN = "pedestrian"    # not usable by any vehicle


#: Road classes a heavy goods vehicle may legally use.
HEAVY_VEHICLE_ALLOWED = frozenset(
    {RoadType.MOTORWAY, RoadType.ARTERIAL, RoadType.BRIDGE}
)

#: Road classes any road vehicle (car, ambulance, AV) may use.
ROAD_VEHICLE_ALLOWED = frozenset(
    {
        RoadType.MOTORWAY,
        RoadType.ARTERIAL,
        RoadType.RESIDENTIAL,
        RoadType.BRIDGE,
        RoadType.TUNNEL,
    }
)


@dataclass(frozen=True, slots=True)
class Node:
    """A location: an intersection, interchange or transit stop.

    ``x`` and ``y`` are planar coordinates in kilometres relative to an
    arbitrary city origin.  They exist for one reason: they give A* an
    admissible heuristic (straight-line distance never overestimates road
    distance).  Without coordinates A* degenerates into Dijkstra.
    """

    id: str
    name: str
    x: float = 0.0
    y: float = 0.0
    zone: str = "core"

    def distance_to(self, other: "Node") -> float:
        """Euclidean (straight-line) distance in km."""
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(slots=True)
class Edge:
    """A directed connection between two nodes.

    A two-way street is stored as two Edge objects sharing a ``pair_id``.  This
    costs ~2x memory over an undirected representation but buys the ability to
    model one-way streets, asymmetric congestion (inbound rush hour vs.
    outbound) and per-direction closures -- all of which a real city has.
    """

    source: str
    target: str
    distance_km: float
    speed_kmh: float
    road_type: RoadType = RoadType.ARTERIAL
    congestion: float = 0.0          # 0.0 = free flow, 1.0 = gridlock
    name: str = ""
    pair_id: int | None = None       # shared by the two halves of a two-way link

    # ---- derived metrics -------------------------------------------------

    @property
    def effective_speed_kmh(self) -> float:
        """Speed after congestion.

        We use a linear degradation capped at 90% loss.  A fully congested link
        is therefore slow but still finite -- an infinite cost would make the
        edge invisible to the router rather than merely unattractive, which is
        not what "gridlock" means operationally.
        """
        return self.speed_kmh * max(0.10, 1.0 - self.congestion)

    @property
    def travel_time_min(self) -> float:
        """Travel time in minutes under current congestion."""
        return 60.0 * self.distance_km / self.effective_speed_kmh

    @property
    def free_flow_time_min(self) -> float:
        """Travel time in minutes ignoring congestion (used as an A* bound)."""
        return 60.0 * self.distance_km / self.speed_kmh

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Edge({self.source}->{self.target}, {self.distance_km:.2f}km, "
            f"{self.road_type.value}, cong={self.congestion:.2f})"
        )


# ---------------------------------------------------------------------------
# Cost metrics
# ---------------------------------------------------------------------------


class Metric:
    """Named edge-cost functions ("optimise for what?").

    Every routing algorithm in this project takes a metric callable, so the same
    Dijkstra implementation answers "shortest", "fastest" and "balanced"
    queries.  Every metric must be non-negative for Dijkstra/A* to stay correct.
    """

    @staticmethod
    def distance(edge: Edge) -> float:
        return edge.distance_km

    @staticmethod
    def time(edge: Edge) -> float:
        return edge.travel_time_min

    @staticmethod
    def free_flow_time(edge: Edge) -> float:
        return edge.free_flow_time_min

    @staticmethod
    def balanced(alpha: float = 0.5):
        """Weighted combination of normalised time and distance.

        ``alpha`` = 1.0 is pure travel time, 0.0 is pure distance.  Distance is
        scaled by 60/50 (minutes to cover 1 km at a nominal 50 km/h) so the two
        terms are in comparable units before weighting.
        """

        def _cost(edge: Edge) -> float:
            return alpha * edge.travel_time_min + (1.0 - alpha) * (
                edge.distance_km * 60.0 / 50.0
            )

        _cost.__name__ = f"balanced(alpha={alpha})"
        return _cost
