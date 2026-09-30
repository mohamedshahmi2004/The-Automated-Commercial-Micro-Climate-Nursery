"""Network representation (Design Decision Area 1).

Two representations are implemented so the choice can be measured rather than
asserted:

    TransportNetwork  -- adjacency list (the one the engine uses)
    MatrixNetwork     -- adjacency matrix (built only for the benchmark)

Why the adjacency list wins for a city road network
---------------------------------------------------
A road network is extremely sparse.  Intersections have a bounded degree -- a
junction of six streets is unusual -- so |E| is O(|V|), not O(|V|^2).  For
n = 10,000 intersections a matrix holds 100,000,000 cells of which ~30,000 are
non-empty: 99.97% waste.

    operation                 adjacency list        adjacency matrix
    -------------------------------------------------------------------
    space                     O(V + E)              O(V^2)
    iterate neighbours of u   O(deg(u))             O(V)
    edge lookup (u,v)         O(1)*                 O(1)
    insert edge               O(1) amortised        O(1)
    add node                  O(1)                  O(V^2) (full rebuild)

    * this class keeps a side index ``_edge_index`` mapping (u,v) -> Edge, so
      the adjacency list gives O(1) existence checks too and gives up nothing
      to the matrix.  The index costs O(E) extra space, which on a sparse graph
      is still O(V).

Dijkstra dominates our workload, and its inner loop is "iterate the neighbours
of the node just popped".  That loop is O(deg(u)) on a list and O(V) on a
matrix, which is exactly the difference between O(E log V) and O(V^2) overall.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Iterator

from .model import Edge, Node, RoadType


class TransportNetwork:
    """Directed, weighted graph stored as an adjacency list.

    Internals
    ---------
    ``_nodes``      id -> Node
    ``_adjacency``  id -> list[Edge]   (outgoing edges; the hot path)
    ``_reverse``    id -> list[Edge]   (incoming edges; used by undirected
                                        traversals such as bridge detection and
                                        by reverse searches)
    ``_edge_index`` (u, v) -> Edge     (O(1) "does this connection exist?")
    """

    __slots__ = ("_nodes", "_adjacency", "_reverse", "_edge_index", "_pair_counter")

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._adjacency: dict[str, list[Edge]] = defaultdict(list)
        self._reverse: dict[str, list[Edge]] = defaultdict(list)
        self._edge_index: dict[tuple[str, str], Edge] = {}
        self._pair_counter = 0

    # ---- construction ----------------------------------------------------

    def add_node(self, node: Node) -> None:
        """O(1)."""
        self._nodes[node.id] = node
        self._adjacency.setdefault(node.id, [])
        self._reverse.setdefault(node.id, [])

    def add_edge(self, edge: Edge) -> Edge:
        """Insert one directed connection.  O(1) amortised."""
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise KeyError(f"unknown endpoint in {edge}")
        if (edge.source, edge.target) in self._edge_index:
            raise ValueError(f"duplicate edge {edge.source}->{edge.target}")
        self._adjacency[edge.source].append(edge)
        self._reverse[edge.target].append(edge)
        self._edge_index[(edge.source, edge.target)] = edge
        return edge

    def add_link(
        self,
        source: str,
        target: str,
        distance_km: float,
        speed_kmh: float,
        road_type: RoadType = RoadType.ARTERIAL,
        congestion: float = 0.0,
        name: str = "",
        two_way: bool = True,
    ) -> list[Edge]:
        """Convenience builder.  A two-way street becomes two paired edges."""
        pair_id = self._pair_counter
        self._pair_counter += 1
        made = [
            self.add_edge(
                Edge(source, target, distance_km, speed_kmh, road_type,
                     congestion, name, pair_id)
            )
        ]
        if two_way:
            made.append(
                self.add_edge(
                    Edge(target, source, distance_km, speed_kmh, road_type,
                         congestion, name, pair_id)
                )
            )
        return made

    # ---- queries ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edge_index)

    def node(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def nodes(self) -> Iterator[Node]:
        return iter(self._nodes.values())

    def node_ids(self) -> Iterable[str]:
        return self._nodes.keys()

    def edges(self) -> Iterator[Edge]:
        return iter(self._edge_index.values())

    def neighbours(self, node_id: str) -> list[Edge]:
        """Outgoing edges of a node.  O(1) to fetch, O(deg) to walk."""
        return self._adjacency[node_id]

    def incoming(self, node_id: str) -> list[Edge]:
        return self._reverse[node_id]

    def edge_between(self, source: str, target: str) -> Edge | None:
        """O(1) existence / lookup, thanks to the side index."""
        return self._edge_index.get((source, target))

    def degree(self, node_id: str) -> int:
        return len(self._adjacency[node_id])

    def undirected_neighbours(self, node_id: str) -> set[str]:
        """Neighbours ignoring direction -- used by connectivity analysis."""
        out = {e.target for e in self._adjacency[node_id]}
        out.update(e.source for e in self._reverse[node_id])
        out.discard(node_id)
        return out

    # ---- live updates ----------------------------------------------------

    def set_congestion(self, source: str, target: str, value: float,
                       both_ways: bool = False) -> None:
        """Real-time update.  O(1) -- the reason we keep the edge index."""
        edge = self._edge_index.get((source, target))
        if edge is None:
            raise KeyError(f"no edge {source}->{target}")
        edge.congestion = min(1.0, max(0.0, value))
        if both_ways:
            back = self._edge_index.get((target, source))
            if back is not None:
                back.congestion = edge.congestion

    def remove_edge(self, source: str, target: str) -> Edge | None:
        """Permanently delete a connection.  O(deg) because of the list scan.

        Note: routing that merely *avoids* an edge should use RouteConstraints
        instead -- that is O(1) per query and does not mutate the network.
        """
        edge = self._edge_index.pop((source, target), None)
        if edge is None:
            return None
        self._adjacency[source] = [e for e in self._adjacency[source] if e is not edge]
        self._reverse[target] = [e for e in self._reverse[target] if e is not edge]
        return edge

    # ---- reporting -------------------------------------------------------

    def summary(self) -> str:
        deg = [self.degree(n) for n in self._nodes]
        avg = sum(deg) / len(deg) if deg else 0.0
        density = (
            self.edge_count / (self.node_count * (self.node_count - 1))
            if self.node_count > 1
            else 0.0
        )
        return (
            f"{self.node_count} nodes, {self.edge_count} directed edges, "
            f"avg out-degree {avg:.2f}, max {max(deg, default=0)}, "
            f"density {density:.5f}"
        )


class MatrixNetwork:
    """Adjacency-matrix representation, kept purely for the comparison.

    Stored as a list of lists of ``float`` costs (``inf`` = no connection).
    Present so the report's Decision Area 1 claims are backed by measurement:
    see ``benchmark.representation_comparison``.
    """

    __slots__ = ("index", "ids", "matrix", "size")

    def __init__(self, network: TransportNetwork, metric) -> None:
        self.ids = list(network.node_ids())
        self.index = {nid: i for i, nid in enumerate(self.ids)}
        self.size = len(self.ids)
        inf = float("inf")
        self.matrix = [[inf] * self.size for _ in range(self.size)]
        for edge in network.edges():
            i, j = self.index[edge.source], self.index[edge.target]
            self.matrix[i][j] = min(self.matrix[i][j], metric(edge))

    def neighbours(self, i: int):
        """O(V): the matrix must scan a whole row to find real neighbours."""
        row = self.matrix[i]
        for j in range(self.size):
            w = row[j]
            if w != float("inf"):
                yield j, w

    def has_edge(self, u: str, v: str) -> bool:
        return self.matrix[self.index[u]][self.index[v]] != float("inf")

    def cell_count(self) -> int:
        return self.size * self.size


def subnetwork(network: TransportNetwork, constraints) -> TransportNetwork:
    """Materialise the sub-network a set of constraints permits.

    Routing never needs this -- constraints are applied inside the search loop
    at O(1) per edge.  Structural analysis does: "is the city still connected
    *for heavy goods vehicles*?" is a question about a different graph, and
    Tarjan's algorithm wants that graph up front.

    Cost: O(V + E), paid once per analysis rather than once per query.
    """
    out = TransportNetwork()
    for node in network.nodes():
        if constraints.allows_node(node.id, node.zone):
            out.add_node(node)
    for edge in network.edges():
        if edge.source in out and edge.target in out and constraints.allows_edge(edge):
            out.add_edge(
                Edge(edge.source, edge.target, edge.distance_km, edge.speed_kmh,
                     edge.road_type, edge.congestion, edge.name, edge.pair_id)
            )
    return out
