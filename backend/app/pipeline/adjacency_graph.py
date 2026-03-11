"""
Adjacency Graph
===============
Builds a connectivity graph between rooms using detected door positions
and shared wall segments.

The adjacency graph is used by:
1. ``RoomClassifier`` – adjacency rules improve classification accuracy.
2. The final JSON output – ``adjacent`` list per room.
3. ``SnapshotRenderer`` – for drawing connectivity overlays.

Graph representation
--------------------
Nodes: room indices (integers)
Edges: (i, j) where rooms i and j share a doorway or a significant wall boundary

Output
------
``adjacency_pairs``  : list of (i, j) tuples (i < j always)
``room_adjacents``   : dict {room_idx: [list of adjacent room names]}
"""

from __future__ import annotations

import logging

import networkx as nx
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

# Minimum shared boundary length (drawing units) to count as adjacency
_MIN_SHARED_LEN: float = 0.3


class AdjacencyGraph:
    """
    Build and query the room adjacency graph.

    Parameters
    ----------
    rooms : list of room dicts
        Each must have a ``polygon`` (Shapely Polygon).
    door_adjacency : list of (i, j) pairs from DoorDetector
    use_shared_walls : bool
        If True, also detect adjacency via shared wall boundaries
        (polygons that share an edge or near-edge).  Slower but more
        complete for DXFs without explicit door entities.
    """

    def __init__(
        self,
        rooms: list[dict],
        door_adjacency: list[tuple[int, int]] | None = None,
        *,
        use_shared_walls: bool = True,
    ) -> None:
        self.rooms = rooms
        self.door_adjacency = door_adjacency or []
        self.use_shared_walls = use_shared_walls
        self._graph: nx.Graph = nx.Graph()
        self._built = False

    # ------------------------------------------------------------------

    def build(self) -> "AdjacencyGraph":
        """Populate the internal graph.  Call once before querying."""
        n = len(self.rooms)
        self._graph = nx.Graph()
        self._graph.add_nodes_from(range(n))

        # 1. Door-based adjacency (from DoorDetector)
        for a, b in self.door_adjacency:
            if 0 <= a < n and 0 <= b < n:
                self._graph.add_edge(a, b, source="door")

        # 2. Shared-wall adjacency (polygon boundary proximity)
        if self.use_shared_walls and n > 1:
            self._add_wall_adjacency()

        self._built = True
        logger.info(
            "AdjacencyGraph: %d rooms, %d door edges, %d total edges",
            n,
            len(self.door_adjacency),
            self._graph.number_of_edges(),
        )
        return self

    def pairs(self) -> list[tuple[int, int]]:
        """Return all (i, j) pairs with i < j."""
        return [(min(a, b), max(a, b)) for a, b in self._graph.edges()]

    def neighbors(self, room_idx: int) -> list[int]:
        """Return indices of rooms adjacent to ``room_idx``."""
        if not self._built:
            self.build()
        return list(self._graph.neighbors(room_idx))

    def adjacent_names(self, room_idx: int) -> list[str]:
        """Return names of rooms adjacent to ``room_idx``."""
        return [
            self.rooms[ni].get("name", f"Room {ni+1}")
            for ni in self.neighbors(room_idx)
            if 0 <= ni < len(self.rooms)
        ]

    def as_dict(self) -> dict[int, list[str]]:
        """Return {room_idx: [adjacent_room_names]} for all rooms."""
        return {i: self.adjacent_names(i) for i in range(len(self.rooms))}

    # ------------------------------------------------------------------

    def _add_wall_adjacency(self) -> None:
        """Add edges for rooms that share a significant boundary."""
        n = len(self.rooms)
        for i in range(n):
            poly_i: Polygon | None = self.rooms[i].get("polygon")
            if poly_i is None:
                continue
            for j in range(i + 1, n):
                if self._graph.has_edge(i, j):
                    continue
                poly_j: Polygon | None = self.rooms[j].get("polygon")
                if poly_j is None:
                    continue
                try:
                    shared = poly_i.intersection(poly_j.boundary)
                    if shared.is_empty:
                        shared = poly_i.boundary.intersection(poly_j.boundary)
                    length = getattr(shared, "length", 0.0)
                    if length >= _MIN_SHARED_LEN:
                        self._graph.add_edge(i, j, source="shared_wall", length=length)
                except Exception:
                    pass
