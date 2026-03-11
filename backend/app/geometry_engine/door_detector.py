"""
Door Detector
=============
Detects doors from ARC entities (90° quarter-circle swings) and
INSERT block references with door-related names.

Doors are used to:
1. Build a room-adjacency graph (rooms sharing a doorway are neighbors).
2. Validate room boundaries — a polygon with zero doors might be a
   shaft, void, or false detection.
3. Confirm merging decisions — adjacent polygons separated only by a
   door opening should NOT be merged.
"""

import logging
from typing import Any

from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


class DoorDetector:
    """Locate doors and compute room adjacency."""

    def __init__(
        self,
        doors: list[dict],
        rooms: list[dict],
        *,
        snap_dist: float = 1.0,
    ):
        self.doors = doors
        self.rooms = rooms
        self.snap_dist = snap_dist

    def detect(self) -> dict[str, Any]:
        """Return door positions and room adjacency info.

        Returns
        -------
        dict with keys:
            door_positions: list of [x, y] coordinates
            room_adjacency: list of (room_i, room_j) index pairs
            room_door_counts: dict mapping room index to door count
        """
        door_positions: list[list[float]] = []
        door_objects: list[dict[str, Any]] = []
        for d in self.doors:
            door_positions.append(d["center"])

        adjacency: list[tuple[int, int]] = []
        door_counts: dict[int, int] = {i: 0 for i in range(len(self.rooms))}

        for door_idx, door in enumerate(self.doors):
            dp = Point(door["center"])
            # Expand search radius: door center + door radius (swing reach)
            search_radius = self.snap_dist + float(door.get("radius", 0.0) or 0.0)
            touching: list[int] = []

            for ri, room in enumerate(self.rooms):
                poly = room.get("polygon")
                if poly is None:
                    continue
                dist = poly.boundary.distance(dp)
                if dist <= search_radius:
                    touching.append(ri)

            # Update door counts
            for ri in touching:
                door_counts[ri] = door_counts.get(ri, 0) + 1

            # Every pair of rooms touching the same door are adjacent
            for i in range(len(touching)):
                for j in range(i + 1, len(touching)):
                    pair = (min(touching[i], touching[j]),
                            max(touching[i], touching[j]))
                    if pair not in adjacency:
                        adjacency.append(pair)

            door_objects.append({
                "door_id": door_idx,
                "position": list(door.get("center", [0.0, 0.0])),
                "width": float(door.get("width", 0.0) or 0.0),
                "source": door.get("source", "unknown"),
                "block_name": door.get("block_name"),
                "connected_rooms": touching,
            })

        logger.info(
            "DoorDetector: %d doors → %d adjacencies across %d rooms",
            len(self.doors), len(adjacency), len(self.rooms),
        )
        return {
            "door_positions": door_positions,
            "doors": door_objects,
            "room_adjacency": adjacency,
            "room_door_counts": door_counts,
        }
