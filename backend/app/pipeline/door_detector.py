"""
Door Detector
=============
Detects door positions from two sources:

1. **ARC entities** – door swings are quarter-circle arcs with radii
   between 700 mm and 1 100 mm.
2. **INSERT blocks** – block names containing door-related keywords.

Outputs
-------
For each detected door:
    id          : sequential integer
    center      : [x, y] in drawing units
    width_m     : estimated door width in metres
    source      : "arc" | "block"
    block_name  : str | None
    layer       : str

Room association
----------------
Each door center is tested against every room polygon boundary.  Rooms
whose boundary is within ``snap_dist`` of the door center are considered
to share that doorway → used to build room adjacency.
"""

from __future__ import annotations

import logging
from typing import Any

from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

# Search radius around door center to find touching rooms (drawing units)
_DEFAULT_SNAP_DIST: float = 1.5


class DoorDetector:
    """
    Associate doors to rooms and build room adjacency.

    Parameters
    ----------
    doors : list of door dicts from DXFParser
        Each dict must have: center, radius_m (may be 0), width_m, source, layer.
        Optional: block_name.
    rooms : list of room candidate dicts
        Each dict must have: polygon (Shapely Polygon).
    snap_dist : float
        Max distance (drawing units) from door center to room boundary to
        classify them as "touching".  Default 1.5 du.
    to_metres : float
        Drawing units → metres conversion.  Used to scale snap_dist.
    """

    def __init__(
        self,
        doors: list[dict],
        rooms: list[dict],
        *,
        snap_dist: float = _DEFAULT_SNAP_DIST,
        to_metres: float = 1.0,
    ) -> None:
        self.raw_doors = doors
        self.rooms = rooms
        self.snap_dist = snap_dist
        self.to_metres = to_metres

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> dict[str, Any]:
        """
        Run detection and return:

        Returns
        -------
        dict:
            doors          : list of enriched door dicts (with connected_rooms)
            room_adjacency : list of (i, j) room-index pairs sharing a door
            room_door_counts : dict {room_index: int}
        """
        enriched: list[dict] = []
        adjacency: list[tuple[int, int]] = []
        door_counts: dict[int, int] = {i: 0 for i in range(len(self.rooms))}

        for door_id, raw in enumerate(self.raw_doors):
            center = raw.get("center", [0.0, 0.0])
            dp = Point(center[0], center[1])

            # Search radius: snap_dist + the door's own swing radius
            radius_du = float(raw.get("radius", 0.0) or 0.0)
            search_r = self.snap_dist + radius_du

            touching: list[int] = []
            for ri, room in enumerate(self.rooms):
                poly: Polygon | None = room.get("polygon")
                if poly is None:
                    continue
                try:
                    # Check boundary distance first (fast); also check
                    # containment so a door center inside a small room registers
                    if poly.boundary.distance(dp) <= search_r or poly.contains(dp):
                        touching.append(ri)
                except Exception:
                    pass

            # Update door counts
            for ri in touching:
                door_counts[ri] = door_counts.get(ri, 0) + 1

            # Build adjacency pairs
            for i in range(len(touching)):
                for j in range(i + 1, len(touching)):
                    pair = (min(touching[i], touching[j]), max(touching[i], touching[j]))
                    if pair not in adjacency:
                        adjacency.append(pair)

            enriched.append({
                "id": door_id,
                "center": list(center),
                "width_m": float(raw.get("width_m", raw.get("radius_m", 0.9)) or 0.9),
                "source": raw.get("source", "unknown"),
                "block_name": raw.get("block_name"),
                "layer": raw.get("layer", "0"),
                "connected_rooms": touching,
            })

        logger.info(
            "DoorDetector: %d doors → %d adjacency pairs across %d rooms",
            len(enriched), len(adjacency), len(self.rooms),
        )

        return {
            "doors": enriched,
            "room_adjacency": adjacency,
            "room_door_counts": door_counts,
        }
