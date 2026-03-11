"""
Room Detector
=============
Builds room candidates from wall geometry using wall-graph extraction,
door-gap closure, and post-filters for nested polygons and thin artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon

from app.pipeline.wall_graph import WallGraph


@dataclass
class RoomDetector:
    segments: list[dict]
    wall_layers: list[str]
    to_metres: float
    area_to_sqft: float
    min_room_area_sqft: float = 8.0
    thin_room_min_compactness: float = 0.10

    def detect_rooms(self) -> list[dict]:
        wall_graph = WallGraph(
            self.segments,
            wall_layers=self.wall_layers,
            to_metres=self.to_metres,
        )
        rooms = wall_graph.extract_polygons(
            min_area_sqft=self.min_room_area_sqft,
            area_to_sqft=self.area_to_sqft,
            remove_outer=True,
        )
        rooms = self._remove_nested(rooms)
        rooms = self._remove_thin_artifacts(rooms)
        for idx, room in enumerate(rooms):
            room.setdefault("name", f"Room {idx + 1}")
        return rooms

    def _remove_nested(self, rooms: list[dict]) -> list[dict]:
        out: list[dict] = []
        for i, room in enumerate(rooms):
            poly_i: Polygon | None = room.get("polygon")
            if poly_i is None:
                continue
            nested = False
            for j, other in enumerate(rooms):
                if i == j:
                    continue
                poly_j: Polygon | None = other.get("polygon")
                if poly_j is None:
                    continue
                if poly_i.within(poly_j) and poly_j.area > poly_i.area * 1.05:
                    nested = True
                    break
            if not nested:
                out.append(room)
        return out

    def _remove_thin_artifacts(self, rooms: list[dict]) -> list[dict]:
        out: list[dict] = []
        for room in rooms:
            poly: Polygon | None = room.get("polygon")
            if poly is None:
                continue
            p = max(poly.length, 1e-9)
            compactness = min(1.0, 4.0 * 3.141592653589793 * poly.area / (p * p))
            room["compactness"] = round(float(compactness), 4)
            if compactness < self.thin_room_min_compactness and room.get("area_sqft", 0.0) < 35.0:
                continue
            out.append(room)
        return out
