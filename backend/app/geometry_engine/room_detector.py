"""
Room Detector – Orchestrator
=============================
Coordinates the full geometry pipeline:

    segments → SegmentCleaner → WallGraphBuilder → PolygonDetector
    → RoomMerger → DoorDetector → label association → RoomClassifier → rooms

Pipeline stages:
1. Layer filtering and geometry cleaning (snap, dedup, node).
2. Wall graph construction and polygon extraction.
3. Polygon merging (shared boundary + centroid proximity).
4. Door detection from ARC entities → room adjacency.
5. Text-label matching (inside-polygon first, then nearest-centroid).
6. Geometry-based classification for unlabelled rooms.
"""

import logging
from typing import Any, List

from shapely.geometry import Point

from app.geometry_engine.segment_cleaner import SegmentCleaner
from app.geometry_engine.room_merger import RoomMerger
from app.geometry_engine.wall_graph_builder import WallGraphBuilder
from app.geometry_engine.polygon_detector import PolygonDetector
from app.geometry_engine.door_detector import DoorDetector
from app.geometry_engine.room_classifier import classify_rooms
from app.geometry_engine.room_classifier import classify_room_contextual

logger = logging.getLogger(__name__)


class RoomDetector:
    """Detects rooms from extracted DXF geometry."""

    def __init__(
        self,
        geometry_data: dict,
        *,
        allowed_layers: list[str] | None = None,
        min_room_area_sqft: float = 30.0,
        min_bath_area_sqft: float = 10.0,
        label_max_dist: float = 5.0,
    ):
        self.raw_segments = geometry_data.get("segments", [])
        self.texts = geometry_data.get("texts", [])
        self.doors_raw = geometry_data.get("doors", [])
        self.blocks_raw = geometry_data.get("blocks", [])
        self.metadata = geometry_data.get("metadata", {})
        inferred_layers = self.metadata.get("candidate_wall_layers") or []
        self.allowed_layers = allowed_layers or inferred_layers or [
            "WALL", "A-WALL", "STRUCTURE", "OUTLINE", "0",
        ]
        self.min_room_area_sqft = min_room_area_sqft
        self.min_bath_area_sqft = min_bath_area_sqft
        self.label_max_dist = label_max_dist

    def detect_rooms(self) -> List[dict]:
        # 1. Determine if layer filtering is useful
        unique_layers = {str(s.get("layer", "")).upper() for s in self.raw_segments}
        effective_layers = self.allowed_layers
        if len(unique_layers) <= 1:
            effective_layers = None  # accept all layers

        # 2. Layer filtering + geometry cleaning
        cleaned_segments = SegmentCleaner(
            self.raw_segments,
            allowed_layers=effective_layers,
            min_len=0.005,
            snap_tol=0.001,
        ).clean()

        if not cleaned_segments:
            logger.warning("No cleaned segments available for room detection.")
            return []

        # 3. Build wall graph and detect cycles
        builder = WallGraphBuilder(cleaned_segments)
        noded = builder.build()
        cycles = builder.detect_cycles()

        # 4. Extract candidate polygons
        area_to_sqft = float(self.metadata.get("area_to_sqft", 1.0) or 1.0)
        candidates = PolygonDetector(
            noded,
            cycles=cycles,
            area_to_sqft=area_to_sqft,
            min_room_area_sqft=self.min_room_area_sqft,
            min_bath_area_sqft=self.min_bath_area_sqft,
            remove_outer=True,
        ).detect()

        if not candidates:
            logger.warning("No room candidates detected.")
            return []

        # 5. Merge fragmented polygons
        merger = RoomMerger(
            candidates,
            min_shared_ratio=0.3,
            thin_wall_gap=0.15,
            centroid_max_dist=2.0,
        )
        merged = merger.merge()
        if merged:
            # Accept merge results; recalculate area_sqft
            for m in merged:
                m["area_sqft"] = round(m["area_raw"] * area_to_sqft, 2)
            candidates = merged

        # 6. Door detection
        door_info: dict[str, Any] = {
            "door_positions": [],
            "doors": [],
            "room_adjacency": [],
            "room_door_counts": {},
        }
        if self.doors_raw:
            door_detector = DoorDetector(
                self.doors_raw, candidates, snap_dist=1.0,
            )
            door_info = door_detector.detect()

        # 7. Attach nearest text label to each room
        rooms = self._associate_labels(candidates)

        # 8. Classify rooms by geometry when no labels were found
        has_any_label = any(r.get("original_label") for r in rooms)
        if not has_any_label:
            rooms = classify_rooms(rooms, door_counts=door_info.get("room_door_counts"))

        # 8a. Associate furniture blocks to room polygons
        self._attach_blocks(rooms)

        # 9. Store door/adjacency metadata on rooms
        for idx, room in enumerate(rooms):
            room["door_count"] = door_info.get("room_door_counts", {}).get(idx, 0)
        self._adjacency = door_info.get("room_adjacency", [])
        self._doors = door_info.get("doors", [])

        # 10. Contextual classification using rules + adjacency + furniture.
        self._classify_with_context(rooms)

        logger.info("Detected %d rooms", len(rooms))
        return rooms

    @property
    def adjacency(self) -> list[tuple[int, int]]:
        return getattr(self, "_adjacency", [])

    @property
    def doors(self) -> list[dict]:
        return getattr(self, "_doors", [])

    # ------------------------------------------------------------------
    def _associate_labels(self, candidates: list[dict]) -> list[dict]:
        """Match extracted text labels to room polygons.

        Priority:
        1. Text that falls **inside** the polygon boundary.
        2. Nearest text to the polygon centroid within ``label_max_dist``.
        """
        rooms: list[dict] = []
        used_texts: set[int] = set()

        for i, cand in enumerate(candidates):
            poly = cand.get("polygon")
            centroid_coords = cand["centroid"]
            centroid_pt = Point(centroid_coords)
            best_text: str | None = None
            best_dist = float("inf")
            best_idx = -1

            for ti, txt in enumerate(self.texts):
                if ti in used_texts:
                    continue
                pt = Point(txt["position"][:2])

                # Priority 1: text inside the polygon
                if poly is not None and poly.contains(pt):
                    best_text = txt["text"]
                    best_dist = 0.0
                    best_idx = ti
                    break

                # Priority 2: nearest to centroid
                dist = centroid_pt.distance(pt)
                if dist < best_dist:
                    best_dist = dist
                    best_text = txt["text"]
                    best_idx = ti

            if best_dist > self.label_max_dist:
                best_text = None

            if best_idx >= 0 and best_text is not None:
                used_texts.add(best_idx)

            label = best_text or f"Room {i + 1}"
            area_raw = cand.get("area_raw", 0.0)
            area_to_sqft = float(self.metadata.get("area_to_sqft", 1.0) or 1.0)
            area_sqft = cand.get("area_sqft", round(area_raw * area_to_sqft, 2))
            rooms.append({
                "name": label,
                "original_label": best_text,
                "area": round(area_raw, 4),
                "area_sqft": area_sqft,
                "centroid": centroid_coords,
                "coordinates": cand["coordinates"],
                "polygon": cand.get("polygon"),  # kept for classifier
            })

        return rooms

    def _attach_blocks(self, rooms: list[dict]) -> None:
        """Attach block/furniture detections to containing rooms."""
        if not self.blocks_raw:
            for room in rooms:
                room["blocks"] = []
                room["furniture"] = []
            return

        for room in rooms:
            room["blocks"] = []
            room["furniture"] = []

        for block in self.blocks_raw:
            bpos = block.get("position")
            if not bpos or len(bpos) < 2:
                continue
            pt = Point(bpos[0], bpos[1])
            for room in rooms:
                poly = room.get("polygon")
                if poly is None:
                    continue
                if poly.contains(pt):
                    room["blocks"].append(block)
                    btype = block.get("type")
                    if btype and btype != "GENERIC":
                        room["furniture"].append(btype)
                    break

    def _classify_with_context(self, rooms: list[dict]) -> None:
        """Run deterministic contextual classification and attach confidence."""
        adjacency_map: dict[int, set[int]] = {i: set() for i in range(len(rooms))}
        for a, b in self._adjacency:
            adjacency_map.setdefault(a, set()).add(b)
            adjacency_map.setdefault(b, set()).add(a)

        current_labels = [str(r.get("name", "")) for r in rooms]
        for idx, room in enumerate(rooms):
            poly = room.get("polygon")
            if poly is None:
                room["confidence"] = 0.5
                room["classification_method"] = "fallback"
                continue

            neighbor_labels = [
                current_labels[n] for n in sorted(adjacency_map.get(idx, set()))
                if 0 <= n < len(current_labels)
            ]
            result = classify_room_contextual(
                area_sqft=float(room.get("area_sqft", 0.0) or 0.0),
                polygon=poly,
                door_count=int(room.get("door_count", 0) or 0),
                furniture_types=room.get("furniture", []),
                adjacent_labels=neighbor_labels,
                text_label=room.get("original_label"),
                vision_label=room.get("vision_label"),
                vision_confidence=float(room.get("vision_confidence", 0.0) or 0.0),
            )
            room["name"] = result["label"]
            room["confidence"] = result["confidence"]
            room["classification_method"] = result["method"]
            room["needs_ai_refine"] = result.get("needs_ai", False)
            room["adjacent_rooms"] = sorted(adjacency_map.get(idx, set()))
