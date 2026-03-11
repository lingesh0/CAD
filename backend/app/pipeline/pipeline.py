"""
CAD Pipeline – Top-Level Orchestrator
======================================
Converts a DXF/DWG file into a structured floorplan JSON model with
very high accuracy (target 95–98%) and minimal AI dependency.

Pipeline stages
---------------
1.  Parse   – DXFParser extracts segments, texts, door arcs, INSERT blocks
2.  Graph   – WallGraph filters layers, closes door-gaps, polygonizes
3.  Doors   – DoorDetector associates arcs/blocks to room boundaries
4.  Blocks  – BlockDetector attaches furniture to room polygons
5.  Classify– RoomClassifier scores each room (geometry+blocks+adjacency)
6.  Adjacency–AdjacencyGraph builds full room connectivity
7.  Snapshot– SnapshotRenderer generates floor + per-room PNGs
8.  Vision  – VisionValidator (Gemini) validates ambiguous rooms (conf<0.75)
9.  Re-classify – rerun scorer with vision results merged in
10. Output  – produce final JSON

Performance
-----------
With ezdxf + shapely + numpy the typical 10k-segment DXF completes in
≈ 2–4 s on a modern server (aim: < 3 s per floor).

Usage
-----
>>> from app.pipeline import run_pipeline
>>> result = run_pipeline("path/to/plan.dxf", output_dir="outputs/")
>>> print(result["floors"][0]["rooms"])
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from app.pipeline.cad_parser        import DXFParser
from app.pipeline.room_detector     import RoomDetector
from app.pipeline.door_detector     import DoorDetector
from app.pipeline.block_detector    import BlockDetector
from app.pipeline.room_classifier   import RoomClassifier
from app.pipeline.adjacency_graph   import AdjacencyGraph
from app.pipeline.snapshot_renderer import SnapshotRenderer
from app.pipeline.ocr_labels        import OCRLabelExtractor
from app.pipeline.vision_validator  import VisionValidator
from app.pipeline.svg_converter     import convert_to_svg
from app.pipeline.raster_segmenter  import RasterSegmenter
from app.pipeline.room_matcher      import RoomMatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DWG → DXF conversion (librecad-based; optional)
# ---------------------------------------------------------------------------

def _convert_dwg(dwg_path: str, output_dir: str) -> str:
    """Convert a DWG file to DXF using LibreCAD / ODA converter.

    Falls back to the original path if conversion is unavailable.
    """
    try:
        from app.cad_parser.converter import DWGConverter
        conv = DWGConverter()
        return conv.convert_to_dxf(dwg_path, output_dir)
    except Exception as exc:
        logger.warning("DWG converter unavailable (%s) – trying to read as DXF.", exc)
        return dwg_path


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

class CADPipeline:
    """
    High-accuracy DXF/DWG floorplan extraction pipeline.

    Parameters
    ----------
    output_dir : str | Path
        Directory for PNG snapshots and intermediate outputs.
    min_room_area_sqft : float
        Polygons smaller than this are discarded (e.g. wall pockets).
    vision_threshold : float
        Rooms with confidence below this value are sent to Gemini Vision.
    use_vision : bool
        Set False to disable Gemini Vision even if GEMINI_API_KEY is set.
    gemini_api_key : str | None
        Override; if None the env-var GEMINI_API_KEY is used.
    dpi : int
        Snapshot resolution.  150 is fast; 300 for production quality.
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs/snapshots",
        *,
        min_room_area_sqft: float = 8.0,
        vision_threshold: float = 0.75,
        use_vision: bool = False,
        use_raster: bool = True,
        generate_svg: bool = False,
        gemini_api_key: str | None = None,
        dpi: int = 150,
    ) -> None:
        self.output_dir       = Path(output_dir)
        self.min_room_area    = min_room_area_sqft
        self.vision_threshold = vision_threshold
        self.use_vision       = use_vision
        self.use_raster       = use_raster
        self.generate_svg     = generate_svg
        self.gemini_api_key   = gemini_api_key
        self.dpi = dpi

    # ------------------------------------------------------------------

    def run(self, file_path: str, *, floor_id: int | str | None = None) -> dict:
        """
        Execute the full pipeline on a single DXF (or DWG) file.

        Parameters
        ----------
        file_path : absolute path to the DXF/DWG file
        floor_id  : identifier used in snapshot filenames.  Defaults to
                    the stem of the filename.

        Returns
        -------
        dict matching the target JSON schema:
            {
              "file": str,
              "status": "completed" | "failed",
              "floors": [{
                "floor": 1,
                "snapshot": str,
                "rooms": [{
                  "name": str,
                  "area_sqft": float,
                  "doors": int,
                  "windows": int,
                  "adjacent": list[str],
                  "confidence": float,
                  "classification_method": str,
                  "furniture": list[str],
                  "snapshot": str,
                }],
              }],
            }
        """
        t0 = time.perf_counter()
        fpath = Path(file_path)

        if floor_id is None:
            floor_id = fpath.stem

        logger.info("=== CADPipeline: %s (floor_id=%s) ===", file_path, floor_id)

        # ── Convert DWG if necessary ────────────────────────────────────
        dxf_path = str(fpath)
        if fpath.suffix.lower() == ".dwg":
            dxf_path = _convert_dwg(str(fpath), str(self.output_dir))

        # ── Stage 1: Parse DXF ─────────────────────────────────────────
        t1 = time.perf_counter()
        parser = DXFParser(dxf_path)
        geometry = parser.parse()
        logger.info("[1/8] Parse  %.2fs | segs=%d texts=%d doors=%d blocks=%d",
                    time.perf_counter() - t1,
                    len(geometry["segments"]),
                    len(geometry["texts"]),
                    len(geometry["doors"]),
                    len(geometry["blocks"]))

        meta = geometry.get("metadata", {})
        area_to_sqft: float = float(meta.get("area_to_sqft", 1.0) or 1.0)
        to_metres:    float = float(meta.get("to_metres", 1.0) or 1.0)
        wall_layers: list[str] = meta.get("wall_layers", []) or []

        # ── Stage 2a: Vector Room Detection (wall graph + gap closure) ──
        t2 = time.perf_counter()
        room_candidates = RoomDetector(
            segments=geometry["segments"],
            wall_layers=wall_layers,
            to_metres=to_metres,
            area_to_sqft=area_to_sqft,
            min_room_area_sqft=self.min_room_area,
        ).detect_rooms()
        logger.info(
            "[2a/11] Vector rooms  %.2fs | candidates=%d",
            time.perf_counter() - t2, len(room_candidates),
        )

        # ── Stage 2b: Optional SVG export (audit / downstream parsing) ─
        if self.generate_svg:
            try:
                svg_file = str(self.output_dir / f"{floor_id}.svg")
                convert_to_svg(dxf_path, svg_file, segments=geometry["segments"])
                logger.info("[2b/11] SVG exported → %s", svg_file)
            except Exception as _svg_exc:
                logger.warning("[2b/11] SVG export failed: %s", _svg_exc)

        # ── Stage 2c: Raster segmentation ──────────────────────────────
        raster_rooms: list[dict] = []
        if self.use_raster:
            t2c = time.perf_counter()
            raster_rooms = RasterSegmenter(
                min_room_area_sqft=self.min_room_area,
                area_to_sqft=area_to_sqft,
                to_metres=to_metres,
            ).segment_rooms(geometry["segments"])
            logger.info(
                "[2c/11] Raster rooms  %.2fs | candidates=%d",
                time.perf_counter() - t2c, len(raster_rooms),
            )

        # ── Stage 2d: Fuse vector + raster rooms ───────────────────────
        if raster_rooms:
            t2d = time.perf_counter()
            room_candidates = RoomMatcher(to_metres=to_metres).match(
                room_candidates, raster_rooms
            )
            logger.info(
                "[2d/11] Fused rooms   %.2fs | total=%d",
                time.perf_counter() - t2d, len(room_candidates),
            )

        if not room_candidates:
            logger.warning("No room candidates – returning empty result.")
            return self._empty_result(str(fpath), floor_id)

        # ── Stage 3: Door Detection ────────────────────────────────────
        t3 = time.perf_counter()
        doors_raw = geometry.get("doors", [])
        door_info = DoorDetector(
            doors_raw,
            room_candidates,
            to_metres=to_metres,
        ).detect()
        logger.info("[3/11] Doors  %.2fs | doors=%d adj_pairs=%d",
                    time.perf_counter() - t3,
                    len(door_info["doors"]),
                    len(door_info["room_adjacency"]))

        # ── Stage 4: Block/Furniture Detection ────────────────────────
        t4 = time.perf_counter()
        blocks_raw = geometry.get("blocks", [])
        BlockDetector(blocks_raw, room_candidates).detect()
        logger.info("[4/11] Blocks %.2fs | blocks=%d", time.perf_counter() - t4, len(blocks_raw))

        # ── Stage 5: Text-Label Association (DXF text) ────────────────
        t5 = time.perf_counter()
        self._associate_text_labels(room_candidates, geometry.get("texts", []))
        logger.info("[5/11] Labels %.2fs", time.perf_counter() - t5)

        # ── Stage 6: Adjacency Graph ───────────────────────────────────
        t6 = time.perf_counter()
        adj_graph = AdjacencyGraph(
            room_candidates,
            door_adjacency=door_info["room_adjacency"],
            use_shared_walls=True,
        ).build()
        adjacency_pairs = adj_graph.pairs()
        logger.info("[6/11] Adj    %.2fs | pairs=%d", time.perf_counter() - t6, len(adjacency_pairs))

        # Attach door counts and adjacency names to each room candidate
        door_counts = door_info.get("room_door_counts", {})
        for idx, room in enumerate(room_candidates):
            room["door_count"] = door_counts.get(idx, 0)
            room["adjacent_rooms"] = adj_graph.adjacent_names(idx)

        # ── Stage 7: Snapshots + OCR labels ────────────────────────────
        t7 = time.perf_counter()
        renderer = SnapshotRenderer(self.output_dir, dpi=self.dpi)
        floor_snapshot = renderer.render_floor(
            geometry["segments"],
            room_candidates,
            floor_id=floor_id,
            doors=door_info["doors"],
            texts=geometry.get("texts"),
        )
        room_snapshots = renderer.render_rooms(
            geometry["segments"],
            room_candidates,
            floor_id=floor_id,
        )
        ocr_count = OCRLabelExtractor().detect_room_labels(
            floor_snapshot=floor_snapshot,
            room_snapshots=room_snapshots,
            rooms=room_candidates,
        )
        logger.info("[7/11] Snap+OCR %.2fs | ocr_labels=%d", time.perf_counter() - t7, ocr_count)

        # ── Stage 8: Rule-based Classification ────────────────────────
        classifier = RoomClassifier(
            room_candidates,
            adjacency_pairs=adjacency_pairs,
            area_to_sqft=area_to_sqft,
        )
        rooms = classifier.classify_all()

        # Optional Vision stage (disabled by default; API-free mode)
        if self.use_vision:
            t10 = time.perf_counter()
            ambiguous = classifier.needs_vision()
            if ambiguous:
                room_snapshots_for_vision: list[str | None] = list(room_snapshots)
                validator = VisionValidator(
                    self.gemini_api_key,
                    confidence_threshold=self.vision_threshold,
                )
                vision_results = validator.validate_rooms(
                    rooms,
                    floor_snapshot,
                    room_snapshots_for_vision,
                )
                if vision_results:
                    classifier.apply_vision(vision_results)
                    logger.info("[10/11] Vision %.2fs | updated=%d rooms",
                                time.perf_counter() - t10, len(vision_results))
                else:
                    logger.info("[10/11] Vision %.2fs | no updates", time.perf_counter() - t10)
            else:
                logger.info("[10/11] Vision skipped (all rooms high-confidence)")
        else:
            logger.info("[10/11] Vision disabled (API-free mode)")

        # ── Final JSON ────────────────────────────────────────────────
        total_elapsed = time.perf_counter() - t0
        logger.info("=== CADPipeline DONE %.2fs ===", total_elapsed)

        return self._build_result(
            filepath=str(fpath),
            floor_id=floor_id,
            rooms=rooms,
            door_objects=door_info["doors"],
            adjacency_pairs=adjacency_pairs,
            floor_snapshot=floor_snapshot,
            room_snapshots=room_snapshots,
            elapsed_sec=total_elapsed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _associate_text_labels(
        rooms: list[dict],
        texts: list[dict],
        max_dist: float = 5.0,
    ) -> None:
        """Match extracted text labels to room polygons (in-place)."""
        from shapely.geometry import Point

        used: set[int] = set()
        for room in rooms:
            poly = room.get("polygon")
            centroid = room.get("centroid", [0, 0])
            cpt = Point(centroid[0], centroid[1])
            best_text: str | None = None
            best_dist = float("inf")
            best_idx = -1

            for ti, txt in enumerate(texts):
                if ti in used:
                    continue
                pos = txt.get("position", [0, 0])
                pt = Point(pos[0], pos[1])

                # Inside wins immediately
                if poly is not None:
                    try:
                        if poly.contains(pt):
                            best_text = txt["text"]
                            best_dist = 0.0
                            best_idx = ti
                            break
                    except Exception:
                        pass

                dist = cpt.distance(pt)
                if dist < best_dist:
                    best_dist = dist
                    best_text = txt["text"]
                    best_idx = ti

            if best_dist > max_dist:
                best_text = None
                best_idx = -1

            if best_idx >= 0 and best_text is not None:
                used.add(best_idx)

            room["original_label"] = best_text
            if not room.get("name") or room["name"].startswith("Room "):
                room["name"] = best_text or room.get("name", "")

    @staticmethod
    def _empty_result(filepath: str, floor_id: Any) -> dict:
        return {
            "file": os.path.basename(filepath),
            "status": "no_rooms_detected",
            "floors": [{"floor": 1, "rooms": [], "snapshot": None}],
        }

    @staticmethod
    def _build_result(
        *,
        filepath: str,
        floor_id: Any,
        rooms: list[dict],
        door_objects: list[dict],
        adjacency_pairs: list[tuple[int, int]],
        floor_snapshot: str,
        room_snapshots: list[str],
        elapsed_sec: float,
    ) -> dict:
        room_payloads: list[dict] = []
        for idx, room in enumerate(rooms):
            area = float(room.get("area_sqft", 0.0) or 0.0)
            snap = room_snapshots[idx] if idx < len(room_snapshots) else None
            room_payloads.append({
                "name":                   room.get("name", f"Room {idx+1}"),
                "area_sqft":              round(area, 2),
                "doors":                  int(room.get("door_count", 0) or 0),
                "windows":                0,   # window detection not yet implemented
                "adjacent":               room.get("adjacent_rooms", []),
                "confidence":             float(room.get("confidence", 0.5) or 0.5),
                "classification_method":  room.get("classification_method", "rules"),
                "furniture":              room.get("furniture", []),
                "centroid":               room.get("centroid", [0.0, 0.0]),
                "coordinates":            room.get("coordinates", []),
                "snapshot":               snap,
            })

        return {
            "file":        os.path.basename(filepath),
            "status":      "completed",
            "elapsed_sec": round(elapsed_sec, 3),
            "floors": [
                {
                    "floor":    1,
                    "snapshot": floor_snapshot,
                    "doors":    door_objects,
                    "adjacency": [[a, b] for a, b in adjacency_pairs],
                    "rooms":    room_payloads,
                }
            ],
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_pipeline(
    file_path: str,
    output_dir: str = "outputs/snapshots",
    *,
    min_room_area_sqft: float = 8.0,
    vision_threshold: float = 0.75,
    use_vision: bool = False,
    use_raster: bool = True,
    generate_svg: bool = False,
    gemini_api_key: str | None = None,
    dpi: int = 150,
    floor_id: int | str | None = None,
) -> dict:
    """Convenience function – creates a ``CADPipeline`` and calls ``run()``."""
    pipeline = CADPipeline(
        output_dir=output_dir,
        min_room_area_sqft=min_room_area_sqft,
        vision_threshold=vision_threshold,
        use_vision=use_vision,
        use_raster=use_raster,
        generate_svg=generate_svg,
        gemini_api_key=gemini_api_key,
        dpi=dpi,
    )
    return pipeline.run(file_path, floor_id=floor_id)
