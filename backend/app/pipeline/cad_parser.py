"""
CAD Parser
==========
Extracts geometry, text labels, door arcs, and block inserts from a DXF file.

Architectural layer filtering
------------------------------
Wall layers  : A-WALL, WALL, WALLS, STRUCTURE, STRUCT, CORE, ARCH, OUTLINE
Door layers  : A-DOOR, DOOR, DOORS
Window layers: A-WIND, WIND, WINDOW
Furniture    : A-FURN, FURN, FURNITURE
Text/labels  : A-TEXT, TEXT, ANNO, LABELS

Ignored layers (noise)
-----------------------
DIMENSION, DIM, GRID, CONSTRUCTION, HATCH, FILL, BOUNDARY, TITLEBLOCK,
BORDER, NORTH, SCALE, REVISION, NOTES, ANNO (only when combined with DIM)

Units
-----
INSUNITS header is respected; drawing coordinates are kept raw (not converted
to feet or mm) – the caller passes ``area_to_sqft`` for any area computation.
"""

from __future__ import annotations

import math
import logging
from typing import Any

import ezdxf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer classification sets (upper-case substrings)
# ---------------------------------------------------------------------------
_WALL_LAYER_HINTS: set[str] = {
    "A-WALL", "WALL", "WALLS", "STRUCTURE", "STRUCT", "CORE",
    "ARCH", "OUTLINE", "PARTITION", "BEARING",
}
_DOOR_LAYER_HINTS: set[str] = {"A-DOOR", "DOOR", "DOORS", "DR"}
_WINDOW_LAYER_HINTS: set[str] = {"A-WIND", "WIND", "WINDOW", "WIN"}
_FURNITURE_LAYER_HINTS: set[str] = {"A-FURN", "FURN", "FURNITURE", "FIXTURE"}
_TEXT_LAYER_HINTS: set[str] = {"A-TEXT", "TEXT", "LABELS", "LABEL", "ROOM"}

# Layers that contribute noise and should be ignored for wall extraction
_IGNORE_LAYER_KEYWORDS: set[str] = {
    "DIM", "DIMENSION", "GRID", "CONSTRUCTION", "HATCH", "FILL",
    "BOUNDARY", "TITLEBLOCK", "TITLE", "BORDER", "NORTH", "SCALE",
    "REVISION", "NOTES", "VIEWPORT", "DEFPOINTS",
}

# ---------------------------------------------------------------------------
# Door block name detection
# ---------------------------------------------------------------------------
_DOOR_BLOCK_KEYWORDS: set[str] = {
    "DOOR", "DR", "SWING", "ENTRY", "GATE", "HINGED",
}

# ---------------------------------------------------------------------------
# Furniture keyword → semantic type mapping (block name substrings, upper-case)
# ---------------------------------------------------------------------------
_FURNITURE_MAP: dict[str, str] = {
    "BED":     "BED",
    "SOFA":    "SOFA",
    "COUCH":   "SOFA",
    "DINING":  "DINING",
    "DINE":    "DINING",
    "TABLE":   "TABLE",
    "STOVE":   "STOVE",
    "OVEN":    "STOVE",
    "KITCHEN": "STOVE",
    "WC":      "WC",
    "TOILET":  "WC",
    "COMMODE": "WC",
    "SINK":    "SINK",
    "BASIN":   "SINK",
    "BATH":    "BATH",
    "TUB":     "BATH",
    "SHOWER":  "SHOWER",
    "DESK":    "DESK",
    "STUDY":   "DESK",
    "WARDROBE":"WARDROBE",
    "ALMIRAH": "WARDROBE",
    "FRIDGE":  "FRIDGE",
    "REFRIG":  "FRIDGE",
    "WASH":    "WASHER",
}

# Door arc radius range in drawing units (raw).  We scale by the unit factor
# below only for detection; values are also stored raw.
# Typical door swing radius: 700 mm – 1 000 mm  →  0.7 m – 1.0 m
_DOOR_ARC_RADIUS_MIN_M = 0.65   # metres (below this = hinge notch / fillet)
_DOOR_ARC_RADIUS_MAX_M = 1.10   # metres (above this = bay / round feature)
_DOOR_ARC_SWEEP_MIN = 60.0      # degrees
_DOOR_ARC_SWEEP_MAX = 105.0     # degrees

# INSUNITS → linear-unit → metres conversion
_INSUNITS_TO_METRES: dict[int, float] = {
    0: 1.0,        # unitless – assume metres
    1: 0.0254,     # inches
    2: 0.3048,     # feet
    4: 0.001,      # millimetres
    5: 0.01,       # centimetres
    6: 1.0,        # metres
    7: 1000.0,     # kilometres (unusual)
}
# Area conversion: drawing_units² → sqft
_INSUNITS_TO_SQFT: dict[int, float] = {
    0: 10.7639,    # unitless (assume metres²)
    1: 1.0 / 144.0,
    2: 1.0,
    4: 0.00001076391,
    5: 0.001076391,
    6: 10.7639,
}


class DXFParser:
    """
    Extracts all geometry required by the CAD pipeline from a single DXF file.

    Returns
    -------
    dict with keys:
        segments    : list[{start, end, layer}]
        texts       : list[{text, position, layer}]
        doors       : list[{center, radius, sweep_deg, width_m, source, layer}]
        blocks      : list[{name, type, position, layer}]
        metadata    : {insunits, to_metres, area_to_sqft, layer_counts,
                       wall_layers, door_layers}
    """

    _ARC_SUBDIVISIONS = 12  # sub-segments per 90° arc (wall arcs only)

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        readfile_fn = getattr(ezdxf, "readfile", None)
        if not callable(readfile_fn):
            raise RuntimeError("ezdxf.readfile is unavailable in this environment")
        dxf_structure_error = getattr(ezdxf, "DXFStructureError", Exception)
        try:
            self._doc: Any = readfile_fn(filepath)
        except (IOError, dxf_structure_error) as exc:
            logger.error("Cannot open DXF file %s: %s", filepath, exc)
            raise
        self._msp: Any = self._doc.modelspace()
        insunits = int(self._doc.header.get("$INSUNITS", 0) or 0)
        self._to_metres: float = _INSUNITS_TO_METRES.get(insunits, 1.0)
        self._area_to_sqft: float = _INSUNITS_TO_SQFT.get(insunits, 10.7639)
        self._insunits = insunits
        self._seen_texts: set[tuple] = set()
        self._visited_blocks: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> dict[str, Any]:
        geometry: dict[str, Any] = {
            "segments": [],
            "texts": [],
            "doors": [],
            "blocks": [],
        }
        layer_counts: dict[str, int] = {}

        self._seen_texts.clear()
        self._visited_blocks.clear()
        self._traverse(self._msp, geometry, layer_counts, ox=0.0, oy=0.0)

        wall_layers = self._identify_wall_layers(layer_counts)
        door_layers = [l for l in layer_counts if self._is_door_layer(l)]

        metadata = {
            "insunits": self._insunits,
            "to_metres": self._to_metres,
            "area_to_sqft": self._area_to_sqft,
            "layer_counts": layer_counts,
            "wall_layers": wall_layers,
            "door_layers": door_layers,
            "candidate_wall_layers": wall_layers,  # compat alias
            "linear_to_feet": self._area_to_sqft ** 0.5,
        }
        geometry["metadata"] = metadata

        logger.info(
            "DXFParser: %d segments, %d texts, %d doors, %d blocks | "
            "wall_layers=%s | file=%s",
            len(geometry["segments"]),
            len(geometry["texts"]),
            len(geometry["doors"]),
            len(geometry["blocks"]),
            wall_layers,
            self.filepath,
        )
        return geometry

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def _traverse(
        self,
        layout: Any,
        geo: dict[str, Any],
        layer_counts: dict[str, int],
        ox: float,
        oy: float,
    ) -> None:
        for entity in layout:
            try:
                self._handle_entity(entity, geo, layer_counts, ox, oy)
            except Exception as exc:
                dxftype_attr = getattr(entity, "dxftype", None)
                etype_name = dxftype_attr() if callable(dxftype_attr) else "?"
                logger.debug("Skip entity %s: %s", etype_name, exc)

    def _handle_entity(
        self,
        entity: Any,
        geo: dict[str, Any],
        layer_counts: dict[str, int],
        ox: float,
        oy: float,
    ) -> None:
        etype = entity.dxftype()
        layer = self._layer(entity)
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

        if etype == "LINE":
            self._handle_line(entity, geo, layer, ox, oy)

        elif etype in ("LWPOLYLINE", "POLYLINE"):
            self._handle_polyline(entity, geo, layer, ox, oy)

        elif etype == "ARC":
            self._handle_arc(entity, geo, layer, ox, oy)

        elif etype == "CIRCLE":
            self._handle_circle(entity, geo, layer, ox, oy)

        elif etype in ("TEXT", "MTEXT"):
            self._handle_text(entity, geo, layer, ox, oy)

        elif etype == "INSERT":
            self._handle_insert(entity, geo, layer_counts, layer, ox, oy)

    # ------------------------------------------------------------------
    # Entity handlers
    # ------------------------------------------------------------------

    def _handle_line(self, e: Any, geo: dict, layer: str, ox: float, oy: float) -> None:
        sx = e.dxf.start.x + ox
        sy = e.dxf.start.y + oy
        ex = e.dxf.end.x + ox
        ey = e.dxf.end.y + oy
        if self._seg_len(sx, sy, ex, ey) < 1e-6:
            return
        geo["segments"].append({"start": [sx, sy], "end": [ex, ey], "layer": layer})

    def _handle_polyline(self, e: Any, geo: dict, layer: str, ox: float, oy: float) -> None:
        etype = e.dxftype()
        try:
            if etype == "LWPOLYLINE":
                pts = [(p[0] + ox, p[1] + oy) for p in e.get_points()]
            else:
                pts = [(v.dxf.location.x + ox, v.dxf.location.y + oy) for v in e.vertices]
            if len(pts) < 2:
                return

            is_closed = getattr(e.dxf, "closed", False) or getattr(e, "is_closed", False)
            # Filter negligibly small closed polygons (hatch noise)
            if is_closed and len(pts) >= 3:
                from shapely.geometry import Polygon as _Poly
                try:
                    if _Poly(pts).area < 0.04:
                        return
                except Exception:
                    pass

            pairs = zip(pts, pts[1:])
            if is_closed:
                pairs = list(pairs) + [(pts[-1], pts[0])]  # type: ignore[assignment]
            for a, b in pairs:
                if self._seg_len(a[0], a[1], b[0], b[1]) < 1e-6:
                    continue
                geo["segments"].append({"start": list(a[:2]), "end": list(b[:2]), "layer": layer})
        except Exception as exc:
            logger.debug("Polyline skip: %s", exc)

    def _handle_arc(self, e: Any, geo: dict, layer: str, ox: float, oy: float) -> None:
        cx = e.dxf.center.x + ox
        cy = e.dxf.center.y + oy
        radius = float(e.dxf.radius)
        start_a = float(e.dxf.start_angle)
        end_a = float(e.dxf.end_angle)
        sweep = (end_a - start_a) % 360.0
        if sweep == 0.0:
            sweep = 360.0

        radius_m = radius * self._to_metres
        is_door = (
            _DOOR_ARC_RADIUS_MIN_M <= radius_m <= _DOOR_ARC_RADIUS_MAX_M
            and _DOOR_ARC_SWEEP_MIN <= sweep <= _DOOR_ARC_SWEEP_MAX
        )

        if is_door:
            # Store door arc – width ≈ radius (chord of a 90° swing)
            geo["doors"].append({
                "center": [cx, cy],
                "radius": radius,
                "radius_m": radius_m,
                "sweep_deg": sweep,
                "width_m": radius_m,
                "source": "arc",
                "layer": layer,
            })
        else:
            # Convert arc to line segments for wall graph
            pts = self._arc_pts(cx, cy, radius, start_a, end_a, sweep)
            for a, b in zip(pts, pts[1:]):
                if self._seg_len(a[0], a[1], b[0], b[1]) < 1e-6:
                    continue
                geo["segments"].append({"start": list(a), "end": list(b), "layer": layer})

    def _handle_circle(self, e: Any, geo: dict, layer: str, ox: float, oy: float) -> None:
        cx = e.dxf.center.x + ox
        cy = e.dxf.center.y + oy
        radius = float(e.dxf.radius)
        # Very small circles → hatch noise or bolt holes
        if radius * self._to_metres < 0.05:
            return
        pts = self._arc_pts(cx, cy, radius, 0.0, 360.0, 360.0, n_override=16)
        for a, b in zip(pts, pts[1:]):
            geo["segments"].append({"start": list(a), "end": list(b), "layer": layer})

    def _handle_text(self, e: Any, geo: dict, layer: str, ox: float, oy: float) -> None:
        try:
            if e.dxftype() == "MTEXT":
                raw = e.plain_mtext() if hasattr(e, "plain_mtext") else e.text
            else:
                raw = e.dxf.text
            text = (raw or "").strip()
            if not text:
                return
            try:
                ins = e.dxf.insert
                x, y = ins.x + ox, ins.y + oy
            except Exception:
                return
            key = (text.upper(), round(x, 2), round(y, 2))
            if key in self._seen_texts:
                return
            self._seen_texts.add(key)
            geo["texts"].append({"text": text, "position": [x, y], "layer": layer})
        except Exception as exc:
            logger.debug("Text skip: %s", exc)

    def _handle_insert(
        self,
        e: Any,
        geo: dict,
        layer_counts: dict[str, int],
        layer: str,
        ox: float,
        oy: float,
    ) -> None:
        block_name = (getattr(e.dxf, "name", None) or "").upper()
        ins = e.dxf.insert
        bx, by = ins.x + ox, ins.y + oy

        # Classify block purpose
        btype = self._classify_block(block_name)

        geo["blocks"].append({
            "name": block_name,
            "type": btype,
            "position": [bx, by],
            "layer": layer,
        })

        # If it's a door block, also emit a door entry
        if btype == "DOOR":
            geo["doors"].append({
                "center": [bx, by],
                "radius": 0.0,
                "radius_m": 0.0,
                "sweep_deg": 90.0,
                "width_m": 0.9,   # assume standard door width
                "source": "block",
                "block_name": block_name,
                "layer": layer,
            })

        # Recurse into block definition
        if block_name not in self._visited_blocks:
            self._visited_blocks.add(block_name)
            try:
                block_def = self._doc.blocks.get(block_name)
                if block_def:
                    self._traverse(block_def, geo, layer_counts, ox=bx, oy=by)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _layer(entity: Any) -> str:
        return (getattr(entity.dxf, "layer", None) or "0").upper().strip()

    @staticmethod
    def _seg_len(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _arc_pts(
        cx: float,
        cy: float,
        r: float,
        start_deg: float,
        end_deg: float,
        sweep: float,
        n_override: int | None = None,
    ) -> list[tuple[float, float]]:
        n = n_override or max(3, int(12 * (sweep / 90.0)))
        pts: list[tuple[float, float]] = []
        for i in range(n + 1):
            a = math.radians(start_deg + sweep * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts

    @staticmethod
    def _classify_block(name: str) -> str:
        uname = name.upper()
        for kw in _DOOR_BLOCK_KEYWORDS:
            if kw in uname:
                return "DOOR"
        for kw, semantic in _FURNITURE_MAP.items():
            if kw in uname:
                return semantic
        return "GENERIC"

    def _is_door_layer(self, layer: str) -> bool:
        ul = layer.upper()
        return any(h in ul for h in _DOOR_LAYER_HINTS)

    def _identify_wall_layers(self, layer_counts: dict[str, int]) -> list[str]:
        """Return layers most likely to contain walls."""
        preferred: list[str] = []
        for layer in layer_counts:
            ul = layer.upper()
            if any(h in ul for h in _WALL_LAYER_HINTS):
                preferred.append(layer)
        if preferred:
            return sorted(preferred, key=lambda l: -layer_counts.get(l, 0))
        # Fallback: busiest layers that are not noise
        ordered = sorted(layer_counts.items(), key=lambda kv: -kv[1])
        return [
            name for name, _ in ordered[:5]
            if not any(ig in name.upper() for ig in _IGNORE_LAYER_KEYWORDS)
        ]
