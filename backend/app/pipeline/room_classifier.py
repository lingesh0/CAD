"""
Room Classifier – Multi-Signal Scoring Engine
=============================================
Classifies each room polygon using four weighted signals:

    1. Geometry score  (0.35) – area + aspect ratio + compactness + convexity
    2. Block score     (0.35) – furniture / fixture detection
    3. Adjacency score (0.20) – connectivity graph (adjacency rules)
    4. Vision score    (0.10) – Gemini Vision (only when confidence < 0.75)

Geometry heuristics
-------------------
area < 35 sqft         → Toilet / Utility
area 35–65 sqft        → Toilet / Bathroom
area 65–130 sqft       → Bedroom / Kitchen
area 130–250 sqft      → Bedroom / Hall
area > 250 sqft        → Hall / Open Area
aspect > 3.5, area>40  → Corridor
compactness < 0.4      → irregular → Corridor / Hall

Compactness = 4π·area / perimeter²  (1.0 = circle, 0.785 = square)
Convexity   = polygon.area / convex_hull.area  (1.0 = convex)

Adjacency rules
---------------
Bedroom → often adjacent to Hall, Toilet, Bathroom
Kitchen → often adjacent to Dining Room, Hall
Toilet  → often adjacent to Bedroom, Bathroom
Hall    → adjacent to most rooms

Confidence thresholds
 ---------------------
>= 0.75 → deterministic; skip Vision AI
< 0.75  → pass to vision_validator for Gemini confirmation
"""

from __future__ import annotations

import logging
import math
from typing import Any

from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid room labels (canonical)
# ---------------------------------------------------------------------------
VALID_LABELS: set[str] = {
    "Bedroom", "Kitchen", "Hall", "Living Room", "Toilet", "Bathroom",
    "Utility", "Corridor", "Balcony", "Staircase", "Dining Room",
    "Study Room", "Pooja Room", "Store Room", "Open Area",
}

# ---------------------------------------------------------------------------
# Geometry score: area → candidate label + base confidence
# ---------------------------------------------------------------------------
# (max_area_sqft, label, base_confidence)
_AREA_RULES: list[tuple[float, str, float]] = [
    (15,   "Utility",    0.80),
    (35,   "Toilet",     0.82),
    (65,   "Toilet",     0.72),   # could be small bathroom
    (130,  "Bedroom",    0.70),
    (220,  "Bedroom",    0.65),
    (400,  "Hall",       0.68),
    (float("inf"), "Open Area", 0.60),
]

# Corridor: elongated rooms
_CORRIDOR_MIN_AREA = 40.0
_CORRIDOR_MIN_ASPECT = 3.5
_CORRIDOR_CONF = 0.78

# Compactness threshold: below → irregular (hall / corridor / open)
_COMPACTNESS_IRREGULAR = 0.45

# ---------------------------------------------------------------------------
# Block type → (room_label, confidence)  – already in block_detector,
# but we keep a compact lookup here for the scoring merge step.
# ---------------------------------------------------------------------------
_BLOCK_ROOM: dict[str, tuple[str, float]] = {
    "BED":      ("Bedroom",    0.90),
    "SOFA":     ("Living Room", 0.70),
    "COUCH":    ("Living Room", 0.70),
    "DINING":   ("Dining Room", 0.85),
    "DINE":     ("Dining Room", 0.80),
    "TABLE":    ("Dining Room", 0.40),
    "STOVE":    ("Kitchen",    0.95),
    "OVEN":     ("Kitchen",    0.90),
    "KITCHEN":  ("Kitchen",    0.95),
    "WC":       ("Toilet",     0.95),
    "TOILET":   ("Toilet",     0.95),
    "COMMODE":  ("Toilet",     0.90),
    "SINK":     ("Bathroom",   0.60),
    "BASIN":    ("Bathroom",   0.60),
    "BATH":     ("Bathroom",   0.85),
    "TUB":      ("Bathroom",   0.85),
    "SHOWER":   ("Bathroom",   0.80),
    "DESK":     ("Study Room", 0.80),
    "STUDY":    ("Study Room", 0.75),
    "WARDROBE": ("Bedroom",    0.60),
    "ALMIRAH":  ("Bedroom",    0.60),
    "FRIDGE":   ("Kitchen",    0.70),
    "REFRIG":   ("Kitchen",    0.70),
    "WASH":     ("Utility",    0.60),
}

# ---------------------------------------------------------------------------
# Adjacency rules: {room_label: [labels that commonly appear next to it]}
# ---------------------------------------------------------------------------
_ADJACENCY_RULES: dict[str, list[str]] = {
    "Bedroom":    ["Hall", "Toilet", "Bathroom", "Study Room"],
    "Kitchen":    ["Dining Room", "Hall", "Utility"],
    "Toilet":     ["Bedroom", "Bathroom", "Hall"],
    "Bathroom":   ["Bedroom", "Toilet", "Hall"],
    "Hall":       ["Bedroom", "Kitchen", "Dining Room", "Living Room",
                   "Toilet", "Bathroom", "Balcony", "Staircase"],
    "Living Room":["Hall", "Dining Room", "Balcony", "Study Room"],
    "Dining Room":["Kitchen", "Hall", "Living Room"],
    "Study Room": ["Bedroom", "Hall"],
    "Utility":    ["Kitchen", "Bathroom"],
    "Balcony":    ["Bedroom", "Living Room", "Hall"],
    "Staircase":  ["Hall", "Corridor"],
    "Pooja Room": ["Hall"],
    "Store Room": ["Kitchen", "Utility"],
}

# Signal weights
_W_GEO  = 0.35
_W_BLOCK = 0.35
_W_ADJ  = 0.20
_W_VIS  = 0.10

# Vision AI trigger threshold
_VISION_TRIGGER = 0.75


# ---------------------------------------------------------------------------
# Geometry feature helpers
# ---------------------------------------------------------------------------

def _aspect_ratio(poly: Polygon) -> float:
    try:
        rect_geom = poly.minimum_rotated_rectangle
        if not isinstance(rect_geom, Polygon):
            return 1.0
        coords = list(rect_geom.exterior.coords)
        if len(coords) < 4:
            return 1.0
        e1 = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        e2 = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
        short, long_ = min(e1, e2), max(e1, e2)
        return long_ / max(short, 1e-9)
    except Exception:
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny
        short = min(w, h) if min(w, h) > 1e-9 else 1e-9
        return max(w, h) / short


def _compactness(poly: Polygon) -> float:
    """Isoperimetric quotient: 4π·area / perimeter²."""
    p = poly.length
    if p < 1e-9:
        return 1.0
    return min(1.0, 4 * math.pi * poly.area / (p * p))


def _convexity(poly: Polygon) -> float:
    """area / convex-hull area."""
    try:
        ch = poly.convex_hull
        if ch.area < 1e-9:
            return 1.0
        return min(1.0, poly.area / ch.area)
    except Exception:
        return 1.0


def compute_geometry_features(poly: Polygon, area_sqft: float) -> dict[str, float]:
    return {
        "area_sqft":   area_sqft,
        "aspect":      _aspect_ratio(poly),
        "compactness": _compactness(poly),
        "convexity":   _convexity(poly),
    }


# ---------------------------------------------------------------------------
# Single-room geometry classification
# ---------------------------------------------------------------------------

def _geometry_score(features: dict[str, float]) -> tuple[str, float]:
    """Return (label, confidence) from geometry features alone."""
    area   = features["area_sqft"]
    aspect = features["aspect"]
    comp   = features["compactness"]

    # Corridor: elongated + large enough
    if aspect >= _CORRIDOR_MIN_ASPECT and area >= _CORRIDOR_MIN_AREA:
        return "Corridor", _CORRIDOR_CONF

    # Irregular shape (low compactness) → hall or open area
    if comp < _COMPACTNESS_IRREGULAR and area > 80:
        return "Hall", 0.55

    for max_a, label, conf in _AREA_RULES:
        if area < max_a:
            return label, conf

    return "Open Area", 0.55


# ---------------------------------------------------------------------------
# Multi-signal scoring engine
# ---------------------------------------------------------------------------

class RoomClassifier:
    """
    Classify a list of rooms using geometry, blocks, adjacency, and vision.

    Parameters
    ----------
    rooms : list of room dicts
        Must have: polygon, area_sqft.
        Optional: original_label (from CAD text), furniture, block_room_hint,
                  block_confidence, adjacent_rooms.
    adjacency_pairs : list of (i, j) room-index pairs that share a door
    vision_results : dict {room_index: {room_type, confidence}} from Gemini
                     (populated after a deferred vision call)
    area_to_sqft : float  drawing-unit² → sqft conversion
    """

    def __init__(
        self,
        rooms: list[dict],
        adjacency_pairs: list[tuple[int, int]] | None = None,
        vision_results: dict[int, dict] | None = None,
        *,
        area_to_sqft: float = 1.0,
    ) -> None:
        self.rooms = rooms
        self.adjacency_pairs = adjacency_pairs or []
        self.vision_results = vision_results or {}
        self.area_to_sqft = area_to_sqft

        # Build adjacency map {room_idx: set of neighbor idxs}
        self._adj_map: dict[int, set[int]] = {i: set() for i in range(len(rooms))}
        for a, b in self.adjacency_pairs:
            self._adj_map.setdefault(a, set()).add(b)
            self._adj_map.setdefault(b, set()).add(a)

    # ------------------------------------------------------------------

    def classify_all(self) -> list[dict]:
        """
        Classify every room and populate:
            name, classification, confidence, classification_method,
            needs_vision (bool)

        Returns rooms list with fields populated.
        """
        for idx, room in enumerate(self.rooms):
            self._classify_room(idx, room)
        return self.rooms

    def needs_vision(self) -> list[int]:
        """Return list of room indices where confidence < _VISION_TRIGGER."""
        return [
            i for i, r in enumerate(self.rooms)
            if not r.get("original_label") and r.get("confidence", 0.0) < _VISION_TRIGGER
        ]

    def apply_vision(self, vision_results: dict[int, dict]) -> None:
        """Re-classify rooms that received Gemini Vision results."""
        self.vision_results = vision_results
        for idx in vision_results:
            if 0 <= idx < len(self.rooms):
                self._classify_room(idx, self.rooms[idx])

    # ------------------------------------------------------------------

    def _classify_room(self, idx: int, room: dict) -> None:
        poly: Polygon | None = room.get("polygon")
        area_sqft = float(room.get("area_sqft", 0.0) or 0.0)

        # ── 1. CAD text label → highest priority ──────────────────────
        orig = (room.get("original_label") or "").strip()
        if orig:
            norm = _normalize_label(orig)
            room["name"] = norm
            room["classification"] = norm
            room["confidence"] = 0.90
            room["classification_method"] = "cad_label"
            return

        # ── 2. Geometry score ──────────────────────────────────────────
        if poly is not None:
            features = compute_geometry_features(poly, area_sqft)
        else:
            features = {"area_sqft": area_sqft, "aspect": 1.0,
                        "compactness": 1.0, "convexity": 1.0}
        geo_label, geo_conf = _geometry_score(features)

        # ── 3. Block score ─────────────────────────────────────────────
        block_label: str = room.get("block_room_hint", "") or ""
        block_conf: float = float(room.get("block_confidence", 0.0) or 0.0)

        # Also check furniture list directly
        furniture = room.get("furniture", []) or []
        for ftype in furniture:
            if ftype in _BLOCK_ROOM:
                blabel, bconf = _BLOCK_ROOM[ftype]
                if bconf > block_conf:
                    block_label = blabel
                    block_conf = bconf

        # ── 4. Adjacency score ─────────────────────────────────────────
        adj_label, adj_conf = self._adjacency_score(idx, geo_label, block_label)

        # ── 5. Vision score ────────────────────────────────────────────
        vis_label: str = ""
        vis_conf: float = 0.0
        if idx in self.vision_results:
            vr = self.vision_results[idx]
            vis_label = vr.get("room_type", "")
            vis_conf = float(vr.get("confidence", 0.0) or 0.0)

        # ── 6. Weighted vote ───────────────────────────────────────────
        votes: dict[str, float] = {}
        _add_vote(votes, geo_label,   geo_conf   * _W_GEO)
        _add_vote(votes, block_label, block_conf * _W_BLOCK)
        _add_vote(votes, adj_label,   adj_conf   * _W_ADJ)
        _add_vote(votes, vis_label,   vis_conf   * _W_VIS)

        if votes:
            final_label = max(votes, key=lambda k: votes[k])
            raw_conf = votes[final_label]
        else:
            final_label = geo_label
            raw_conf = geo_conf * _W_GEO

        # Normalise to [0, 1]
        total_w = _W_GEO + (block_conf > 0) * _W_BLOCK + (adj_conf > 0) * _W_ADJ + (vis_conf > 0) * _W_VIS
        final_conf = min(1.0, raw_conf / max(total_w, 1e-9)) if total_w > 0 else raw_conf

        # Determine method string
        if vis_label and vis_conf > 0:
            method = "vision+rules"
        elif block_label and block_conf > 0:
            method = "rules+blocks"
        else:
            method = "rules"

        # Raster-confirmation provides additional evidence → small confidence boost
        if room.get("raster_confirmed"):
            final_conf = min(1.0, final_conf + 0.05)
            if method == "rules":
                method = "rules+raster"

        room["name"] = final_label
        room["classification"] = final_label
        room["confidence"] = round(final_conf, 3)
        room["classification_method"] = method

    def _adjacency_score(
        self,
        idx: int,
        geo_label: str,
        block_label: str,
    ) -> tuple[str, float]:
        """Return (suggested_label, confidence) from adjacency context."""
        neighbors = self._adj_map.get(idx, set())
        if not neighbors:
            return "", 0.0

        neighbor_labels: list[str] = []
        for ni in neighbors:
            if 0 <= ni < len(self.rooms):
                nr = self.rooms[ni]
                nl = nr.get("name") or nr.get("classification") or ""
                if nl:
                    neighbor_labels.append(nl)

        if not neighbor_labels:
            return "", 0.0

        # Check if geo_label is consistent with any adjacency rule
        candidate = block_label or geo_label
        rules = _ADJACENCY_RULES.get(candidate, [])
        matching = [n for n in neighbor_labels if n in rules]
        if matching:
            # Reinforce candidate label
            adj_conf = min(0.85, 0.4 + 0.15 * len(matching))
            return candidate, adj_conf

        # No direct rule match → look for reverse rule
        for nl in neighbor_labels:
            for label, nbrs in _ADJACENCY_RULES.items():
                if candidate in nbrs and nl == label:
                    return candidate, 0.50

        return "", 0.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _add_vote(votes: dict[str, float], label: str, weight: float) -> None:
    if label and weight > 0:
        votes[label] = votes.get(label, 0.0) + weight


# Label normalisation table (common DXF text → canonical label)
_LABEL_MAP: dict[str, str] = {
    # Bedroom variants
    "BED":          "Bedroom",
    "BEDROOM":      "Bedroom",
    "BED ROOM":     "Bedroom",
    "MASTER BED":   "Bedroom",
    "MASTER BEDROOM": "Bedroom",
    "GUEST ROOM":   "Bedroom",
    # Kitchen variants
    "KIT":          "Kitchen",
    "KITCHEN":      "Kitchen",
    "KITCHENETTE":  "Kitchen",
    # Hall / Living variants
    "HALL":         "Hall",
    "LOBBY":        "Hall",
    "FOYER":        "Hall",
    "LOUNGE":       "Living Room",
    "DRAWING":      "Living Room",
    "DRAWING ROOM": "Living Room",
    "LIVING":       "Living Room",
    "LIVING ROOM":  "Living Room",
    "LIV":          "Living Room",
    "FAMILY":       "Living Room",
    "FAMILY ROOM":  "Living Room",
    # Dining variants
    "DINING":       "Dining Room",
    "DINING ROOM":  "Dining Room",
    "DINE":         "Dining Room",
    # Toilet / Bathroom
    "TOILET":       "Toilet",
    "WC":           "Toilet",
    "WASHROOM":     "Toilet",
    "WATER CLOSET": "Toilet",
    "LAVATORY":     "Toilet",
    "BATH":         "Bathroom",
    "BATHROOM":     "Bathroom",
    "BATH ROOM":    "Bathroom",
    "ENSUITE":      "Bathroom",
    # Utility / Store
    "UTILITY":      "Utility",
    "UTIL":         "Utility",
    "LAUNDRY":      "Utility",
    "STORE":        "Store Room",
    "STORE ROOM":   "Store Room",
    "STORAGE":      "Store Room",
    "PANTRY":       "Store Room",
    # Circulation
    "BALC":         "Balcony",
    "BALCONY":      "Balcony",
    "TERRACE":      "Balcony",
    "VERANDAH":     "Balcony",
    "CORRIDOR":     "Corridor",
    "PASSAGE":      "Corridor",
    "PASSAGEWAY":   "Corridor",
    "GALLERY":      "Corridor",
    "STAIR":        "Staircase",
    "STAIRCASE":    "Staircase",
    "STAIRWAY":     "Staircase",
    # Special
    "POOJA":        "Pooja Room",
    "PRAYER":       "Pooja Room",
    "PUJA":         "Pooja Room",
    "STUDY":        "Study Room",
    "OFFICE":       "Study Room",
    "OPEN":         "Open Area",
    "COURTYARD":    "Open Area",
    "ATRIUM":       "Open Area",
}


def _normalize_label(raw: str) -> str:
    """Convert a raw CAD text label to a canonical room type."""
    upper = raw.upper().strip()
    # Exact match
    if upper in _LABEL_MAP:
        return _LABEL_MAP[upper]
    # Prefix/substring match
    for key, val in _LABEL_MAP.items():
        if upper.startswith(key) or key in upper:
            return val
    # Title-case as fallback if it's a known valid label substring
    title = raw.title().strip()
    for valid in VALID_LABELS:
        if title in valid or valid in title:
            return valid
    return title
