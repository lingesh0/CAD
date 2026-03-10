"""
Room Classifier
===============
Classifies room polygons into semantic types based on geometric features
when no CAD text labels are available.

Classification uses two signals:
1. **Area** (sqft) — maps to expected room size ranges.
2. **Minimum-rectangle aspect ratio** — elongated shapes → Corridor.

Rules (area in sqft):
    < 25          →  Utility / Shaft
    25–45         →  Toilet / Bathroom
    45–80         →  Kitchen
    80–160        →  Bedroom
    160–300       →  Hall / Living Room
    > 300         →  Hall / Open Area
    aspect > 4.0 AND area > 40 sqft  →  Corridor / Passage
"""

import logging
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

# Ordered from most-specific to least-specific.
_RULES: list[dict] = [
    {"max_area": 25,  "label": "Utility"},
    {"max_area": 45,  "label": "Toilet"},
    {"max_area": 80,  "label": "Kitchen"},
    {"max_area": 160, "label": "Bedroom"},
    {"max_area": 300, "label": "Hall"},
]

# Corridor requires BOTH elongation AND minimum size.
_CORRIDOR_MIN_AREA = 40.0   # sqft
_CORRIDOR_MIN_ASPECT = 4.0


def _min_rect_aspect(polygon: Polygon) -> float:
    """Return the aspect ratio of the minimum-area rotated bounding rectangle.

    This is more accurate than axis-aligned bounds for L-shaped or angled rooms.
    """
    try:
        rect = polygon.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)  # type: ignore[union-attr]
        if len(coords) < 4:
            return 1.0
        # Edges of the rectangle
        e1 = ((coords[1][0] - coords[0][0]) ** 2 + (coords[1][1] - coords[0][1]) ** 2) ** 0.5
        e2 = ((coords[2][0] - coords[1][0]) ** 2 + (coords[2][1] - coords[1][1]) ** 2) ** 0.5
        short = min(e1, e2)
        long = max(e1, e2)
        return long / max(short, 1e-9)
    except Exception:
        # Fallback: axis-aligned bounds
        minx, miny, maxx, maxy = polygon.bounds
        w = maxx - minx
        h = maxy - miny
        short = min(w, h) if min(w, h) > 0 else 1e-9
        return max(w, h) / short


def classify_room(
    polygon: Polygon,
    area_sqft: float,
    door_count: int = 0,
) -> str:
    """Return a semantic label for *polygon* based on its geometry.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The room polygon.
    area_sqft : float
        Pre-computed area in square feet.
    door_count : int
        Number of doors touching this room (0 if unknown).

    Returns
    -------
    str
        Semantic room label.
    """
    aspect = _min_rect_aspect(polygon)

    # Corridor: must be elongated AND large enough (prevents small false positives)
    if aspect >= _CORRIDOR_MIN_ASPECT and area_sqft >= _CORRIDOR_MIN_AREA:
        return "Corridor"

    # Door-count heuristic: rooms with 0 doors and tiny area → likely a shaft/utility
    if door_count == 0 and area_sqft < 25:
        return "Utility"

    for rule in _RULES:
        if area_sqft < rule["max_area"]:
            return rule["label"]

    return "Open Area"


def classify_rooms(
    rooms: list[dict],
    door_counts: dict[int, int] | None = None,
) -> list[dict]:
    """Add a ``classification`` key to each room dict when no label exists.

    Only rooms whose ``original_label`` is *None* are classified;
    rooms that already carry a CAD label keep their existing name.

    Parameters
    ----------
    door_counts : dict, optional
        Maps room index → number of doors touching that room.
    """
    if door_counts is None:
        door_counts = {}

    for idx, room in enumerate(rooms):
        if room.get("original_label"):
            continue  # keep the real label
        poly = room.get("polygon")
        area_sqft = room.get("area_sqft", 0.0)
        if poly is None:
            continue
        dc = door_counts.get(idx, 0)
        room["classification"] = classify_room(poly, area_sqft, door_count=dc)
        room["name"] = room["classification"]
    return rooms


# ---------------------------------------------------------------------------
# Multi-signal scoring
# ---------------------------------------------------------------------------

def _canonical(label: str | None) -> str:
    """Return upper-cased stripped label for comparison."""
    return (label or "").upper().strip()


def classify_room_multi_signal(
    *,
    area_sqft: float,
    polygon=None,
    door_count: int = 0,
    geometry_label: str | None = None,
    vision_label: str | None = None,
    vision_confidence: float = 0.8,
    text_label: str | None = None,
) -> dict:
    """Combine geometry, Vision AI, and text signals into a final room label.

    Signal priority and base confidence:
        text    → 0.92  (CAD-embedded label; highest reliability)
        vision  → 0.80  (Vision AI; boosted if corroborated by geometry)
        geometry→ 0.60  (area/aspect/door heuristics; always available)

    Confidence is boosted when two independent signals agree.

    Returns
    -------
    dict  with keys: ``label``, ``confidence``, ``method``
    """
    # Geometry is always available as a baseline
    if geometry_label is None and polygon is not None:
        geometry_label = classify_room(polygon, area_sqft, door_count=door_count)

    candidates: list[dict] = []

    if text_label:
        conf = 0.92
        method = "text"
        # Agreement bonus
        if vision_label and _canonical(vision_label) == _canonical(text_label):
            conf = 0.97
            method = "text+vision"
        elif geometry_label and _canonical(geometry_label) == _canonical(text_label):
            conf = 0.95
            method = "text+geometry"
        candidates.append({"label": text_label, "confidence": conf, "method": method})

    if vision_label:
        conf = vision_confidence
        method = "vision"
        if geometry_label and _canonical(vision_label) == _canonical(geometry_label):
            conf = min(0.92, conf + 0.10)
            method = "vision+geometry"
        candidates.append({"label": vision_label, "confidence": conf, "method": method})

    if geometry_label:
        candidates.append({
            "label": geometry_label,
            "confidence": 0.60,
            "method": "geometry",
        })

    if not candidates:
        fallback = classify_room(polygon, area_sqft, door_count) if polygon else "Unknown"
        return {"label": fallback, "confidence": 0.50, "method": "geometry"}

    # Return the highest-confidence candidate
    return max(candidates, key=lambda c: c["confidence"])
