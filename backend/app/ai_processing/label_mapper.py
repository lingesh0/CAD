"""Map Vision-AI labels to room centroids.

The LLM returns either:
  (a) room_index  (new indexed format): direct 1-to-1 assignment by index
  (b) approximate_location (legacy 3x3 grid): greedy nearest-centroid match

Vision labels are stored as ``vision_label`` and ``vision_confidence``.
The caller is responsible for running the multi-signal scorer afterward
to decide the final ``name``.
"""
import math
from typing import List


# 3×3 grid → normalised (x, y) where (0,0)=bottom-left, (1,1)=top-right
_GRID_XY = {
    "top-left":      (0.0, 1.0),
    "top-center":    (0.5, 1.0),
    "top-right":     (1.0, 1.0),
    "center-left":   (0.0, 0.5),
    "center":        (0.5, 0.5),
    "center-right":  (1.0, 0.5),
    "bottom-left":   (0.0, 0.0),
    "bottom-center": (0.5, 0.0),
    "bottom-right":  (1.0, 0.0),
}


def _bbox(rooms: list[dict]) -> tuple[float, float, float, float]:
    xs = [r["centroid"][0] for r in rooms]
    ys = [r["centroid"][1] for r in rooms]
    return min(xs), min(ys), max(xs), max(ys)


def _label_to_xy(label: dict, bbox: tuple) -> tuple[float, float]:
    loc = label.get("approximate_location", "center").lower().strip()
    nx, ny = _GRID_XY.get(loc, (0.5, 0.5))
    min_x, min_y, max_x, max_y = bbox
    w = max_x - min_x or 1.0
    h = max_y - min_y or 1.0
    return (min_x + nx * w, min_y + ny * h)


def map_labels_to_rooms(
    rooms: List[dict],
    labels: List[dict],
) -> List[dict]:
    """Assign Vision-AI labels to rooms; store as ``vision_label`` / ``vision_confidence``.

    Uses direct index mapping when the response contains ``room_index``,
    otherwise falls back to greedy nearest-centroid using ``approximate_location``.
    The ``name`` field is also updated for backward compatibility.
    """
    if not labels or not rooms:
        return rooms

    # ── Path A: indexed response (new prompt format) ──────────────────
    if "room_index" in labels[0]:
        for label in labels:
            idx = label.get("room_index", 0) - 1  # 1-based → 0-based
            if 0 <= idx < len(rooms):
                rtype = label.get("room_type", rooms[idx]["name"])
                conf = label.get("confidence", 0.80)
                rooms[idx]["vision_label"] = rtype
                rooms[idx]["vision_confidence"] = conf
                rooms[idx]["name"] = rtype          # apply immediately
                rooms[idx]["confidence"] = conf
        return rooms

    # ── Path B: grid-based fallback (legacy approximate_location) ─────
    bbox = _bbox(rooms)
    pairs: list[tuple[int, int, float]] = []
    for li, label in enumerate(labels):
        lx, ly = _label_to_xy(label, bbox)
        for ri, room in enumerate(rooms):
            cx, cy = room["centroid"]
            d = math.hypot(lx - cx, ly - cy)
            pairs.append((li, ri, d))

    pairs.sort(key=lambda t: t[2])
    used_labels: set[int] = set()
    used_rooms: set[int] = set()

    for li, ri, _dist in pairs:
        if li in used_labels or ri in used_rooms:
            continue
        rtype = labels[li].get("room_type", rooms[ri]["name"])
        conf = labels[li].get("confidence", 0.80)
        rooms[ri]["vision_label"] = rtype
        rooms[ri]["vision_confidence"] = conf
        rooms[ri]["name"] = rtype
        rooms[ri]["confidence"] = conf
        used_labels.add(li)
        used_rooms.add(ri)

    return rooms