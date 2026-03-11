"""
Block Detector
==============
Detects furniture and fixture blocks from INSERT entities and associates
them with room polygons.

Semantic block types and their room-type influence
---------------------------------------------------
BED       → Bedroom
SOFA      → Living Room
DINING    → Dining Room
STOVE     → Kitchen
WC        → Toilet
SINK      → Bathroom
BATH      → Bathroom
SHOWER    → Bathroom
DESK      → Study Room
WARDROBE  → Bedroom
FRIDGE    → Kitchen
TABLE     → Dining Room / Kitchen

Block-to-room scoring
---------------------
Each room accumulates a ``furniture_score`` (0‥1) based on the types
of furniture blocks found inside its polygon.  High-confidence mappings
(e.g. BED → Bedroom) contribute a large weight; generic fixtures
contribute less.
"""

from __future__ import annotations

import logging

from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Furniture type → most-likely room label + confidence weight
# ---------------------------------------------------------------------------
_FURNITURE_HINTS: dict[str, tuple[str, float]] = {
    "BED":      ("Bedroom",    0.9),
    "SOFA":     ("Living Room", 0.7),
    "COUCH":    ("Living Room", 0.7),
    "DINING":   ("Dining Room", 0.85),
    "DINE":     ("Dining Room", 0.8),
    "TABLE":    ("Dining Room", 0.4),   # generic – weaker signal
    "STOVE":    ("Kitchen",    0.95),
    "OVEN":     ("Kitchen",    0.9),
    "KITCHEN":  ("Kitchen",    0.95),
    "WC":       ("Toilet",     0.95),
    "TOILET":   ("Toilet",     0.95),
    "COMMODE":  ("Toilet",     0.9),
    "SINK":     ("Bathroom",   0.6),
    "BASIN":    ("Bathroom",   0.6),
    "BATH":     ("Bathroom",   0.85),
    "TUB":      ("Bathroom",   0.85),
    "SHOWER":   ("Bathroom",   0.8),
    "DESK":     ("Study Room", 0.8),
    "STUDY":    ("Study Room", 0.75),
    "WARDROBE": ("Bedroom",    0.6),
    "ALMIRAH":  ("Bedroom",    0.6),
    "FRIDGE":   ("Kitchen",    0.7),
    "REFRIG":   ("Kitchen",    0.7),
    "WASH":     ("Utility",    0.6),
    "LAUNDRY":  ("Utility",    0.65),
}


def classify_block(block_name: str) -> tuple[str, str, float]:
    """
    Map a block name to (semantic_type, room_suggestion, confidence).

    Returns
    -------
    (semantic_type, room_suggestion, confidence)
        semantic_type  : e.g. "BED", "STOVE", "WC", "GENERIC"
        room_suggestion: e.g. "Bedroom", "Kitchen", "Toilet", ""
        confidence     : 0.0–1.0
    """
    uname = block_name.upper()
    best: tuple[str, float] = ("GENERIC", 0.0)
    best_kw: str = ""

    for kw, (room, conf) in _FURNITURE_HINTS.items():
        if kw in uname and conf > best[1]:
            best = (room, conf)
            best_kw = kw

    if best[1] > 0.0:
        return (best_kw, best[0], best[1])
    return ("GENERIC", "", 0.0)


class BlockDetector:
    """
    Associates INSERT blocks to room polygons.

    Parameters
    ----------
    blocks : list of block dicts from DXFParser
        Each dict: {name, type, position, layer}
    rooms : list of room candidate dicts
        Each must have: polygon (Shapely Polygon)
    """

    def __init__(self, blocks: list[dict], rooms: list[dict]) -> None:
        self.blocks = blocks
        self.rooms = rooms

    # ------------------------------------------------------------------

    def detect(self) -> list[dict]:
        """
        Populate each room with ``furniture`` list and ``block_score`` dict.

        Returns the rooms list with added fields:
            furniture       : list of semantic type strings
            block_room_hint : most-suggested room type from blocks (or "")
            block_confidence: max block confidence for room suggestion
        """
        for room in self.rooms:
            room.setdefault("furniture", [])
            room.setdefault("_block_votes", {})  # {room_label: net_conf}

        for block in self.blocks:
            bpos = block.get("position", [])
            if len(bpos) < 2:
                continue
            bname = block.get("name", "")
            sem_type, room_hint, conf = classify_block(bname)

            pt = Point(bpos[0], bpos[1])
            matched = False
            for room in self.rooms:
                poly: Polygon | None = room.get("polygon")
                if poly is None:
                    continue
                try:
                    if poly.contains(pt):
                        if sem_type != "GENERIC":
                            room["furniture"].append(sem_type)
                        if room_hint:
                            votes = room["_block_votes"]
                            votes[room_hint] = votes.get(room_hint, 0.0) + conf
                        matched = True
                        break
                except Exception:
                    pass
            if not matched:
                logger.debug("Block %s at %s not inside any room", bname, bpos)

        # Summarise votes
        for room in self.rooms:
            votes: dict[str, float] = room.pop("_block_votes", {})
            if votes:
                best_label = max(votes, key=lambda k: votes[k])
                room["block_room_hint"] = best_label
                room["block_confidence"] = round(min(votes[best_label], 1.0), 3)
            else:
                room["block_room_hint"] = ""
                room["block_confidence"] = 0.0

        total_placed = sum(1 for r in self.rooms if r.get("furniture"))
        logger.info(
            "BlockDetector: %d blocks processed; %d/%d rooms have furniture",
            len(self.blocks), total_placed, len(self.rooms),
        )
        return self.rooms
