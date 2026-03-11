"""
SVG Parser
==========
Extracts line segments from an SVG file, returning them in the same dict
format used by ``DXFParser`` so they can be fed directly into ``WallGraph``.

Supported SVG elements
----------------------
``<line>``       x1, y1, x2, y2 attributes
``<polyline>``   points="x1,y1 x2,y2 ..."
``<polygon>``    same as polyline but auto-closed
``<path>``       d="M L H V Z C Q A ..."  (curves simplified to endpoints)
``<rect>``       decomposed into four line segments

Coordinate convention
---------------------
The SVG produced by ``svg_converter`` wraps geometry in a
``scale(1,-1) translate(0,-H)`` group to map DXF Y-up coordinates into
SVG Y-down space.  This parser detects that transform and automatically
reverses the Y-flip so that returned coordinates are in the original
DXF drawing unit space.

Returned segment format
-----------------------
Each entry is a dict::

    {
        "start":  [x: float, y: float],
        "end":    [x: float, y: float],
        "length": float,      # Euclidean distance in drawing units
        "angle":  float,      # degrees, [−180, 180]; 0 = rightward, CCW positive
        "layer":  str,        # from data-layer attribute; "" if absent
    }

Usage
-----
>>> from app.pipeline.svg_parser import SVGParser
>>> segments = SVGParser().parse("floor_plan.svg")
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

_MIN_SEG_LEN: float = 1e-6


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class SVGParser:
    """
    Parse an SVG file and return a flat list of line segments.

    Parameters
    ----------
    min_seg_len : discard segments shorter than this threshold (drawing units)
    """

    def __init__(self, min_seg_len: float = _MIN_SEG_LEN) -> None:
        self.min_seg_len = min_seg_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, svg_path: str) -> list[dict]:
        """
        Parse the SVG file at *svg_path* and return all extracted segments.

        Parameters
        ----------
        svg_path : path to the SVG file

        Returns
        -------
        List of ``{start, end, length, angle, layer}`` dicts.
        """
        tree = ET.parse(svg_path)
        root = tree.getroot()

        segments: list[dict] = []
        self._walk(root, [], segments)

        # Remove exact duplicates (order-independent)
        seen: set[tuple[float, float, float, float]] = set()
        unique: list[dict] = []
        for seg in segments:
            x1, y1 = seg["start"]
            x2, y2 = seg["end"]
            key = (round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))
            rev = (round(x2, 4), round(y2, 4), round(x1, 4), round(y1, 4))
            if key not in seen and rev not in seen:
                seen.add(key)
                unique.append(seg)

        logger.info("SVGParser: %d segments from %s", len(unique), svg_path)
        return unique

    # ------------------------------------------------------------------
    # Tree walker
    # ------------------------------------------------------------------

    def _walk(
        self,
        elem: ET.Element,
        transform_stack: list[str],
        out: list[dict],
    ) -> None:
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        layer = elem.get("data-layer", "")

        # Accumulate transforms
        own_tf = elem.get("transform", "")
        if own_tf:
            transform_stack = transform_stack + [own_tf]

        y_flip = _detect_y_flip(transform_stack)

        if tag == "line":
            seg = self._parse_line(elem, layer, y_flip)
            if seg:
                out.append(seg)

        elif tag in ("polyline", "polygon"):
            out.extend(self._parse_polyline(elem, layer, y_flip, closed=(tag == "polygon")))

        elif tag == "path":
            out.extend(self._parse_path(elem, layer, y_flip))

        elif tag == "rect":
            out.extend(self._parse_rect(elem, layer, y_flip))

        for child in elem:
            self._walk(child, transform_stack, out)

    # ------------------------------------------------------------------
    # Element parsers
    # ------------------------------------------------------------------

    def _parse_line(self, elem: ET.Element, layer: str, y_flip: bool) -> dict | None:
        try:
            x1 = float(elem.get("x1", 0))
            y1 = float(elem.get("y1", 0))
            x2 = float(elem.get("x2", 0))
            y2 = float(elem.get("y2", 0))
        except (ValueError, TypeError):
            return None
        if y_flip:
            y1, y2 = -y1, -y2
        return self._make_seg(x1, y1, x2, y2, layer)

    def _parse_polyline(
        self,
        elem: ET.Element,
        layer: str,
        y_flip: bool,
        *,
        closed: bool = False,
    ) -> list[dict]:
        pts_str = elem.get("points", "").strip()
        pts = _parse_points_str(pts_str)
        if y_flip:
            pts = [(x, -y) for x, y in pts]
        pairs = list(zip(pts, pts[1:]))
        if closed and len(pts) >= 3:
            pairs.append((pts[-1], pts[0]))
        return [s for p1, p2 in pairs if (s := self._make_seg(p1[0], p1[1], p2[0], p2[1], layer))]

    def _parse_path(self, elem: ET.Element, layer: str, y_flip: bool) -> list[dict]:
        d = elem.get("d", "")
        pts: list[tuple[float, float] | None] = _path_d_to_points(d)
        if y_flip:
            pts = [(p[0], -p[1]) if p is not None else None for p in pts]
        segs: list[dict] = []
        prev: tuple[float, float] | None = None
        for pt in pts:
            if pt is None:
                prev = None
            else:
                if prev is not None:
                    seg = self._make_seg(prev[0], prev[1], pt[0], pt[1], layer)
                    if seg:
                        segs.append(seg)
                prev = pt
        return segs

    def _parse_rect(self, elem: ET.Element, layer: str, y_flip: bool) -> list[dict]:
        try:
            x = float(elem.get("x", 0))
            y = float(elem.get("y", 0))
            w = float(elem.get("width", 0))
            h = float(elem.get("height", 0))
        except (ValueError, TypeError):
            return []
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        if y_flip:
            corners = [(px, -py) for px, py in corners]
        pairs = list(zip(corners, corners[1:])) + [(corners[-1], corners[0])]
        return [s for p1, p2 in pairs if (s := self._make_seg(p1[0], p1[1], p2[0], p2[1], layer))]

    # ------------------------------------------------------------------

    def _make_seg(
        self, x1: float, y1: float, x2: float, y2: float, layer: str
    ) -> dict | None:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < self.min_seg_len:
            return None
        angle = math.degrees(math.atan2(dy, dx))
        return {
            "start":  [round(x1, 6), round(y1, 6)],
            "end":    [round(x2, 6), round(y2, 6)],
            "length": round(length, 6),
            "angle":  round(angle, 4),
            "layer":  layer,
        }


# ---------------------------------------------------------------------------
# Coordinate / transform helpers
# ---------------------------------------------------------------------------


def _detect_y_flip(transform_stack: list[str]) -> bool:
    """Return True if the net effect of the transform stack is a Y-flip."""
    flips = sum(1 for tf in transform_stack if "scale(1,-1)" in tf)
    return flips % 2 == 1


def _parse_points_str(pts_str: str) -> list[tuple[float, float]]:
    """Parse ``points="x1,y1 x2,y2 ..."`` into a list of (x, y) tuples."""
    nums = re.split(r"[\s,]+", pts_str.strip())
    result: list[tuple[float, float]] = []
    for i in range(0, len(nums) - 1, 2):
        try:
            result.append((float(nums[i]), float(nums[i + 1])))
        except (ValueError, IndexError):
            pass
    return result


# ---------------------------------------------------------------------------
# SVG path 'd' attribute parser
# ---------------------------------------------------------------------------


def _path_d_to_points(d: str) -> list[tuple[float, float] | None]:
    """
    Parse an SVG path ``d`` attribute into a sequence of (x, y) waypoints.

    A ``None`` entry is a pen-up marker (sub-path boundary).

    Supported commands: M m L l H h V v Z z C c Q q A a
    Curves (C, Q, A) are simplified: only the endpoint is recorded.
    """
    # Tokenise: single-letter commands OR numbers (including scientific notation)
    tokens: list[str] = re.findall(
        r"[MmLlHhVvZzCcQqAa]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?",
        d,
    )

    result: list[tuple[float, float] | None] = []
    cx = cy = 0.0
    start_x = start_y = 0.0
    cmd = "M"
    idx = 0

    def consume(n: int) -> list[float]:
        nonlocal idx
        vals: list[float] = []
        for _ in range(n):
            if idx < len(tokens):
                try:
                    vals.append(float(tokens[idx]))
                except ValueError:
                    vals.append(0.0)
                idx += 1
        return vals

    while idx < len(tokens):
        t = tokens[idx]
        if re.match(r"[A-Za-z]", t):
            cmd = t
            idx += 1
        # else: implicit command repeat with next numeric argument

        if cmd in ("M", "m"):
            v = consume(2)
            if len(v) < 2:
                break
            if cmd == "M":
                cx, cy = v[0], v[1]
            else:
                cx += v[0];  cy += v[1]
            start_x, start_y = cx, cy
            result.append((cx, cy))
            # Implicit lineto for subsequent coordinate pairs
            cmd = "L" if cmd == "M" else "l"

        elif cmd in ("L", "l"):
            v = consume(2)
            if len(v) < 2:
                break
            if cmd == "L":
                cx, cy = v[0], v[1]
            else:
                cx += v[0];  cy += v[1]
            result.append((cx, cy))

        elif cmd in ("H", "h"):
            v = consume(1)
            if not v:
                break
            cx = v[0] if cmd == "H" else cx + v[0]
            result.append((cx, cy))

        elif cmd in ("V", "v"):
            v = consume(1)
            if not v:
                break
            cy = v[0] if cmd == "V" else cy + v[0]
            result.append((cx, cy))

        elif cmd in ("Z", "z"):
            result.append((start_x, start_y))
            result.append(None)  # pen-up marker
            cx, cy = start_x, start_y

        elif cmd in ("C", "c"):
            # Cubic Bézier: (x1,y1, x2,y2, x,y) – record only endpoint
            v = consume(6)
            if len(v) < 6:
                break
            if cmd == "C":
                cx, cy = v[4], v[5]
            else:
                cx += v[4];  cy += v[5]
            result.append((cx, cy))

        elif cmd in ("Q", "q"):
            # Quadratic Bézier: (x1,y1, x,y)
            v = consume(4)
            if len(v) < 4:
                break
            if cmd == "Q":
                cx, cy = v[2], v[3]
            else:
                cx += v[2];  cy += v[3]
            result.append((cx, cy))

        elif cmd in ("A", "a"):
            # Arc: (rx,ry,x-rotation,large-arc-flag,sweep-flag,x,y)
            v = consume(7)
            if len(v) < 7:
                break
            if cmd == "A":
                cx, cy = v[5], v[6]
            else:
                cx += v[5];  cy += v[6]
            result.append((cx, cy))

        else:
            idx += 1  # skip unknown command

    return result
