"""
Raster Segmenter
================
Detects room polygons by rendering the floor plan as a binary raster image
and applying OpenCV image-segmentation techniques.

Algorithm
---------
1.  **Render** wall segments onto a high-resolution binary image (numpy/cv2
    ``line``).  The image resolution is scaled so the longer drawing axis maps
    to ``img_size`` pixels; the pixel ↔ drawing-unit transform is tracked
    precisely for later coordinate inversion.
2.  **Dilate** the wall image slightly (``MORPH_DILATE``) to connect
    near-miss endpoint gaps that ``DoorGapCloser`` may not have bridged.
3.  **Morphological closing** (``MORPH_CLOSE``) seals remaining micro-gaps.
4.  **Flood-fill** the inverted (interior) image from the image border to
    mark the exterior region.
5.  Extract the **interior mask** (pixels not reached by the border flood).
6.  Find **contours** of interior connected components.
7.  **Simplify** each contour with ``approxPolyDP`` and convert pixel
    coordinates back to drawing units using the stored transform.
8.  Filter by minimum area, compactness (``≥ 0.08``), and aspect ratio
    (``≤ 8``).

Returns
-------
``list[dict]`` with the same schema as ``WallGraph.extract_polygons()``::

    {
        "polygon":     Shapely Polygon (drawing units),
        "area_raw":    float,
        "area_sqft":   float,
        "centroid":    [cx, cy],
        "coordinates": [(x, y), …],
        "source":      "raster",
    }

The ``source`` key is added so downstream code can distinguish raster
candidates from vector-graph candidates.

Dependency
----------
Requires ``opencv-python-headless`` (cv2).  If cv2 is not importable the
method returns an empty list with a warning and the pipeline continues using
only the vector path.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Default image resolution (pixels along the longer axis)
_IMG_SIZE: int = 1024

# Wall stroke width in pixels when drawing the binary image
_STROKE_PX: int = 2

# Morphological kernel size (pixels)
_MORPH_K: int = 3

# Minimum compactness ratio (4π·area / perimeter²) to keep a contour
_MIN_COMPACTNESS: float = 0.08

# Maximum aspect ratio of the bounding box to keep (eliminates thin strips)
_MAX_ASPECT: float = 8.0


@dataclass
class RasterSegmenter:
    """
    Raster-based room segmentation using OpenCV.

    Parameters
    ----------
    min_room_area_sqft : rooms smaller than this are discarded.
    area_to_sqft       : drawing-unit² → sqft conversion factor.
    to_metres          : drawing-unit → metres (currently unused; stored for
                         future calibration use).
    img_size           : target image resolution in pixels (longer axis).
    stroke_px          : wall thickness in pixels in the binary render.
    """

    min_room_area_sqft: float = 8.0
    area_to_sqft: float = 1.0
    to_metres: float = 1.0
    img_size: int = _IMG_SIZE
    stroke_px: int = _STROKE_PX

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_rooms(self, segments: list[dict]) -> list[dict]:
        """
        Run raster segmentation on the provided wall segments.

        Parameters
        ----------
        segments : list of ``{start, end, layer}`` dicts from ``DXFParser``.

        Returns
        -------
        List of room candidate dicts, or ``[]`` if cv2 is unavailable.
        """
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "RasterSegmenter: opencv-python-headless not installed – "
                "raster segmentation skipped."
            )
            return []

        if not segments:
            return []

        # ── Step 1: compute drawing bounds ─────────────────────────────
        min_x, min_y, max_x, max_y = _bounds(segments)
        dw = max(max_x - min_x, 1e-9)
        dh = max(max_y - min_y, 1e-9)
        scale = self.img_size / max(dw, dh)
        img_w = max(int(math.ceil(dw * scale)), 4)
        img_h = max(int(math.ceil(dh * scale)), 4)

        # ── Step 2: render binary wall image ───────────────────────────
        img = np.zeros((img_h, img_w), dtype=np.uint8)
        for seg in segments:
            px1, py1 = _to_px(
                float(seg["start"][0]), float(seg["start"][1]),
                min_x, min_y, dw, dh, img_w, img_h,
            )
            px2, py2 = _to_px(
                float(seg["end"][0]), float(seg["end"][1]),
                min_x, min_y, dw, dh, img_w, img_h,
            )
            cv2.line(img, (px1, py1), (px2, py2), 255, self.stroke_px)

        # ── Step 3: dilate to connect near-miss endpoints ──────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_MORPH_K, _MORPH_K))
        img = cv2.dilate(img, kernel, iterations=1)

        # ── Step 4: morphological close (seal micro-gaps) ──────────────
        img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

        # ── Step 5: invert; flood-fill exterior from border ────────────
        inverted = cv2.bitwise_not(img_closed)
        flood = inverted.copy()
        mask = np.zeros((img_h + 2, img_w + 2), dtype=np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 128)

        # Interior = pixels that were white (255) before flood but not reached
        interior = np.where(flood == 255, np.uint8(255), np.uint8(0))

        # ── Step 6: find contours ──────────────────────────────────────
        contours, _ = cv2.findContours(
            interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ── Step 7: convert contours → Shapely polygons ───────────────
        rooms: list[dict] = []
        min_area_raw = self.min_room_area_sqft / max(self.area_to_sqft, 1e-9)
        # Minimum pixel area (at least 4×4 pixels)
        min_px_area = max(16.0, (self.stroke_px * 3) ** 2)

        for cnt in contours:
            if cv2.contourArea(cnt) < min_px_area:
                continue

            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue

            # Convert pixels → drawing units
            pts_draw = [
                _from_px(
                    float(pt[0][0]), float(pt[0][1]),
                    min_x, min_y, dw, dh, img_w, img_h,
                )
                for pt in approx
            ]

            try:
                from shapely.geometry import Polygon  # type: ignore[import-untyped]

                poly = Polygon(pts_draw)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty or poly.area < min_area_raw:
                    continue

                # Compactness filter (reject thin corridors / wall artifacts)
                perimeter = poly.length
                compactness = (
                    4.0 * math.pi * poly.area / (perimeter * perimeter)
                    if perimeter > 0
                    else 0.0
                )
                if compactness < _MIN_COMPACTNESS:
                    continue

                # Aspect ratio filter
                bounds = poly.bounds  # (minx, miny, maxx, maxy)
                bw = max(bounds[2] - bounds[0], 1e-9)
                bh = max(bounds[3] - bounds[1], 1e-9)
                aspect = max(bw, bh) / min(bw, bh)
                if aspect > _MAX_ASPECT:
                    continue

                c = poly.centroid
                area_sqft = round(float(poly.area) * self.area_to_sqft, 2)
                rooms.append(
                    {
                        "polygon":     poly,
                        "area_raw":    round(float(poly.area), 4),
                        "area_sqft":   area_sqft,
                        "centroid":    [round(c.x, 2), round(c.y, 2)],
                        "coordinates": [
                            (round(x, 4), round(y, 4))
                            for x, y in poly.exterior.coords
                        ],
                        "source": "raster",
                    }
                )

            except Exception as exc:
                logger.debug("RasterSegmenter polygon error: %s", exc)

        logger.info(
            "RasterSegmenter: %d raw contours → %d valid rooms (img=%dx%d, segs=%d)",
            len(contours),
            len(rooms),
            img_w,
            img_h,
            len(segments),
        )
        return rooms


# ---------------------------------------------------------------------------
# Coordinate transform helpers (module-level for performance)
# ---------------------------------------------------------------------------


def _bounds(segments: list[dict]) -> tuple[float, float, float, float]:
    xs = [float(s["start"][0]) for s in segments] + [float(s["end"][0]) for s in segments]
    ys = [float(s["start"][1]) for s in segments] + [float(s["end"][1]) for s in segments]
    return min(xs), min(ys), max(xs), max(ys)


def _to_px(
    x: float, y: float,
    min_x: float, min_y: float,
    dw: float, dh: float,
    img_w: int, img_h: int,
) -> tuple[int, int]:
    """Map drawing coordinates to integer pixel coordinates (Y-flipped)."""
    px = int((x - min_x) / dw * (img_w - 1))
    py = int(img_h - 1 - (y - min_y) / dh * (img_h - 1))
    return (
        max(0, min(img_w - 1, px)),
        max(0, min(img_h - 1, py)),
    )


def _from_px(
    px: float, py: float,
    min_x: float, min_y: float,
    dw: float, dh: float,
    img_w: int, img_h: int,
) -> tuple[float, float]:
    """Map pixel coordinates back to drawing units (Y-flipped)."""
    x = min_x + (px / max(img_w - 1, 1)) * dw
    y = min_y + (1.0 - py / max(img_h - 1, 1)) * dh
    return (x, y)
