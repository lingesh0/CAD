"""
Polygon Detector
================
Extracts closed polygons (candidate rooms) from a noded planar geometry
produced by ``WallGraphBuilder``.

Algorithm
---------
1.  ``shapely.ops.polygonize`` extracts every minimal closed polygon
    from the noded edge set.
2.  Filter out polygons that are too small (wall-thickness pockets,
    annotation artifacts) via an area threshold.
3.  Optionally remove the single largest polygon (outer building
    boundary) so only interior rooms remain.
4.  Compute area and centroid for each surviving polygon.
"""

import logging
import time
from typing import Any, List

from shapely.geometry import Polygon
from shapely.ops import polygonize

logger = logging.getLogger(__name__)


class PolygonDetector:
    """Detects room-candidate polygons from a noded planar geometry."""

    def __init__(
        self,
        noded_geometry: Any,
        cycles: list[list[tuple[float, float]]] | None = None,
        *,
        area_to_sqft: float = 1.0,
        min_room_area_sqft: float = 30.0,
        min_bath_area_sqft: float = 10.0,
        remove_outer: bool = True,
    ):
        self.noded = noded_geometry
        self.cycles = cycles or []
        self.area_to_sqft = area_to_sqft
        self.min_room_area_sqft = min_room_area_sqft
        self.min_bath_area_sqft = min_bath_area_sqft
        self.remove_outer = remove_outer

    def detect(self) -> List[dict]:
        """Return list of ``{polygon, area, centroid}`` dicts."""
        t0 = time.perf_counter()

        raw_polys, from_cycles = self._extract_raw_polygons()
        logger.info("Polygonize produced %d raw polygons", len(raw_polys))

        min_raw_area = self.min_bath_area_sqft / max(self.area_to_sqft, 1e-9)
        candidates = [p for p in raw_polys if p.is_valid and p.area >= min_raw_area]

        # --- remove outer boundary (largest polygon) ---
        if self.remove_outer and candidates:
            candidates.sort(key=lambda p: p.area)
            largest = candidates[-1]
            # Only remove if it's significantly larger than the rest
            if len(candidates) > 1 and largest.area > candidates[-2].area * 1.8:
                candidates = candidates[:-1]

        # --- remove containment-redundant polygons ---
        # Only needed for cycle-basis results which can overlap;
        # polygonize already produces non-overlapping minimal polygons.
        if from_cycles:
            candidates = self._remove_containing_polygons(candidates)

        results: list[dict] = []
        for poly in candidates:
            c = poly.centroid
            area_raw = float(poly.area)
            area_sqft = area_raw * self.area_to_sqft
            if area_sqft < self.min_bath_area_sqft:
                continue
            results.append({
                "polygon": poly,
                "area_raw": round(area_raw, 4),
                "area_sqft": round(area_sqft, 2),
                "centroid": [round(c.x, 2), round(c.y, 2)],
                "coordinates": [(round(x, 4), round(y, 4)) for x, y in poly.exterior.coords],
            })

        elapsed = time.perf_counter() - t0
        logger.info(
            "PolygonDetector: %d raw → %d room candidates (%.2fs)",
            len(raw_polys), len(results), elapsed,
        )
        return results

    def _extract_raw_polygons(self) -> tuple[list[Polygon], bool]:
        """Return (polygons, from_cycles) where from_cycles indicates the source."""
        noded_polys = list(polygonize(self.noded))

        # Prefer polygonize — it produces non-overlapping minimal polygons.
        if noded_polys:
            return noded_polys, False

        # Fallback to cycle basis only when polygonize returns nothing.
        cycle_polys: list[Polygon] = []
        if self.cycles:
            for cycle in self.cycles:
                if len(cycle) < 3:
                    continue
                coords = [(float(x), float(y)) for x, y in cycle]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                poly = Polygon(coords)
                if poly.is_valid and poly.area > 0:
                    cycle_polys.append(poly)

        return cycle_polys, True

    @staticmethod
    def _remove_containing_polygons(polys: list[Polygon]) -> list[Polygon]:
        """Remove polygons that contain the centroid of at least one other polygon.

        These are composite super-regions (e.g. a cycle that wraps around
        multiple rooms) and should be discarded so only the minimal
        room-level polygons remain.
        """
        if len(polys) <= 1:
            return polys

        centroids = [p.centroid for p in polys]
        keep = []
        for i, big in enumerate(polys):
            contains_other = False
            for j, small in enumerate(polys):
                if i == j:
                    continue
                # A polygon that contains another polygon's centroid is a superset
                if big.contains(centroids[j]) and big.area > small.area * 1.1:
                    contains_other = True
                    break
            if not contains_other:
                keep.append(big)
        logger.info(
            "Containment filter: %d → %d polygons", len(polys), len(keep),
        )
        return keep
