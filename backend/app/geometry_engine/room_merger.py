import logging

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

logger = logging.getLogger(__name__)

_MAX_MERGE_ITERS = 3  # safety valve


class RoomMerger:
    """Merge fragmented room polygons into larger room candidates.

    Two polygons are merged when:
    1. They overlap or share significant boundary (buffered intersection), OR
    2. Their centroids are close AND they nearly touch.
    """

    def __init__(
        self,
        polygons: list[dict],
        *,
        min_shared_ratio: float = 0.3,
        thin_wall_gap: float = 0.15,
        centroid_max_dist: float = 2.0,
    ):
        self.polygons = polygons
        self.min_shared_ratio = min_shared_ratio
        self.thin_wall_gap = thin_wall_gap
        self.centroid_max_dist = centroid_max_dist

    def _filter_small_holes(self, polygon: Polygon, min_hole_area: float) -> Polygon:
        if not polygon.interiors:
            return polygon
        filtered_interiors = [
            interior for interior in polygon.interiors
            if Polygon(interior).area >= min_hole_area
        ]
        return Polygon(polygon.exterior, filtered_interiors)

    def merge(self) -> list[dict]:
        if not self.polygons:
            return []

        polys = [p["polygon"] for p in self.polygons if p.get("polygon") is not None]
        if not polys:
            return []

        # Iterative merge with a safety ceiling
        for iteration in range(_MAX_MERGE_ITERS):
            merged_polys, did_merge = self._single_pass(polys)
            polys = merged_polys
            if not did_merge:
                break

        out: list[dict] = []
        for poly in polys:
            if not poly.is_valid or poly.area <= 0:
                continue
            c = poly.centroid
            out.append({
                "polygon": poly,
                "area_raw": float(poly.area),
                "centroid": [round(c.x, 3), round(c.y, 3)],
                "coordinates": [(round(x, 4), round(y, 4)) for x, y in poly.exterior.coords],
            })

        logger.info("RoomMerger: %d -> %d polygons", len(self.polygons), len(out))
        return out

    def _single_pass(self, polys: list[Polygon]) -> tuple[list[Polygon], bool]:
        """One merging pass over the polygon list. Returns (result, changed)."""
        n = len(polys)
        used = [False] * n
        changed = False
        result: list[Polygon] = []

        for i in range(n):
            if used[i]:
                continue
            group = [polys[i]]
            used[i] = True

            for j in range(i + 1, n):
                if used[j]:
                    continue
                if self._should_merge(polys[i], polys[j]):
                    group.append(polys[j])
                    used[j] = True
                    changed = True

            merged = unary_union(group)
            if isinstance(merged, (MultiPolygon, GeometryCollection)):
                for g in merged.geoms:
                    if isinstance(g, Polygon):
                        result.append(self._filter_small_holes(g, 0.5))
            elif isinstance(merged, Polygon):
                result.append(self._filter_small_holes(merged, 0.5))

        return result, changed

    def _should_merge(self, p1: Polygon, p2: Polygon) -> bool:
        # Rule 1: Overlap check via fast buffer intersection.
        # Buffer each polygon slightly and check intersection area.
        try:
            buf = 0.01  # very thin buffer
            if p1.buffer(buf).intersects(p2.buffer(buf)):
                overlap = p1.buffer(buf).intersection(p2.buffer(buf)).area
                shorter_peri = min(p1.length, p2.length)
                # Convert overlap area to an approximate "shared length"
                # by dividing by buffer width (rough estimate)
                approx_shared = overlap / max(2 * buf, 1e-9)
                ratio = approx_shared / max(shorter_peri, 1e-9)
                if ratio >= self.min_shared_ratio:
                    return True
        except Exception:
            pass

        # Rule 2: Centroid proximity + near-touching
        c1 = p1.centroid
        c2 = p2.centroid
        centroid_dist = c1.distance(c2)
        try:
            gap = p1.distance(p2)
        except Exception:
            gap = float("inf")

        if centroid_dist <= self.centroid_max_dist and gap <= self.thin_wall_gap:
            return True

        return False
