import logging
from collections import defaultdict
from typing import Iterable

from shapely.geometry import LineString
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


class SegmentCleaner:
    """Preprocess CAD segments for robust room detection.

    Steps:
    1. Layer filtering to remove hatch/annotation noise.
    2. Remove tiny segments.
    3. Node intersections via unary_union.
    4. Snap endpoints and deduplicate segments.
    """

    def __init__(
        self,
        segments: list[dict],
        *,
        allowed_layers: list[str] | None = None,
        ignore_layer_keywords: list[str] | None = None,
        min_len: float = 0.005,
        snap_tol: float = 0.001,
    ):
        self.segments = segments
        # None means "accept everything" (used when all data is on one layer)
        self.allowed_layers = (
            [s.upper() for s in allowed_layers] if allowed_layers is not None else None
        )
        self.ignore_layer_keywords = [
            s.upper()
            for s in (
                ignore_layer_keywords
                or ["HATCH", "GRID", "DIM", "ANNO", "TEXT", "FURN", "FURNITURE"]
            )
        ]
        self.min_len = min_len
        self.snap_tol = snap_tol

    def clean(self) -> list[dict]:
        filtered = self._layer_filter(self.segments)
        lines = self._to_lines(filtered)
        if not lines:
            return []

        noded = unary_union(lines)
        exploded = self._explode_lines(noded)
        deduped = self._snap_and_dedupe(exploded)

        logger.info(
            "SegmentCleaner: %d raw -> %d filtered -> %d cleaned",
            len(self.segments),
            len(filtered),
            len(deduped),
        )
        return deduped

    def _layer_filter(self, segments: list[dict]) -> list[dict]:
        # When allowed_layers is None we accept everything (single-layer files).
        if self.allowed_layers is None:
            return segments

        out: list[dict] = []
        for seg in segments:
            layer = str(seg.get("layer", "") or "").upper()

            if any(k in layer for k in self.ignore_layer_keywords):
                continue

            if self.allowed_layers:
                if not any(a in layer for a in self.allowed_layers):
                    continue

            out.append(seg)
        return out

    def _to_lines(self, segments: list[dict]) -> list[LineString]:
        lines: list[LineString] = []
        for seg in segments:
            try:
                p1 = tuple(seg["start"][:2])
                p2 = tuple(seg["end"][:2])
                ls = LineString([p1, p2])
                if ls.length >= self.min_len:
                    lines.append(ls)
            except Exception:
                continue
        return lines

    def _explode_lines(self, geom: object) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        parts: list[tuple[tuple[float, float], tuple[float, float]]] = []

        def walk(g: object) -> None:
            if hasattr(g, "geoms"):
                for child in getattr(g, "geoms"):
                    walk(child)
                return

            if not hasattr(g, "coords"):
                return

            coords = list(getattr(g, "coords"))
            if len(coords) < 2:
                return
            for i in range(len(coords) - 1):
                a = (float(coords[i][0]), float(coords[i][1]))
                b = (float(coords[i + 1][0]), float(coords[i + 1][1]))
                parts.append((a, b))

        walk(geom)
        return parts

    def _snap_and_dedupe(
        self,
        segments: Iterable[tuple[tuple[float, float], tuple[float, float]]],
    ) -> list[dict]:
        seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
        out: list[dict] = []

        def snap(pt: tuple[float, float]) -> tuple[float, float]:
            return (
                round(pt[0] / self.snap_tol) * self.snap_tol,
                round(pt[1] / self.snap_tol) * self.snap_tol,
            )

        # Optional lightweight collinear merge bucket by orientation.
        buckets: dict[tuple[int, int], list[tuple[tuple[float, float], tuple[float, float]]]] = defaultdict(list)

        for a, b in segments:
            sa = snap(a)
            sb = snap(b)
            if sa == sb:
                continue

            key = (sa, sb) if sa < sb else (sb, sa)
            if key in seen:
                continue
            seen.add(key)

            dx = sb[0] - sa[0]
            dy = sb[1] - sa[1]
            ori = (0, 1) if abs(dx) < abs(dy) else (1, 0)
            buckets[ori].append((sa, sb))

        # Keep deduped edges; merging is intentionally conservative for stability.
        for edges in buckets.values():
            for sa, sb in edges:
                ls = LineString([sa, sb])
                if ls.length >= self.min_len:
                    out.append({"start": [sa[0], sa[1]], "end": [sb[0], sb[1]], "layer": "CLEANED"})

        return out
