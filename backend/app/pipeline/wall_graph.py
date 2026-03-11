"""
Wall Graph
==========
Converts raw DXF segments into a planar, noded graph and extracts room
boundary polygons.

Algorithm
---------
1.  Layer-filter segments: keep only wall/structural layers; ignore
    dimension/annotation/grid noise.
2.  Gap closing: small gaps between wall endpoints (≤ door-width threshold)
    are bridged with synthetic segments so the graph forms closed cycles.
3.  Node intersections via ``shapely.ops.unary_union`` (splits every crossing
    pair at every intersection so no two edges share interior points).
4.  Polygon extraction via ``shapely.ops.polygonize``.
5.  Graph cycle detection via ``networkx.cycle_basis`` as a fallback when
    polygonize produces too few results.

Door gap threshold
------------------
Walls typically terminate at each side of a doorway, leaving a gap.
A gap smaller than the maximum door width (1.1 m) is bridged.
This is crucial for forming closed room polygons in doorway regions.
"""

from __future__ import annotations

import logging
import time

import networkx as nx
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize, unary_union

from app.pipeline.door_gap_closer import DoorGapCloser

logger = logging.getLogger(__name__)

# Segments shorter than this are dust / zero-length artefacts
_MIN_SEG_LEN: float = 1e-6

# Layer substrings to KEEP (wall candidates)
_WALL_HINTS: tuple[str, ...] = (
    "A-WALL", "WALL", "WALLS", "STRUCT", "CORE", "ARCH",
    "OUTLINE", "PARTITION", "BEARING",
)

# Layer substrings to ALWAYS REJECT when wall_layers is given
_IGNORE_HINTS: tuple[str, ...] = (
    "DIM", "DIMENSION", "GRID", "HATCH", "FILL", "ANNO",
    "TEXT", "FURN", "FURNITURE", "BORDER", "TITLE",
    "VIEWPORT", "DEFPOINT", "CONSTRUCTION",
)

# Maximum gap (in drawing units) to bridge.  Caller must supply the
# conversion factor (to_metres) so we can work in absolute distance.
_DEFAULT_DOOR_GAP_M: float = 1.1   # metres
_MIN_DOOR_GAP_M: float = 0.7
_MIN_BRIDGE_LEN: float = 0.01      # drawing units; below this skip


class WallGraph:
    """
    Builds a planar wall graph and extracts room-candidate polygons.

    Parameters
    ----------
    segments : list of {start, end, layer} dicts from DXFParser
    wall_layers : list of layer names identified as walls (may be empty)
    to_metres : conversion factor from drawing units to metres
    door_gap_m : maximum gap to bridge when closing doorway openings
    """

    def __init__(
        self,
        segments: list[dict],
        wall_layers: list[str] | None = None,
        *,
        to_metres: float = 1.0,
        door_gap_m: float = _DEFAULT_DOOR_GAP_M,
    ) -> None:
        self.segments = segments
        self.wall_layers: list[str] = [l.upper() for l in (wall_layers or [])]
        self.to_metres = max(to_metres, 1e-12)
        self.door_gap_units = door_gap_m / self.to_metres  # gap in drawing units
        self._gap_closer = DoorGapCloser(
            to_metres=self.to_metres,
            min_gap_m=_MIN_DOOR_GAP_M,
            max_gap_m=door_gap_m,
            min_bridge_units=_MIN_BRIDGE_LEN,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_polygons(
        self,
        *,
        min_area_sqft: float = 8.0,
        area_to_sqft: float = 1.0,
        remove_outer: bool = True,
    ) -> list[dict]:
        """
        Full pipeline: filter → clean → gap-close → node → polygonize.

        Returns
        -------
        list of {polygon, area_raw, area_sqft, centroid, coordinates}
        """
        t0 = time.perf_counter()

        filtered = self._filter_segments()
        if not filtered:
            logger.warning("WallGraph: no segments survive layer filter.")
            return []

        lines = self._to_linestrings(filtered)
        lines = self._close_gaps(lines)

        noded = unary_union(MultiLineString(lines))
        polys = list(polygonize(noded))

        # Cycle-basis fallback when polygonize yields nothing useful
        if len(polys) < 2:
            extra = self._cycle_polygons(filtered)
            by_wkt = {p.wkt: p for p in polys}
            for p in extra:
                if p.wkt not in by_wkt:
                    polys.append(p)
                    by_wkt[p.wkt] = p

        min_area_raw = min_area_sqft / max(area_to_sqft, 1e-9)
        valid = [p for p in polys if p.is_valid and not p.is_empty and p.area >= min_area_raw]

        if remove_outer and len(valid) > 1:
            valid.sort(key=lambda p: p.area)
            largest = valid[-1]
            second = valid[-2]
            if largest.area > second.area * 1.5:
                valid = valid[:-1]

        results: list[dict] = []
        for poly in valid:
            c = poly.centroid
            area_raw = float(poly.area)
            results.append({
                "polygon": poly,
                "area_raw": round(area_raw, 4),
                "area_sqft": round(area_raw * area_to_sqft, 2),
                "centroid": [round(c.x, 2), round(c.y, 2)],
                "coordinates": [(round(x, 4), round(y, 4)) for x, y in poly.exterior.coords],
            })

        elapsed = time.perf_counter() - t0
        logger.info(
            "WallGraph: %d raw segs → %d polygons in %.2fs",
            len(filtered), len(results), elapsed,
        )
        return results

    def build_networkx_graph(self) -> nx.Graph:
        """Return a NetworkX graph of the filtered, cleaned wall segments."""
        filtered = self._filter_segments()
        g: nx.Graph = nx.Graph()
        for seg in filtered:
            a = tuple(seg["start"][:2])
            b = tuple(seg["end"][:2])
            if a != b:
                g.add_edge(a, b)
        return g

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _filter_segments(self) -> list[dict]:
        """Keep only wall-layer segments; reject annotation noise."""
        # No wall_layers configured → accept everything except known noise
        if not self.wall_layers:
            all_layers = {str(s.get("layer", "")).upper() for s in self.segments}
            if len(all_layers) <= 1:
                return self.segments  # single-layer file – use everything
            # Multi-layer file without explicit wall list: infer
            return [
                s for s in self.segments
                if not any(ig in str(s.get("layer", "")).upper() for ig in _IGNORE_HINTS)
            ]

        out: list[dict] = []
        for seg in self.segments:
            layer = str(seg.get("layer", "")).upper()
            if any(ig in layer for ig in _IGNORE_HINTS):
                continue
            # Accept if layer contains any known wall hint OR if it is in
            # the explicit wall_layers list (exact prefix match)
            if any(hint in layer for hint in _WALL_HINTS):
                out.append(seg)
                continue
            if any(layer.startswith(wl) or wl in layer for wl in self.wall_layers):
                out.append(seg)
        return out

    @staticmethod
    def _to_linestrings(segments: list[dict]) -> list[LineString]:
        lines: list[LineString] = []
        for seg in segments:
            try:
                p1 = tuple(seg["start"][:2])
                p2 = tuple(seg["end"][:2])
                ls = LineString([p1, p2])
                if ls.length > _MIN_SEG_LEN:
                    lines.append(ls)
            except Exception:
                pass
        return lines

    def _close_gaps(self, lines: list[LineString]) -> list[LineString]:
        """Bridge likely doorway gaps (0.7m–1.1m) using DoorGapCloser."""
        closed_lines, bridge_count = self._gap_closer.close(lines)
        if bridge_count:
            logger.debug("WallGraph: added %d gap-bridging segments", bridge_count)
        return closed_lines

    def _cycle_polygons(self, segments: list[dict]) -> list[Polygon]:
        """Fallback: extract polygons from NetworkX cycle basis."""
        g: nx.Graph = nx.Graph()
        for seg in segments:
            a = (round(float(seg["start"][0]), 4), round(float(seg["start"][1]), 4))
            b = (round(float(seg["end"][0]), 4), round(float(seg["end"][1]), 4))
            if a != b:
                g.add_edge(a, b)
        if g.number_of_edges() == 0:
            return []
        cycles = nx.cycle_basis(g)
        polys: list[Polygon] = []
        for cycle in cycles:
            if len(cycle) >= 3:
                try:
                    p = Polygon(cycle)
                    if p.is_valid and p.area > 0:
                        polys.append(p)
                except Exception:
                    pass
        return polys
