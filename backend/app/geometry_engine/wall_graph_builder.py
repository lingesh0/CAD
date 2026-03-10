"""
Wall Graph Builder
==================
Converts raw DXF segments into a noded planar graph suitable for
room detection via polygon extraction.

Algorithm
---------
1.  Convert every extracted segment into a Shapely ``LineString``.
2.  Combine into a ``MultiLineString`` and call ``unary_union`` which
    performs *noding* – it splits every segment at every intersection
    point so that no two edges cross without sharing a vertex.
3.  Return the noded geometry (a collection of non-crossing edges)
    which is the input for ``polygonize`` in the polygon detector.
"""

import logging
import time
from typing import Any, List

import networkx as nx
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# Minimum segment length to keep (filters out zero-length / dust)
_MIN_SEG_LEN = 1e-6


class WallGraphBuilder:
    """Builds a noded planar geometry from raw DXF segments."""

    def __init__(self, segments: List[dict]):
        self.segments = segments

    def build(self) -> Any:
        """Return a noded Shapely geometry for polygonize fallback.

        The returned object can be passed directly to
        ``shapely.ops.polygonize``.
        """
        t0 = time.perf_counter()

        lines: list[LineString] = []
        for seg in self.segments:
            try:
                p1 = tuple(seg["start"][:2])
                p2 = tuple(seg["end"][:2])
                ls = LineString([p1, p2])
                if ls.length > _MIN_SEG_LEN:
                    lines.append(ls)
            except Exception:
                continue

        if not lines:
            logger.warning("No valid line segments to build graph from.")
            return MultiLineString()

        ml = MultiLineString(lines)
        # unary_union performs intersection noding on all segments
        noded = unary_union(ml)

        elapsed = time.perf_counter() - t0
        if isinstance(noded, MultiLineString):
            seg_count = len(noded.geoms)
        else:
            seg_count = 1
        logger.info(
            "WallGraphBuilder: %d input segs → %d noded edges (%.2fs)",
            len(lines), seg_count, elapsed,
        )
        return noded

    def build_graph(self) -> nx.Graph:
        """Build an undirected planar graph from cleaned segments."""
        g = nx.Graph()
        for seg in self.segments:
            try:
                a = (float(seg["start"][0]), float(seg["start"][1]))
                b = (float(seg["end"][0]), float(seg["end"][1]))
                if a == b:
                    continue
                g.add_edge(a, b)
            except Exception:
                continue
        logger.info("WallGraphBuilder graph: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())
        return g

    def detect_cycles(self) -> list[list[tuple[float, float]]]:
        """Detect fundamental graph cycles via networkx.cycle_basis."""
        g = self.build_graph()
        if g.number_of_edges() == 0:
            return []
        cycles = nx.cycle_basis(g)
        logger.info("WallGraphBuilder cycle basis: %d cycles", len(cycles))
        return cycles
