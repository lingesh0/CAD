"""
Door Gap Closer
===============
Closes likely doorway openings by bridging wall endpoints whose distance falls
within a configurable physical door-width range.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString

logger = logging.getLogger(__name__)


@dataclass
class DoorGapCloser:
    """Bridge wall endpoint gaps likely representing door openings."""

    to_metres: float = 1.0
    min_gap_m: float = 0.7
    max_gap_m: float = 1.1
    min_bridge_units: float = 0.01

    def close(self, lines: list[LineString]) -> tuple[list[LineString], int]:
        if not lines:
            return lines, 0

        scale = max(self.to_metres, 1e-12)
        min_gap_units = self.min_gap_m / scale
        max_gap_units = self.max_gap_m / scale

        endpoints: list[tuple[float, float]] = []
        for ls in lines:
            coords = list(ls.coords)
            endpoints.append((float(coords[0][0]), float(coords[0][1])))
            endpoints.append((float(coords[-1][0]), float(coords[-1][1])))

        seen: set[tuple[float, float]] = set()
        unique: list[tuple[float, float]] = []
        for pt in endpoints:
            key = (round(pt[0], 6), round(pt[1], 6))
            if key not in seen:
                seen.add(key)
                unique.append(pt)

        if len(unique) < 2:
            return lines, 0

        pts_arr = np.array(unique, dtype=float)
        bridges: list[LineString] = []

        for i in range(len(pts_arr)):
            diffs = pts_arr[i + 1 :] - pts_arr[i]
            if diffs.size == 0:
                continue
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            close_idx = np.where(
                (dists >= min_gap_units)
                & (dists <= max_gap_units)
                & (dists > self.min_bridge_units)
            )[0]
            for j_offset in close_idx:
                j = i + 1 + int(j_offset)
                a = (float(pts_arr[i, 0]), float(pts_arr[i, 1]))
                b = (float(pts_arr[j, 0]), float(pts_arr[j, 1]))
                bridges.append(LineString([a, b]))

        if bridges:
            logger.debug(
                "DoorGapCloser: bridged %d endpoint gaps in [%.2fm, %.2fm]",
                len(bridges),
                self.min_gap_m,
                self.max_gap_m,
            )
        return lines + bridges, len(bridges)
