"""
Snapshot Renderer
=================
Generates high-quality floor plan PNG snapshots using matplotlib.

Outputs
-------
full_floor_snapshot.png  – Full floor with all rooms coloured and labelled
room_1.png               – Cropped view of room 1 (used by Gemini Vision)
room_2.png               – Cropped view of room 2
...

Design choices
--------------
- Black background for maximum contrast (white walls)
- Pastel room fills with alpha=0.45 (walls still visible through rooms)
- Room labels rendered at centroid in bold yellow
- Doors rendered as red markers (arc-detected) or green markers (block-detected)
- Per-room crops include 20% padding and all walls for context

The renderer uses the Agg (non-interactive) matplotlib backend so it works
on headless Linux servers without a display.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import colorsys
import matplotlib
matplotlib.use("Agg")   # headless – must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

# Colour palette helpers
def _pastel_palette(n: int) -> list[tuple[float, float, float]]:
    """Return N visually distinct pastel RGB tuples."""
    colours: list[tuple[float, float, float]] = []
    for i in range(max(n, 1)):
        h = i / max(n, 1)
        r, g, b = colorsys.hls_to_rgb(h, 0.72, 0.55)
        colours.append((r, g, b))
    return colours


class SnapshotRenderer:
    """
    Render floor plan snapshots.

    Parameters
    ----------
    output_dir : str | Path
        Directory where PNG files are written.  Created if necessary.
    dpi : int
        Resolution for saved images.  300 for production, 150 for quick runs.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        dpi: int = 150,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_floor(
        self,
        segments: list[dict],
        rooms: list[dict],
        *,
        floor_id: int | str = "floor",
        doors: list[dict] | None = None,
        texts: list[dict] | None = None,
        filename: str | None = None,
    ) -> str:
        """
        Render the full floor plan.

        Parameters
        ----------
        segments : wall segments [{start, end, layer}]
        rooms    : room dicts [{name, coordinates, centroid, ...}]
        floor_id : appended to filename when ``filename`` is None
        doors    : door dicts [{center, source, ...}]
        texts    : text label dicts [{text, position}]
        filename : override output filename (absolute or relative to output_dir)

        Returns
        -------
        str  absolute path of the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_facecolor("#0d0d0d")
        fig.patch.set_facecolor("#0d0d0d")

        # Draw walls
        for seg in segments:
            sx, sy = seg["start"][0], seg["start"][1]
            ex, ey = seg["end"][0], seg["end"][1]
            ax.plot([sx, ex], [sy, ey], color="white", linewidth=1.0, zorder=1)

        # Draw rooms
        colours = _pastel_palette(len(rooms))
        for idx, room in enumerate(rooms):
            coords = room.get("coordinates", [])
            if len(coords) < 3:
                continue
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            colour = colours[idx % len(colours)]
            ax.fill(xs, ys, alpha=0.45, facecolor=colour,
                    edgecolor="white", linewidth=1.2, zorder=2)
            # Label at centroid
            centroid = room.get("centroid")
            if centroid:
                cx, cy = centroid
                label = room.get("name", f"Room {idx+1}")
                area  = room.get("area_sqft", 0.0)
                conf  = room.get("confidence", 0.0)
                ax.text(
                    cx, cy,
                    f"{label}\n{area:.0f} sqft",
                    color="yellow",
                    fontsize=6.5,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=5,
                    clip_on=True,
                )

        # Draw doors
        if doors:
            for door in doors:
                cx, cy = door.get("center", [0, 0])
                src = door.get("source", "arc")
                colour = "red" if src == "arc" else "lime"
                ax.plot(cx, cy, marker="D", color=colour,
                        markersize=6, zorder=4, markeredgecolor="black",
                        markeredgewidth=0.5)

        # Draw text labels (light gray, behind rooms)
        if texts:
            for t in texts:
                px, py = t["position"][0], t["position"][1]
                ax.text(px, py, t["text"], color="#aaaaaa",
                        fontsize=5, ha="center", va="center",
                        zorder=3, alpha=0.7)

        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.axis("off")

        if filename:
            out_path = Path(filename)
            if not out_path.is_absolute():
                out_path = self.output_dir / filename
        else:
            out_path = self.output_dir / f"floor_{floor_id}.png"

        plt.savefig(
            str(out_path), dpi=self.dpi,
            bbox_inches="tight", pad_inches=0.05,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        logger.info("SnapshotRenderer: saved floor snapshot → %s", out_path)
        return str(out_path)

    def render_rooms(
        self,
        segments: list[dict],
        rooms: list[dict],
        *,
        floor_id: int | str = "floor",
        filenames: list[str] | None = None,
    ) -> list[str]:
        """
        Generate one cropped PNG per room.

        Parameters
        ----------
        segments  : wall segments (full set – used for context in crop)
        rooms     : room dicts
        floor_id  : prefix for default filenames
        filenames : list of explicit output paths (must match len(rooms))

        Returns
        -------
        list of absolute paths (one per room; empty string for rooms that fail).
        """
        colours = _pastel_palette(len(rooms))
        paths: list[str] = []

        for idx, room in enumerate(rooms):
            coords = room.get("coordinates", [])
            if len(coords) < 3:
                paths.append("")
                continue

            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            pad_x = max(0.5, (max_x - min_x) * 0.25)
            pad_y = max(0.5, (max_y - min_y) * 0.25)
            xlim = (min_x - pad_x, max_x + pad_x)
            ylim = (min_y - pad_y, max_y + pad_y)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_facecolor("#0d0d0d")
            fig.patch.set_facecolor("#0d0d0d")

            # All walls (context)
            for seg in segments:
                sx, sy = seg["start"][0], seg["start"][1]
                ex, ey = seg["end"][0], seg["end"][1]
                ax.plot([sx, ex], [sy, ey], color="white",
                        linewidth=0.8, zorder=1, alpha=0.6)

            # Highlight this room
            colour = colours[idx % len(colours)]
            ax.fill(xs, ys, alpha=0.55, facecolor=colour,
                    edgecolor="white", linewidth=1.5, zorder=2)

            # Label
            centroid = room.get("centroid")
            if centroid:
                cx, cy = centroid
                label = room.get("name", f"Room {idx + 1}")
                area  = room.get("area_sqft", 0.0)
                ax.text(cx, cy, f"{label}\n{area:.0f} sqft",
                        color="yellow", fontsize=9, fontweight="bold",
                        ha="center", va="center", zorder=5)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.axis("off")

            if filenames and idx < len(filenames):
                out_path = Path(filenames[idx])
                if not out_path.is_absolute():
                    out_path = self.output_dir / filenames[idx]
            else:
                out_path = self.output_dir / f"floor_{floor_id}_room_{idx + 1}.png"

            try:
                plt.savefig(
                    str(out_path), dpi=self.dpi,
                    bbox_inches="tight", pad_inches=0.05,
                    facecolor=fig.get_facecolor(),
                )
                paths.append(str(out_path))
            except Exception as exc:
                logger.warning("SnapshotRenderer: failed to save room %d: %s", idx + 1, exc)
                paths.append("")
            finally:
                plt.close(fig)

        logger.info(
            "SnapshotRenderer: saved %d/%d room snapshots for floor %s",
            sum(1 for p in paths if p), len(rooms), floor_id,
        )
        return paths
