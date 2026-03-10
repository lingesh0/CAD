import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
import os
import uuid
import colorsys
from typing import List, Optional
from pathlib import Path
from app.config import settings


# Generate N visually distinct pastel colours
def _room_colors(n: int) -> list[tuple[float, float, float]]:
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hls_to_rgb(h, 0.75, 0.55)
        colors.append((r, g, b))
    return colors


class SnapshotGenerator:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else settings.snapshot_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_snapshot(
        self,
        walls: List[dict],
        rooms: List[dict],
        floor_id: Optional[int | str] = None,
        *,
        doors: List[dict] | None = None,
        texts: List[dict] | None = None,
    ) -> str:
        """Render walls + room polygons + doors + labels → PNG file."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor("black")  # Set background to black

        # Ensure walls are formatted as lists of coordinate pairs
        formatted_walls = [[wall["start_point"], wall["end_point"]] for wall in walls]

        # Render walls as white lines
        for wall in walls:
            x_coords = [wall["start_point"][0], wall["end_point"][0]]
            y_coords = [wall["start_point"][1], wall["end_point"][1]]
            ax.plot(x_coords, y_coords, color="white", linewidth=2, zorder=1)

        # Render each room polygon separately with centroid labels
        for idx, room in enumerate(rooms):
            polygon_coords = room["coordinates"]
            x, y = zip(*polygon_coords)
            ax.fill(x, y, alpha=0.4, facecolor=plt.cm.tab20(idx % 20), edgecolor="white", linewidth=1.5, zorder=2)
            # Label at centroid so Vision AI can see room names
            if "centroid" in room:
                cx, cy = room["centroid"]
                label = room.get("name", f"Room {idx+1}")
                ax.text(cx, cy, label, color="yellow", fontsize=7,
                        ha="center", va="center", fontweight="bold", zorder=5)

        # Render doors as red diamonds
        if doors:
            for door in doors:
                ax.plot(door["center"][0], door["center"][1], marker="D", color="red", markersize=8, zorder=3)

        # Render text labels
        if texts:
            for text in texts:
                ax.text(text["position"][0], text["position"][1], text["text"], color="white", fontsize=8, zorder=4)

        ax.set_aspect("equal")
        ax.autoscale()
        ax.axis("off")

        # Save the snapshot with floor_id in the filename
        fname = f"floor_{floor_id}.png" if floor_id else "floor_snapshot.png"
        output_path = os.path.join(self.output_dir, fname)
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        return output_path
