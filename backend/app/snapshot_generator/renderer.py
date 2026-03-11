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

    def generate_room_snapshots(
        self,
        walls: List[dict],
        rooms: List[dict],
        floor_id: Optional[int | str] = None,
    ) -> list[str]:
        """Generate cropped snapshots focused on each detected room."""
        output_paths: list[str] = []
        for idx, room in enumerate(rooms):
            coords = room.get("coordinates", [])
            if len(coords) < 3:
                continue

            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            pad_x = max(0.5, (max_x - min_x) * 0.2)
            pad_y = max(0.5, (max_y - min_y) * 0.2)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_facecolor("black")

            for wall in walls:
                x_coords = [wall["start_point"][0], wall["end_point"][0]]
                y_coords = [wall["start_point"][1], wall["end_point"][1]]
                ax.plot(x_coords, y_coords, color="white", linewidth=1.0, zorder=1)

            x, y = zip(*coords)
            ax.fill(
                x,
                y,
                alpha=0.5,
                facecolor=plt.cm.tab20(idx % 20),
                edgecolor="white",
                linewidth=1.5,
                zorder=2,
            )

            if "centroid" in room:
                cx, cy = room["centroid"]
                ax.text(
                    cx,
                    cy,
                    room.get("name", f"Room {idx+1}"),
                    color="yellow",
                    fontsize=7,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    zorder=3,
                )

            ax.set_xlim(min_x - pad_x, max_x + pad_x)
            ax.set_ylim(min_y - pad_y, max_y + pad_y)
            ax.set_aspect("equal")
            ax.axis("off")

            base = f"floor_{floor_id}_room_{idx+1}.png" if floor_id else f"room_{idx+1}.png"
            output_path = os.path.join(self.output_dir, base)
            plt.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            output_paths.append(output_path)

        return output_paths
