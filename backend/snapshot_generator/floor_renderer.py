import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

def render_floorplan(polygons, labels, output_path):
    """
    Render a floorplan with room polygons and labels.

    Args:
        polygons (list): List of Shapely Polygons representing rooms.
        labels (list): List of labels corresponding to the polygons.
        output_path (str): Path to save the rendered image.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    patches = []
    for polygon in polygons:
        mpl_poly = MplPolygon(list(polygon.exterior.coords), closed=True, edgecolor='black')
        patches.append(mpl_poly)

    # Add polygons to the plot
    collection = PatchCollection(patches, facecolor='lightblue', edgecolor='black', alpha=0.5)
    ax.add_collection(collection)

    # Add labels
    for polygon, label in zip(polygons, labels):
        x, y = polygon.centroid.x, polygon.centroid.y
        ax.text(x, y, label, ha='center', va='center', fontsize=8, color='black')

    # Set plot limits
    all_coords = [coord for polygon in polygons for coord in polygon.exterior.coords]
    x_coords, y_coords = zip(*all_coords)
    ax.set_xlim(min(x_coords) - 10, max(x_coords) + 10)
    ax.set_ylim(min(y_coords) - 10, max(y_coords) + 10)
    ax.set_aspect('equal', adjustable='datalim')

    # Save the plot
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    # Example usage
    polygons = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(12, 0), (22, 0), (22, 10), (12, 10)])
    ]
    labels = ["Bedroom", "Kitchen"]
    output_path = "floorplan.png"

    render_floorplan(polygons, labels, output_path)
    print(f"Floorplan saved to {output_path}")