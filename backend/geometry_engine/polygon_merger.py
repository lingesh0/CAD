from shapely.geometry import Polygon
from shapely.ops import unary_union

def merge_polygons(polygons, shared_edge_ratio=0.4, centroid_distance_threshold=5):
    """
    Merge adjacent polygons based on shared edge ratio and centroid distance.

    Args:
        polygons (list): List of Shapely Polygons.
        shared_edge_ratio (float): Minimum shared edge ratio to merge polygons.
        centroid_distance_threshold (float): Maximum centroid distance to merge polygons.

    Returns:
        list: Merged polygons as Shapely Polygon objects.
    """
    merged = []

    while polygons:
        base = polygons.pop(0)
        to_merge = []

        for other in polygons:
            # Check shared edge ratio
            shared_length = base.intersection(other).length
            min_length = min(base.length, other.length)
            if shared_length / min_length >= shared_edge_ratio:
                # Check centroid distance
                if base.centroid.distance(other.centroid) <= centroid_distance_threshold:
                    to_merge.append(other)

        # Merge polygons
        for poly in to_merge:
            polygons.remove(poly)
        base = unary_union([base] + to_merge)
        merged.append(base)

    return merged

if __name__ == "__main__":
    # Example usage
    polygons = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),
        Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    ]

    merged_polygons = merge_polygons(polygons, shared_edge_ratio=0.4, centroid_distance_threshold=5)
    for poly in merged_polygons:
        print("Merged Polygon:", list(poly.exterior.coords))