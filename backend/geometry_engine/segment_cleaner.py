from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union, snap
import numpy as np

def clean_segments(segments, tolerance=0.01, min_length=0.1):
    """
    Clean and preprocess wall segments.

    Args:
        segments (list): List of raw segments as [(x1, y1), (x2, y2)].
        tolerance (float): Tolerance for snapping endpoints.
        min_length (float): Minimum length of segments to keep.

    Returns:
        list: Cleaned segments as [(x1, y1), (x2, y2)].
    """
    # Convert segments to Shapely LineStrings
    lines = [LineString(segment) for segment in segments if LineString(segment).length >= min_length]

    # Snap endpoints within the given tolerance
    snapped = snap(unary_union(lines), unary_union(lines), tolerance)

    # Merge collinear segments
    merged = linemerge(snapped)

    # Ensure output is a list of LineStrings
    if isinstance(merged, LineString):
        merged = [merged]
    elif isinstance(merged, list):
        merged = [line for line in merged if line.length >= min_length]

    # Convert back to list of segments
    cleaned_segments = [list(line.coords) for line in merged]

    return cleaned_segments

if __name__ == "__main__":
    # Example usage
    raw_segments = [
        [(0, 0), (1, 1)],
        [(1, 1), (2, 2)],
        [(2, 2), (3, 3)],
        [(0, 0), (0.05, 0.05)],  # Short segment
        [(3, 3), (4, 4)]
    ]

    cleaned = clean_segments(raw_segments, tolerance=0.1, min_length=0.2)
    print("Cleaned Segments:", cleaned)