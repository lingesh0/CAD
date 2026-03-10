import networkx as nx
from shapely.geometry import Point, LineString

def build_wall_graph(segments):
    """
    Build a wall graph from cleaned segments.

    Args:
        segments (list): List of cleaned segments as [(x1, y1), (x2, y2)].

    Returns:
        networkx.Graph: Graph representation of the wall segments.
    """
    graph = nx.Graph()

    for segment in segments:
        start, end = segment
        start_point = Point(start)
        end_point = Point(end)

        # Add nodes for segment endpoints
        graph.add_node(start_point)
        graph.add_node(end_point)

        # Add edge for the segment
        graph.add_edge(start_point, end_point, weight=LineString(segment).length)

    return graph

if __name__ == "__main__":
    # Example usage
    cleaned_segments = [
        [(0, 0), (1, 1)],
        [(1, 1), (2, 2)],
        [(2, 2), (3, 3)],
        [(3, 3), (4, 4)]
    ]

    wall_graph = build_wall_graph(cleaned_segments)
    print("Nodes:", wall_graph.nodes)
    print("Edges:", wall_graph.edges)