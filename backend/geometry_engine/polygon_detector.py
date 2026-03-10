import networkx as nx
from shapely.geometry import Polygon

def detect_cycles(graph):
    """
    Detect cycles (room polygons) in the wall graph.

    Args:
        graph (networkx.Graph): Graph representation of the wall segments.

    Returns:
        list: Detected room polygons as lists of coordinates.
    """
    cycles = []

    # Find all cycles in the graph
    for cycle in nx.cycle_basis(graph):
        # Convert cycle nodes to coordinates
        coords = [(node.x, node.y) for node in cycle]

        # Ensure the cycle forms a valid polygon
        if len(coords) >= 3:
            polygon = Polygon(coords)
            if polygon.is_valid and polygon.area > 0:
                cycles.append(coords)

    return cycles

if __name__ == "__main__":
    # Example usage
    import networkx as nx
    from shapely.geometry import Point

    # Create a simple graph with a cycle
    graph = nx.Graph()
    graph.add_edge(Point(0, 0), Point(1, 0))
    graph.add_edge(Point(1, 0), Point(1, 1))
    graph.add_edge(Point(1, 1), Point(0, 1))
    graph.add_edge(Point(0, 1), Point(0, 0))

    detected_cycles = detect_cycles(graph)
    print("Detected Cycles:", detected_cycles)