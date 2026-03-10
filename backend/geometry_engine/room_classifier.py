def classify_room(polygon, area_thresholds):
    """
    Classify a room based on its geometry.

    Args:
        polygon (shapely.geometry.Polygon): Room polygon.
        area_thresholds (dict): Area thresholds for classification.
            Example: {
                "hall": 150,
                "bedroom": (60, 120),
                "toilet": 40,
                "corridor_aspect_ratio": 5
            }

    Returns:
        str: Room classification label.
    """
    area = polygon.area
    aspect_ratio = max(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1]) / \
                   min(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1])

    # Classification rules
    if area > area_thresholds.get("hall", 150):
        return "Hall"
    elif area_thresholds.get("bedroom", (60, 120))[0] <= area <= area_thresholds.get("bedroom", (60, 120))[1]:
        return "Bedroom"
    elif area < area_thresholds.get("toilet", 40):
        return "Toilet"
    elif aspect_ratio > area_thresholds.get("corridor_aspect_ratio", 5):
        return "Corridor"
    else:
        return "Unknown"

if __name__ == "__main__":
    from shapely.geometry import Polygon

    # Example usage
    room_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    thresholds = {
        "hall": 150,
        "bedroom": (60, 120),
        "toilet": 40,
        "corridor_aspect_ratio": 5
    }

    room_type = classify_room(room_polygon, thresholds)
    print("Room Type:", room_type)