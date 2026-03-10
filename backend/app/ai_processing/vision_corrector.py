import requests
import os

def detect_labels(image_path):
    """
    Send the floor plan snapshot to the Vision AI model and retrieve room labels.

    Args:
        image_path (str): Path to the snapshot image.

    Returns:
        list: Detected labels with their positions and confidence scores.
    """
    # Replace with the actual Vision AI API endpoint and key
    API_URL = "https://api.vision-ai.example.com/detect"
    API_KEY = os.getenv("VISION_AI_API_KEY")

    if not API_KEY:
        raise ValueError("VISION_AI_API_KEY environment variable is not set.")

    with open(image_path, "rb") as image_file:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"image": image_file}
        )

    if response.status_code != 200:
        raise RuntimeError(f"Vision AI API request failed with status {response.status_code}: {response.text}")

    return response.json().get("labels", [])

if __name__ == "__main__":
    # Example usage
    snapshot_path = "../outputs/floor_snapshot.png"  # Replace with the actual snapshot path
    labels = detect_labels(snapshot_path)
    print(labels)