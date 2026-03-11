# CAD: Advanced Floorplan Extraction System

A high-accuracy DXF/DWG floorplan processing engine designed for high-precision room detection and structured data extraction.

## Key Features
- **High-Accuracy Room Detection**: Target 95-98% accuracy using a hybrid Wall Graph algorithm.
- **Wall Graph Algorithm**: Converts DXF wall segments into a geometric graph to detect closed cycles as room boundaries.
- **Door & Block Detection**: Automatically identifies doors (arcs/inserts) and furniture blocks (beds, sofas, etc.) to influence room classification.
- **Gemini Vision Integration**: Uses Google Gemini Vision for final validation of ambiguous room types, eliminating reliance on multiple AI providers.
- **Modular Pipeline**: Extensible architecture including CAD parsing, geometry engines, and snapshot rendering.

## Architecture
1. **CAD Parser**: Layer-filtered extraction of LINE, LWPOLYLINE, ARC, and INSERT entities.
2. **Wall Graph Service**: Endpoint-based graph construction with intelligent gap closing (0.7m–1.1m).
3. **Room Classifier**: Heuristic scoring based on area, perimeter, aspect ratio, and detected blocks.
4. **Adjacency Engine**: Builds connectivity maps between rooms via detected doors.
5. **Vision Validator**: Triggers Gemini Vision for low-confidence classifications.

## Tech Stack
- **Python 3.11+**
- **Geometry**: ezdxf, shapely, networkx, numpy
- **Visualization**: matplotlib, PIL
- **AI**: Google Gemini Vision API

## Output Format
Generates a structured JSON building model including room dimensions, adjacent connectivity, and confidence scores.
