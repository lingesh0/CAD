# CAD: Advanced Floorplan Extraction System

A high-accuracy DXF/DWG floorplan processing engine designed for high-precision room detection and structured data extraction.

## Key Features
- **High-Accuracy Room Detection**: Target 95-98% accuracy using a hybrid Wall Graph algorithm.
- **Wall Graph Algorithm**: Converts DXF wall segments into a geometric graph to detect closed cycles as room boundaries.
- **Door & Block Detection**: Automatically identifies doors (arcs/inserts) and furniture blocks (beds, sofas, etc.) to influence room classification.
- **Gemini Vision Integration**: Uses Google Gemini Vision for final validation of ambiguous room types, eliminating reliance on multiple AI providers.
- **Modular Pipeline**: Extensible architecture including CAD parsing, geometry engines, and snapshot rendering.

## Architecture

The system is designed as a modular pipeline, eliminating complex multi-AI chains in favor of robust geometry processing and a single Gemini Vision validation step.

### Core Modules (`backend/app/pipeline/`)
1. **`cad_parser.py`**: Reads DXF files using `ezdxf`, filtering entities by specific layer names (e.g., `A-WALL`, `A-DOOR`). Extracts lines, polylines, arcs, texts, and block inserts.
2. **`wall_graph.py`**: Converts raw wall segments into a geometric graph using `networkx`. Connects close endpoints (door gaps) and detects closed cycles to form base room polygons.
3. **`door_detector.py`**: Identifies doors by detecting ARC entities (door swings) and door block inserts. Incorporates doors into room adjacency logic.
4. **`block_detector.py`**: Parses INSERT entities for furniture and fixtures (BED, SOFA, WC, STOVE) to assign semantic meaning to spaces.
5. **`room_classifier.py`**: Employs heuristic scoring based on area, perimeter, furniture presence, and nearby text labels to classify rooms without AI.
6. **`adjacency_graph.py`**: Uses detected doors to build a connectivity graph between rooms (e.g., verifying Kitchen is near Dining).
7. **`vision_validator.py`**: Fallback module that triggers **Google Gemini Vision** *only* if a room's heuristic confidence is below a defined threshold (`< 0.75`).

## Testing the Pipeline

To run the pipeline tests, you need to set up the Python environment and run the provided test scripts.

### 1. Environment Setup
```bash
cd backend
python -m venv venv
# Activate the virtual environment:
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

*(If using Gemini Vision validation, ensure an `.env` file exists with `GEMINI_API_KEY=...` in the `backend/` directory).*

### 2. Running the Full Test Pipeline
The main testing script processes the test DXF files, runs the full analytical pipeline, generates snapshots in `backend/outputs/test/`, and outputs a detailed JSON report.

```bash
cd backend
python test_pipeline.py
```

### 3. Running Smoke Tests (Geometry Only)
To run a fast validation of just the extraction, wall graph, and snapshot generation without AI calls:

```bash
cd backend
python scripts/smoke_pipeline.py
```

## Output Format
Generates a structured JSON building model including room dimensions, adjacent connectivity, and confidence scores. Output snapshots are generated in `backend/outputs/`.
