import os
import sys
import json
from pathlib import Path

# Load .env so OPENAI_API_KEY / GEMINI_API_KEY are available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables

# Add the backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from app.cad_parser.extractor import DXFExtractor
from app.geometry_engine.room_detector import RoomDetector
from app.geometry_engine.room_classifier import classify_room_multi_signal
from app.snapshot_generator.renderer import SnapshotGenerator
from app.ai_processing.label_normalizer import LabelNormalizer
from app.ai_processing.label_mapper import map_labels_to_rooms
from app.services.gemini_service import interpret_floor_plan


def test_file(dxf_path: str, output_dir: str, tag: str):
    print(f"\n{'='*60}")
    print(f"  Processing: {dxf_path}")
    print(f"{'='*60}")

    # 1. Extract geometry ------------------------------------------------
    print("[1/7] Extracting geometry ...")
    extractor = DXFExtractor(dxf_path)
    geometry = extractor.extract_geometry()
    segments = geometry["segments"]
    texts = geometry["texts"]
    doors = geometry.get("doors", [])
    blocks = geometry.get("blocks", [])
    metadata = geometry.get("metadata", {})
    print(f"      {len(segments)} wall segments, {len(texts)} text labels, {len(doors)} doors, {len(blocks)} blocks")

    # 2. Detect rooms (wall graph -> cycles -> polygons -> merge) --------
    print("[2/7] Detecting rooms ...")
    detector = RoomDetector(geometry)
    rooms = detector.detect_rooms()
    print(f"      {len(rooms)} rooms detected")
    for i, r in enumerate(rooms):
        print(
            f"        Room {i+1}: {r['name']:30s}  "
            f"area_sqft={r.get('area_sqft', 0.0):.1f}  doors={r.get('door_count', 0)}"
        )

    # 3. Normalise text labels (local abbreviation map + optional AI) ----
    print("[3/7] Normalising labels ...")
    normalizer = LabelNormalizer()
    raw_labels = [r["original_label"] for r in rooms if r.get("original_label")]
    if raw_labels:
        norm_map = normalizer.normalize_labels(raw_labels)
        for room in rooms:
            orig = room.get("original_label")
            if orig and orig in norm_map:
                room["name"] = norm_map[orig]
                room["text_label"] = norm_map[orig]  # store for multi-signal
        print(f"      Normalised {len(raw_labels)} labels")
    else:
        print("      (no raw labels to normalise)")

    # 4. Generate snapshot -----------------------------------------------
    print("[4/7] Generating snapshot ...")
    walls_data = [{"start_point": s["start"], "end_point": s["end"]} for s in segments]
    snap_gen = SnapshotGenerator(output_dir=output_dir)
    snap_path = snap_gen.generate_snapshot(
        walls_data, rooms, floor_id=tag,
        doors=doors, texts=texts,
    )
    print(f"      Snapshot saved -> {snap_path}")

    # 5. Vision AI — send snapshot + full room context -------------------
    print("[5/7] Vision AI (low-confidence rooms only) ...")
    vision_labels = []
    rooms_context = [
        {
            "centroid": r["centroid"],
            "area_sqft": r.get("area_sqft", 0),
            "geometry_label": r.get("name", "?"),
            "door_count": r.get("door_count", 0),
        }
        for r in rooms
    ]
    low_conf_rooms = [r for r in rooms if float(r.get("confidence", 0.0) or 0.0) < 0.75]
    if low_conf_rooms:
        try:
            vision_labels = interpret_floor_plan(snap_path, rooms_context=rooms_context)
            print(f"      Vision AI returned {len(vision_labels)} labels")
            for vl in vision_labels:
                idx = vl.get("room_index", "?")
                print(f"        Room {idx}: {vl.get('room_type'):20s}  conf={vl.get('confidence')}")
        except Exception as e:
            print(f"      Vision AI skipped: {e}")
    else:
        print("      All rooms above confidence threshold; AI refinement skipped")

    # 6. Map Vision AI labels -> rooms -----------------------------------
    print("[6/7] Mapping labels ...")
    if vision_labels:
        rooms = map_labels_to_rooms(rooms, vision_labels)
        print("      Vision labels mapped to rooms")
    else:
        print("      (no vision labels to map)")

    # 7. Multi-signal scoring: geometry + vision + text ------------------
    print("[7/7] Multi-signal scoring ...")
    for room in rooms:
        result = classify_room_multi_signal(
            area_sqft=room.get("area_sqft", 0),
            polygon=room.get("polygon"),
            door_count=room.get("door_count", 0),
            geometry_label=room.get("classification") or room.get("name"),
            vision_label=room.get("vision_label"),
            vision_confidence=room.get("vision_confidence", 0.80),
            text_label=room.get("text_label") or room.get("original_label"),
        )
        room["name"] = result["label"]
        room["confidence"] = round(result["confidence"], 2)
        room["method"] = result["method"]

    # Print final output -------------------------------------------------
    print("\nFinal rooms:")
    for r in rooms:
        print(
            f"  {r['name']:25s}  {r.get('area_sqft', 0):6.1f} sqft  "
            f"conf={r.get('confidence', 0.0):.2f}  [{r.get('method', 'geometry')}]"
        )

    result_json = {
        "file": os.path.basename(dxf_path),
        "insunits": metadata.get("insunits"),
        "linear_to_feet": metadata.get("linear_to_feet"),
        "total_segments": len(segments),
        "total_texts": len(texts),
        "total_doors": len(doors),
        "total_blocks": len(blocks),
        "rooms": [
            {
                "name": r["name"],
                "area_sqft": round(r.get("area_sqft", 0.0), 2),
                "centroid": [round(c, 2) for c in r.get("centroid", [0, 0])],
                "original_label": r.get("original_label"),
                "door_count": r.get("door_count", 0),
                "adjacency": r.get("adjacent_rooms", []),
                "furniture": r.get("furniture", []),
                "confidence": r.get("confidence"),
                "method": r.get("method", "geometry"),
            }
            for r in rooms
        ],
        "snapshot": snap_path,
    }
    print("\nJSON output:")
    print(json.dumps(result_json, indent=2))
    return result_json

from pathlib import Path
import os

def main():

    base_dir = Path(__file__).resolve().parent.parent

    input_dir = base_dir / "data" / "input"
    output_dir = base_dir / "data" / "output"

    os.makedirs(output_dir, exist_ok=True)

    files = [
        (input_dir / "FF.dxf", "ff"),
        (input_dir / "FTC-SHEET.dxf", "ftc"),
    ]

    for dxf_path, tag in files:
        if dxf_path.exists():
            test_file(str(dxf_path), str(output_dir), tag)
        else:
            print(f"SKIP: {dxf_path} not found")


if __name__ == "__main__":
    main()
