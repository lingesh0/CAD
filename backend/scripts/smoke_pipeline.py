import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.cad_parser.extractor import DXFExtractor
from app.geometry_engine.room_detector import RoomDetector
from app.snapshot_generator.renderer import SnapshotGenerator


def run_file(dxf_path: Path, tag: str, output_dir: Path) -> dict:
    geometry = DXFExtractor(str(dxf_path)).extract_geometry()
    rooms = RoomDetector(geometry).detect_rooms()

    walls = [
        {"start_point": s["start"], "end_point": s["end"]}
        for s in geometry.get("segments", [])
    ]

    snap_gen = SnapshotGenerator(output_dir=str(output_dir))
    full_snapshot = snap_gen.generate_snapshot(
        walls,
        rooms,
        floor_id=tag,
        doors=geometry.get("doors"),
        texts=geometry.get("texts"),
    )
    room_snapshots = snap_gen.generate_room_snapshots(
        walls,
        rooms,
        floor_id=tag,
    )

    return {
        "file": dxf_path.name,
        "segments": len(geometry.get("segments", [])),
        "texts": len(geometry.get("texts", [])),
        "doors": len(geometry.get("doors", [])),
        "blocks": len(geometry.get("blocks", [])),
        "rooms": len(rooms),
        "snapshot": full_snapshot,
        "room_snapshots": len(room_snapshots),
    }


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    repo_root = backend_root.parent
    output_dir = backend_root / "outputs" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        (repo_root / "files" / "FF.dxf", "ff_smoke"),
        (repo_root / "files" / "FTC- SHEET.dxf", "ftc_smoke"),
    ]

    results = []
    for path, tag in files:
        if not path.exists():
            results.append({"file": str(path), "status": "missing"})
            continue
        results.append(run_file(path, tag, output_dir))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
