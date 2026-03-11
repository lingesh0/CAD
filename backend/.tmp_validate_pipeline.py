import json
from app.pipeline.pipeline import run_pipeline

files = [
    r"e:/civil/files/FF.dxf",
    r"e:/civil/files/FTC- SHEET.dxf",
]
summary = []
for path in files:
    result = run_pipeline(path, output_dir="outputs/test_validate", use_vision=False)
    floor = (result.get("floors") or [{}])[0]
    rooms = floor.get("rooms") or []
    summary.append({
        "file": result.get("file"),
        "status": result.get("status"),
        "elapsed_sec": result.get("elapsed_sec"),
        "room_count": len(rooms),
        "snapshot": floor.get("snapshot"),
        "rooms": [
            {
                "name": r.get("name"),
                "area_sqft": r.get("area_sqft"),
                "doors": r.get("doors"),
                "confidence": r.get("confidence"),
            }
            for r in rooms
        ],
    })

print(json.dumps(summary, indent=2))
