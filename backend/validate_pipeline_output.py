import json
from pathlib import Path

from app.pipeline.pipeline import run_pipeline


def main() -> None:
    files = [
        "e:/civil/files/FF.dxf",
        "e:/civil/files/FTC- SHEET.dxf",
    ]
    summary = []

    for path in files:
        result = run_pipeline(path, output_dir="outputs/test_validate", use_vision=False)
        floor = (result.get("floors") or [{}])[0]
        rooms = floor.get("rooms") or []
        summary.append(
            {
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
            }
        )

    out = Path("outputs/test_validate/validation_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out.as_posix())


if __name__ == "__main__":
    main()
