import json
import time
from pathlib import Path

from app.pipeline.pipeline import run_pipeline

FILES = [
    "e:/civil/files/FF.dxf",
    "e:/civil/files/FTC- SHEET.dxf",
]
OUT_DIR = Path("outputs/test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary = {"total_elapsed": 0.0, "runs": []}
t0 = time.perf_counter()

for fpath in FILES:
    start = time.perf_counter()
    floor_id = Path(fpath).stem
    result = run_pipeline(
        fpath,
        output_dir=str(OUT_DIR),
        use_vision=False,
        floor_id=floor_id,
    )
    elapsed = time.perf_counter() - start

    floor = (result.get("floors") or [{}])[0]
    rooms = floor.get("rooms") or []
    snapshot = floor.get("snapshot")

    room_png_count = 0
    for r in rooms:
        sp = r.get("snapshot")
        if sp and Path(sp).exists():
            room_png_count += 1

    summary["runs"].append(
        {
            "file": result.get("file"),
            "elapsed_sec": round(elapsed, 3),
            "room_count": len(rooms),
            "first_rooms": [r.get("name") for r in rooms[:8]],
            "snapshot_exists": bool(snapshot and Path(snapshot).exists()),
            "room_snapshot_count": room_png_count,
            "json_has_required_keys": all(
                k in result for k in ("file", "status", "floors")
            ),
        }
    )

summary["total_elapsed"] = round(time.perf_counter() - t0, 3)

out_json = OUT_DIR / "runtime_validation_summary.json"
out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"Saved summary to {out_json}")
