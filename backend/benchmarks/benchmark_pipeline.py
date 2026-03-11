import argparse
import gc
import importlib
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.cad_parser.extractor import DXFExtractor
from app.geometry_engine.room_detector import RoomDetector


def _rss_mb() -> float:
    try:
        psutil = importlib.import_module("psutil")
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _bench_file(path: Path) -> dict:
    gc.collect()
    tracemalloc.start()
    rss_before = _rss_mb()

    t0 = time.perf_counter()
    extractor = DXFExtractor(str(path))
    geometry = extractor.extract_geometry()
    t_extract = time.perf_counter()

    detector = RoomDetector(geometry)
    rooms = detector.detect_rooms()
    t_detect = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _rss_mb()

    extract_s = t_extract - t0
    detect_s = t_detect - t_extract
    total_s = t_detect - t0

    return {
        "file": str(path),
        "segments": len(geometry.get("segments", [])),
        "doors": len(geometry.get("doors", [])),
        "blocks": len(geometry.get("blocks", [])),
        "rooms": len(rooms),
        "extract_sec": round(extract_s, 4),
        "detect_sec": round(detect_s, 4),
        "total_sec": round(total_s, 4),
        "peak_tracemalloc_mb": round(peak / (1024 * 1024), 2),
        "rss_delta_mb": round(max(0.0, rss_after - rss_before), 2),
        "under_3s_target": total_s < 3.0,
    }


def _find_default_dxf_files(root: Path) -> list[Path]:
    files_dir = root / "files"
    if not files_dir.exists():
        return []
    paths = sorted(files_dir.glob("*.dxf"), key=lambda p: p.stat().st_size, reverse=True)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CAD pipeline on DXF files")
    parser.add_argument("--file", action="append", default=[], help="DXF file path (repeatable)")
    parser.add_argument("--top", type=int, default=3, help="Use N largest DXFs from ../files when --file not set")
    parser.add_argument("--output", default="outputs/test/benchmark_report.json", help="Output report path")
    args = parser.parse_args()

    backend_root = BACKEND_ROOT
    file_paths: list[Path]
    if args.file:
        file_paths = [Path(p).resolve() for p in args.file]
    else:
        all_default = _find_default_dxf_files(backend_root.parent)
        file_paths = all_default[: max(1, args.top)]

    existing = [p for p in file_paths if p.exists() and p.suffix.lower() == ".dxf"]
    if not existing:
        print("No DXF files found for benchmarking.")
        return

    results = []
    for path in existing:
        print(f"Benchmarking {path} ...")
        results.append(_bench_file(path))

    avg_total = sum(r["total_sec"] for r in results) / max(1, len(results))
    target_pass_count = sum(1 for r in results if r["under_3s_target"])
    summary = {
        "files_tested": len(results),
        "avg_total_sec": round(avg_total, 4),
        "target_under_3s_ratio": round(target_pass_count / max(1, len(results)), 3),
        "results": results,
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = backend_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Benchmark summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
