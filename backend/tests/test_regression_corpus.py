import json
from pathlib import Path

import pytest

from app.cad_parser.extractor import DXFExtractor
from app.geometry_engine.room_detector import RoomDetector


VALID_LABELS = {
    "bedroom",
    "kitchen",
    "hall",
    "living room",
    "toilet",
    "bathroom",
    "utility",
    "corridor",
    "balcony",
    "staircase",
    "dining room",
    "study room",
    "pooja room",
    "store room",
    "open area",
}


def _normalize_label(label: str) -> str:
    return " ".join((label or "").lower().strip().split())


def _precision(predicted: list[str], expected: list[str]) -> float:
    if not predicted:
        return 0.0
    expected_set = {_normalize_label(x) for x in expected}
    predicted_set = {_normalize_label(x) for x in predicted}
    hit = sum(1 for p in predicted_set if p in expected_set)
    return hit / max(1, len(predicted_set))


def _load_corpus() -> tuple[float, list[dict]]:
    corpus_path = Path(__file__).resolve().parent / "fixtures" / "labeled_corpus.json"
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    min_precision = float(data.get("min_precision", 0.5))
    return min_precision, data.get("samples", [])


@pytest.mark.regression
def test_labeled_corpus_precision() -> None:
    min_precision, samples = _load_corpus()
    if not samples:
        pytest.skip("No labeled corpus samples configured")

    precisions = []
    evaluated = 0

    for sample in samples:
        rel = sample.get("file", "")
        expected = sample.get("expected_labels", [])
        if not expected:
            continue

        file_path = (Path(__file__).resolve().parent / rel).resolve()
        if not file_path.exists():
            continue

        geometry = DXFExtractor(str(file_path)).extract_geometry()
        rooms = RoomDetector(geometry).detect_rooms()
        predicted = [_normalize_label(r.get("name", "")) for r in rooms if r.get("name")]

        p = _precision(predicted, expected)
        precisions.append(p)
        evaluated += 1

        unknown = [lbl for lbl in predicted if lbl not in VALID_LABELS]
        assert len(unknown) == 0, f"Unexpected labels produced for {file_path.name}: {unknown}"

    if evaluated == 0:
        pytest.skip("No sample files found locally for regression run")

    avg_precision = sum(precisions) / max(1, len(precisions))
    assert avg_precision >= min_precision, (
        f"Regression precision too low: {avg_precision:.3f} < {min_precision:.3f}"
    )
