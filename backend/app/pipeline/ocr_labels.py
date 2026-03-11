"""
OCR Labels
==========
Extracts visible room labels from rendered snapshots and assigns labels to
room polygons by centroid fallback (room crops) and optional full-floor point
containment if coordinate mapping is supplied.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


_LABEL_NORMALIZE = {
    "BED ROOM": "Bedroom",
    "BEDROOM": "Bedroom",
    "KITCHEN": "Kitchen",
    "LIVING": "Hall",
    "LIVING ROOM": "Living Room",
    "HALL": "Hall",
    "DINING": "Dining Room",
    "DINING ROOM": "Dining Room",
    "TOILET": "Toilet",
    "WC": "Toilet",
    "BATH": "Bathroom",
    "BATHROOM": "Bathroom",
    "UTILITY": "Utility",
    "STORE": "Store Room",
    "STORE ROOM": "Store Room",
    "STUDY": "Study Room",
    "POOJA": "Pooja Room",
    "BALCONY": "Balcony",
    "CORRIDOR": "Corridor",
    "PASSAGE": "Corridor",
}


def _ocr_words_impl(image_path: str, min_confidence: float) -> list[str]:
    if not image_path or not Path(image_path).exists():
        return []

    try:
        import easyocr  # type: ignore

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        result = reader.readtext(image_path, detail=1)
        words: list[str] = []
        for item in result:
            try:
                text = str(item[1]).strip()
                conf = float(item[2])
            except Exception:
                continue
            if conf >= min_confidence and text:
                words.append(text)
        if words:
            return words
    except Exception as exc:
        logger.debug("EasyOCR unavailable/failed: %s", exc)

    try:
        import cv2  # type: ignore
        import pytesseract  # type: ignore

        img = cv2.imread(image_path)
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(th)
        chunks = [c.strip() for c in re.split(r"[\n,;|]+", text) if c.strip()]
        return chunks
    except Exception as exc:
        logger.debug("Tesseract fallback unavailable/failed: %s", exc)
        return []


def _ocr_words_worker(image_path: str, min_confidence: float, out_q: mp.Queue) -> None:
    try:
        out_q.put(_ocr_words_impl(image_path, min_confidence))
    except Exception:
        out_q.put([])


@dataclass
class OCRLabelExtractor:
    min_confidence: float = 0.35
    per_image_timeout_sec: float = 8.0
    total_ocr_budget_sec: float = 60.0

    def detect_room_labels(
        self,
        floor_snapshot: str,
        room_snapshots: list[str],
        rooms: list[dict],
    ) -> int:
        """Assign OCR labels in-place to rooms, returns number of assigned labels."""
        assigned = 0
        started_at = time.perf_counter()

        # Fast and robust path: OCR per-room crops.
        for idx, room_image in enumerate(room_snapshots):
            if idx >= len(rooms) or not room_image:
                continue
            if (time.perf_counter() - started_at) > self.total_ocr_budget_sec:
                logger.warning(
                    "OCR budget exceeded (%.1fs). Stopping OCR early.",
                    self.total_ocr_budget_sec,
                )
                break
            text = self._ocr_primary_label(room_image)
            if not text:
                continue
            room = rooms[idx]
            if not room.get("original_label"):
                room["original_label"] = text
                room["label_source"] = "ocr"
                assigned += 1

        # Optional floor-level OCR for rooms still unlabeled.
        # We keep this as a weak fallback by nearest-centroid association.
        leftovers = [i for i, r in enumerate(rooms) if not r.get("original_label")]
        if leftovers and floor_snapshot and (time.perf_counter() - started_at) <= self.total_ocr_budget_sec:
            floor_words = self._ocr_words(floor_snapshot)
            for word in floor_words:
                label = self._normalize(word)
                if not label:
                    continue
                target_idx = self._nearest_unlabeled_room(rooms, leftovers)
                if target_idx is None:
                    break
                rooms[target_idx]["original_label"] = label
                rooms[target_idx]["label_source"] = "ocr_floor"
                leftovers.remove(target_idx)
                assigned += 1

        return assigned

    def _nearest_unlabeled_room(self, rooms: list[dict], candidates: list[int]) -> int | None:
        if not candidates:
            return None
        # Deterministic fallback: smallest area unlabeled room first
        best = min(candidates, key=lambda i: float(rooms[i].get("area_sqft", 0.0) or 0.0))
        return best

    def _ocr_primary_label(self, image_path: str) -> str | None:
        words = self._ocr_words(image_path)
        if not words:
            return None
        # Prefer longest normalized phrase, typically full room name.
        candidates = [self._normalize(w) for w in words]
        candidates = [c for c in candidates if c]
        if not candidates:
            return None
        candidates.sort(key=len, reverse=True)
        return candidates[0]

    def _ocr_words(self, image_path: str) -> list[str]:
        if not image_path or not Path(image_path).exists():
            return []

        # Use a subprocess to guarantee timeout enforcement for OCR backends.
        try:
            ctx = mp.get_context("spawn")
            out_q: mp.Queue = ctx.Queue(maxsize=1)
            proc = ctx.Process(
                target=_ocr_words_worker,
                args=(image_path, self.min_confidence, out_q),
                daemon=True,
            )
            proc.start()
            proc.join(self.per_image_timeout_sec)
            if proc.is_alive():
                proc.terminate()
                proc.join(1.0)
                logger.warning("OCR timed out after %.1fs for %s", self.per_image_timeout_sec, image_path)
                return []
            try:
                result = out_q.get_nowait()
                if isinstance(result, list):
                    return [str(x) for x in result if str(x).strip()]
            except Exception:
                return []
        except Exception as exc:
            logger.warning("OCR subprocess failed for %s: %s", image_path, exc)
            return []

        return []

    def _normalize(self, raw: str) -> str | None:
        txt = re.sub(r"[^A-Za-z ]+", " ", raw.upper()).strip()
        txt = re.sub(r"\s+", " ", txt)
        if not txt:
            return None
        if txt in _LABEL_NORMALIZE:
            return _LABEL_NORMALIZE[txt]

        # Keyword fallback
        for key, value in _LABEL_NORMALIZE.items():
            if key in txt:
                return value
        return None
