"""
Vision Validator
================
Sends ambiguous room snapshots to **Google Gemini Vision** for classification.

This module is the *only* Vision AI integration in the pipeline.
OpenAI and Groq are intentionally excluded.

Trigger condition
-----------------
Only called when a room's classification_confidence < 0.75.

Inputs to Gemini
----------------
- Full floor snapshot (PNG)
- Cropped per-room snapshot (PNG)
- Room area (sqft)
- Detected furniture block types
- Adjacent room type names

Outputs
-------
Per-room: {room_type: str, confidence: float}

Gemini model priority
---------------------
1. gemini-2.5-flash-preview-05-20  (newest, best accuracy)
2. gemini-2.0-flash                 (stable, fast)
3. gemini-2.0-flash-lite            (cheapest, fastest fallback)

The module degrades gracefully:
- If the API key is missing, returns empty results (no crash).
- If all models are rate-limited, returns empty results.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid room labels (must match room_classifier.VALID_LABELS)
# ---------------------------------------------------------------------------
_VALID_LABELS: set[str] = {
    "Bedroom", "Kitchen", "Hall", "Living Room", "Toilet", "Bathroom",
    "Utility", "Corridor", "Balcony", "Staircase", "Dining Room",
    "Study Room", "Pooja Room", "Store Room", "Open Area",
}

# Gemini model preference order
_GEMINI_MODELS = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# Rate-limit / quota error substrings
_QUOTA_SIGNALS = (
    "429", "quota", "rate_limit", "rate limit",
    "resource_exhausted", "RESOURCE_EXHAUSTED", "NOT_FOUND",
)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_PROMPT_TEMPLATE = """You are an expert architectural floor plan analyst.

I am showing you a cropped floor plan image of a single room.

Room details:
  Area     : {area_sqft:.0f} sqft
  Furniture: {furniture}
  Adjacent rooms: {adjacent}

Classify this room into EXACTLY ONE of these types:
Bedroom, Kitchen, Hall, Living Room, Toilet, Bathroom, Utility,
Corridor, Balcony, Staircase, Dining Room, Study Room, Pooja Room,
Store Room, Open Area

Return JSON only (no markdown, no explanation):
{{"room_type": "Bedroom", "confidence": 0.92}}
"""

_BULK_PROMPT_TEMPLATE = """You are an expert architectural floor plan analyst.

I am showing you a rendered floor plan image.

I have already detected {n} rooms using geometry analysis.
Your task is to visually confirm or correct the room types.

Detected rooms:
{room_list}

Valid room types (use ONLY these):
Bedroom, Kitchen, Hall, Living Room, Toilet, Bathroom, Utility,
Corridor, Balcony, Staircase, Dining Room, Study Room, Pooja Room,
Store Room, Open Area

Return a JSON array with EXACTLY {n} entries (same order):
[
  {{"room_index": 1, "room_type": "Bedroom", "confidence": 0.93}},
  {{"room_index": 2, "room_type": "Kitchen",  "confidence": 0.88}}
]

Rules:
- Classify EVERY room listed.
- confidence: 0.0–1.0 (your visual certainty).
- Return JSON only, no markdown.
"""


class VisionValidator:
    """
    Gemini Vision validator for ambiguous room classifications.

    Parameters
    ----------
    api_key : str | None
        GEMINI_API_KEY.  Falls back to the environment variable if None.
    confidence_threshold : float
        Rooms with confidence below this will be sent to Gemini.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_rooms(
        self,
        rooms: list[dict],
        floor_snapshot: str | None,
        room_snapshots: list[str | None],
    ) -> dict[int, dict]:
        """
        Send ambiguous rooms to Gemini and return vision results.

        Parameters
        ----------
        rooms : list of classified room dicts
        floor_snapshot : path to full floor PNG (may be None)
        room_snapshots : per-room PNG paths (index-aligned with rooms)

        Returns
        -------
        dict {room_index: {room_type, confidence}}
        """
        if not self.api_key:
            logger.warning("VisionValidator: no GEMINI_API_KEY – skipping vision.")
            return {}

        # Select ambiguous rooms
        ambiguous: list[tuple[int, dict]] = [
            (i, r) for i, r in enumerate(rooms)
            if not r.get("original_label")
            and float(r.get("confidence", 0.0) or 0.0) < self.threshold
        ]

        if not ambiguous:
            logger.info("VisionValidator: all rooms are high-confidence – skipping vision.")
            return {}

        logger.info(
            "VisionValidator: %d/%d rooms need vision validation",
            len(ambiguous), len(rooms),
        )

        results: dict[int, dict] = {}

        # ── Strategy: bulk call using floor snapshot if available ──────
        if floor_snapshot and Path(floor_snapshot).exists():
            bulk = self._bulk_validate(floor_snapshot, ambiguous)
            results.update(bulk)

        # ── Per-room fallback for rooms that bulk didn't cover ─────────
        for idx, room in ambiguous:
            if idx in results:
                continue
            snap = room_snapshots[idx] if idx < len(room_snapshots) else None
            if snap and Path(snap).exists():
                res = self._validate_single(snap, room)
                if res:
                    results[idx] = res

        return results

    # ------------------------------------------------------------------
    # Gemini calls
    # ------------------------------------------------------------------

    def _bulk_validate(
        self, floor_snapshot: str, ambiguous: list[tuple[int, dict]]
    ) -> dict[int, dict]:
        """Send floor snapshot + room summary to Gemini; parse bulk response."""
        room_lines: list[str] = []
        for idx, room in ambiguous:
            furniture = ", ".join(room.get("furniture", []) or []) or "none"
            adjacent  = ", ".join(room.get("adjacent_rooms", []) or []) or "unknown"
            cur_guess = room.get("classification", "?")
            room_lines.append(
                f"  Room {idx + 1}: area={room.get('area_sqft', 0):.0f}sqft  "
                f"furniture={furniture}  adjacent=[{adjacent}]  "
                f"current_guess={cur_guess}"
            )
        prompt = _BULK_PROMPT_TEMPLATE.format(
            n=len(ambiguous),
            room_list="\n".join(room_lines),
        )

        raw = self._call_gemini(floor_snapshot, prompt)
        if raw is None:
            return {}

        parsed = _parse_json_array(raw)
        results: dict[int, dict] = {}
        for entry in parsed:
            ri = int(entry.get("room_index", 0)) - 1  # convert to 0-based
            rt = entry.get("room_type", "")
            conf = float(entry.get("confidence", 0.5) or 0.5)
            if rt in _VALID_LABELS:
                # Map back: ambiguous[ri] gives (original_room_idx, room)
                if 0 <= ri < len(ambiguous):
                    orig_idx = ambiguous[ri][0]
                    results[orig_idx] = {"room_type": rt, "confidence": conf}
        return results

    def _validate_single(self, snapshot_path: str, room: dict) -> dict | None:
        """Send one room snapshot to Gemini."""
        furniture = ", ".join(room.get("furniture", []) or []) or "none"
        adjacent  = ", ".join(room.get("adjacent_rooms", []) or []) or "unknown"
        prompt = _PROMPT_TEMPLATE.format(
            area_sqft=float(room.get("area_sqft", 0.0) or 0.0),
            furniture=furniture,
            adjacent=adjacent,
        )
        raw = self._call_gemini(snapshot_path, prompt)
        if raw is None:
            return None
        return _parse_json_object(raw)

    def _call_gemini(self, image_path: str, prompt: str) -> str | None:
        """Call Gemini Vision API; try models in order; return raw text or None."""
        try:
            from google import genai
            from PIL import Image
        except ImportError:
            logger.error("google-genai or Pillow not installed – cannot call Gemini.")
            return None

        client = genai.Client(api_key=self.api_key)
        try:
            img = Image.open(image_path)
        except Exception as exc:
            logger.warning("Cannot open image %s: %s", image_path, exc)
            return None

        last_exc: Exception | None = None
        for model in _GEMINI_MODELS:
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[prompt, img],
                )
                text = resp.text or ""
                logger.info("Gemini [%s] responded (%d chars)", model, len(text))
                return text
            except Exception as exc:
                err = str(exc)
                last_exc = exc
                if any(sig in err for sig in _QUOTA_SIGNALS):
                    logger.warning("Gemini [%s] quota/rate-limit: %s", model, err[:120])
                    continue
                # Non-quota error – stop trying
                logger.error("Gemini [%s] error: %s", model, err[:200])
                return None

        if last_exc:
            logger.error("Gemini: all models exhausted. Last error: %s", last_exc)
        return None


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Strip markdown code fences and return raw JSON string."""
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("`").strip()
    return text


def _parse_json_array(text: str) -> list[dict]:
    raw = _extract_json(text)
    # Find first [...] block
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    logger.warning("Gemini: could not parse JSON array from: %s", text[:300])
    return []


def _parse_json_object(text: str) -> dict | None:
    raw = _extract_json(text)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            rt = data.get("room_type", "")
            if rt in _VALID_LABELS:
                return {"room_type": rt, "confidence": float(data.get("confidence", 0.5) or 0.5)}
    except json.JSONDecodeError:
        pass
    logger.warning("Gemini: could not parse JSON object from: %s", text[:300])
    return None
