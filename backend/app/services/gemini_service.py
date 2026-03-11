"""
gemini_service.py – Gemini Vision wrapper
==========================================
Only Google Gemini Vision is used.  OpenAI and Groq have been removed.
For the full accuracy-optimised pipeline use app.pipeline.vision_validator
which implements bulk + per-room validation.  This module is kept as a
compatibility shim for older API routes.
"""

import base64
import json
import logging
import re
import os

logger = logging.getLogger(__name__)

# ── Quota / rate-limit error substrings ────────────────────────────────────
_QUOTA_SIGNALS = (
    "429",
    "quota",
    "rate_limit",
    "rate limit",
    "resource_exhausted",
    "insufficient_quota",
    "tokens per",
    "requests per",
)


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(sig in msg for sig in _QUOTA_SIGNALS)


# ── Master prompt ──────────────────────────────────────────────────────────

_PROMPT_HEADER = """You are an expert architectural floor plan interpreter.

I have already detected {n} rooms from this floor plan using geometry analysis.
Your task is to visually classify each room by looking at the rendered image.

Valid room types (use ONLY these):
Bedroom, Kitchen, Hall, Living Room, Toilet, Bathroom, Utility, Corridor,
Balcony, Staircase, Dining Room, Study Room, Pooja Room, Store Room, Open Area

Ignore dimension lines, north arrows, title blocks, and external boundaries.

Detected rooms (centroid in drawing units, area in sqft, geometry guess):
{room_list}

Return a JSON array with EXACTLY {n} entries in the SAME order as the rooms listed:
[
  {{"room_index": 1, "room_type": "Bedroom", "confidence": 0.93}},
  {{"room_index": 2, "room_type": "Kitchen", "confidence": 0.88}}
]

Rules:
- Classify EVERY room (1 through {n}) - do not skip any
- confidence: 0.0-1.0 (your visual certainty)
- If two rooms look the same type, that is valid (e.g. two Bedrooms)
- Return JSON only, no markdown, no other text
"""


def _build_prompt(rooms_context: list[dict] | None) -> str:
    if not rooms_context:
        return (
            "You are an expert architectural floor plan interpreter.\n"
            "Identify all rooms. Return JSON array:\n"
            '[{"room_index":1,"room_type":"Bedroom","confidence":0.9},...]\n'
            "Valid types: Bedroom, Kitchen, Hall, Living Room, Toilet, Bathroom, "
            "Utility, Corridor, Balcony, Staircase, Dining Room, Study Room, Open Area\n"
            "Return JSON only."
        )

    lines = []
    for i, r in enumerate(rooms_context, 1):
        cx, cy = r["centroid"]
        geo = r.get("geometry_label", "?")
        doors = r.get("door_count", 0)
        lines.append(
            f"  Room {i}: centroid=({cx:.1f},{cy:.1f})  "
            f"area={r['area_sqft']:.0f}sqft  "
            f"doors={doors}  geometry_guess={geo}"
        )

    return _PROMPT_HEADER.format(
        n=len(rooms_context),
        room_list="\n".join(lines),
    )


# ── Main entry point ───────────────────────────────────────────────────────

def interpret_floor_plan(image_path: str, rooms_context: list[dict] | None = None) -> list[dict]:
    """Send floor plan snapshot to Google Gemini Vision.

    Gemini is the ONLY Vision AI provider.  OpenAI and Groq have been
    removed from this codebase.

    Raises RuntimeError if GEMINI_API_KEY is not set or all Gemini
    model variants are exhausted.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Gemini Vision is the only supported AI provider."
        )

    prompt = _build_prompt(rooms_context)
    result = _call_gemini(image_path, prompt, gemini_key)
    logger.info("[Vision AI] Gemini succeeded.")
    return result


# ── Provider implementation ────────────────────────────────────────────────

def _call_gemini(image_path: str, prompt: str, api_key: str) -> list[dict]:
    """Google Gemini Flash vision."""
    from google import genai
    from PIL import Image

    client = genai.Client(api_key=api_key)
    img = Image.open(image_path)
    last_exc: Exception | None = None

    for model_name in ["gemini-2.0-flash", "gemini-2.0-flash-lite"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, img],
            )
            logger.info("Gemini (%s) response: %d chars", model_name, len(response.text))
            return _parse_json_response(response.text)
        except Exception as e:
            last_exc = e
            err_str = str(e)
            if any(s in err_str for s in ("429", "RESOURCE_EXHAUSTED", "NOT_FOUND")):
                logger.warning("Gemini %s unavailable: %s", model_name, err_str[:100])
                continue
            raise  # non-quota error - don't try next model

    raise last_exc or RuntimeError("All Gemini models exhausted.")


# ── Response parser ────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> list[dict]:
    """Extract a JSON array from LLM response text (strips markdown fences)."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[", cleaned)
        if match:
            try:
                data = json.loads(cleaned[match.start():])
            except json.JSONDecodeError:
                logger.error("Could not parse Vision AI JSON: %s", text[:300])
                return []
        else:
            logger.error("No JSON array in Vision AI response: %s", text[:300])
            return []
    if isinstance(data, list):
        return data
    return []