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
    """Send floor plan snapshot to Vision AI with automatic provider fallback.

    Fallback order:
        1. OpenAI  GPT-4o          (OPENAI_API_KEY)
        2. Google  Gemini Flash     (GEMINI_API_KEY)
        3. Groq    Llama-4 Vision   (GROQ_API_KEY)

    Each provider logs a clear message when its quota/rate-limit is hit
    before moving to the next one.

    Raises RuntimeError only when ALL configured providers fail.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key   = os.getenv("GROQ_API_KEY")

    if not any([openai_key, gemini_key, groq_key]):
        raise ValueError(
            "No Vision AI API key found. "
            "Set at least one of: OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY."
        )

    prompt = _build_prompt(rooms_context)
    errors: list[str] = []

    # ── 1. OpenAI ─────────────────────────────────────────────────────
    if openai_key:
        try:
            result = _call_openai(image_path, prompt, openai_key)
            logger.info("[Vision AI] OpenAI succeeded.")
            return result
        except Exception as e:
            if _is_quota_error(e):
                msg = f"[Vision AI] OpenAI quota/rate-limit exceeded: {e}"
                logger.warning(msg)
                errors.append(msg)
                print(f"      [OpenAI] Limit exceeded - trying Gemini...")
            else:
                msg = f"[Vision AI] OpenAI error: {e}"
                logger.warning(msg)
                errors.append(msg)
                print(f"      [OpenAI] Failed ({type(e).__name__}) - trying Gemini...")

    # ── 2. Gemini ──────────────────────────────────────────────────────
    if gemini_key:
        try:
            result = _call_gemini(image_path, prompt, gemini_key)
            logger.info("[Vision AI] Gemini succeeded.")
            return result
        except Exception as e:
            if _is_quota_error(e):
                msg = f"[Vision AI] Gemini quota/rate-limit exceeded: {e}"
                logger.warning(msg)
                errors.append(msg)
                print(f"      [Gemini] Limit exceeded - trying Groq...")
            else:
                msg = f"[Vision AI] Gemini error: {e}"
                logger.warning(msg)
                errors.append(msg)
                print(f"      [Gemini] Failed ({type(e).__name__}) - trying Groq...")

    # ── 3. Groq ────────────────────────────────────────────────────────
    if groq_key:
        try:
            result = _call_groq(image_path, prompt, groq_key)
            logger.info("[Vision AI] Groq succeeded.")
            return result
        except Exception as e:
            if _is_quota_error(e):
                msg = f"[Vision AI] Groq quota/rate-limit exceeded: {e}"
                logger.warning(msg)
                errors.append(msg)
                print(f"      [Groq] Limit exceeded - all providers exhausted.")
            else:
                msg = f"[Vision AI] Groq error: {e}"
                logger.warning(msg)
                errors.append(msg)
                print(f"      [Groq] Failed ({type(e).__name__}) - all providers exhausted.")

    raise RuntimeError(
        "All Vision AI providers failed.\n" + "\n".join(errors)
    )


# ── Provider implementations ───────────────────────────────────────────────

def _call_openai(image_path: str, prompt: str, api_key: str) -> list[dict]:
    """OpenAI GPT-4o vision."""
    from openai import OpenAI

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    client = OpenAI(api_key=api_key, timeout=30.0)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=1500,
    )
    raw = response.choices[0].message.content
    logger.info("OpenAI response: %d chars", len(raw))
    return _parse_json_response(raw)


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


def _call_groq(image_path: str, prompt: str, api_key: str) -> list[dict]:
    """Groq Llama-4 Scout vision (free, fast)."""
    from groq import Groq

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    client = Groq(api_key=api_key, timeout=30.0)
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                        },
                    },
                ],
            }
        ],
        max_tokens=1500,
    )
    raw = response.choices[0].message.content
    logger.info("Groq response: %d chars", len(raw))
    return _parse_json_response(raw)


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