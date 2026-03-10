# pyright: reportPrivateImportUsage=false
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Local abbreviation map (fast, no API needed) ────────────────────────
_LOCAL_MAP: dict[str, str] = {
    "BED RM":       "Bedroom",
    "BED ROOM":     "Bedroom",
    "BEDROOM":      "Bedroom",
    "BR":           "Bedroom",
    "M.BED":        "Master Bedroom",
    "MASTER BED":   "Master Bedroom",
    "MASTER BEDROOM": "Master Bedroom",
    "KIT":          "Kitchen",
    "KITCHEN":      "Kitchen",
    "KTCHN":        "Kitchen",
    "LIV":          "Living Room",
    "LIVING":       "Living Room",
    "LIVING ROOM":  "Living Room",
    "HALL":         "Hall",
    "DRAWING":      "Drawing Room",
    "DRAWING ROOM": "Drawing Room",
    "DINING":       "Dining Room",
    "DINING ROOM":  "Dining Room",
    "TOILET":       "Toilet",
    "WC":           "Toilet",
    "W.C.":         "Toilet",
    "BATH":         "Bathroom",
    "BATHROOM":     "Bathroom",
    "BATH ROOM":    "Bathroom",
    "BALCONY":      "Balcony",
    "BAL":          "Balcony",
    "POOJA":        "Pooja Room",
    "PUJA":         "Pooja Room",
    "STORE":        "Store Room",
    "STORE ROOM":   "Store Room",
    "UTILITY":      "Utility",
    "UTILITY ROOM": "Utility",
    "LOBBY":        "Lobby",
    "PASSAGE":      "Passage",
    "CORRIDOR":     "Corridor",
    "STAIRCASE":    "Staircase",
    "STAIRS":       "Staircase",
    "FOYER":        "Foyer",
    "ENTRANCE":     "Entrance",
    "GARAGE":       "Garage",
    "CAR PARK":     "Parking",
    "PARKING":      "Parking",
    "WASH":         "Wash Area",
    "WASH AREA":    "Wash Area",
    "LAUNDRY":      "Laundry",
    "STUDY":        "Study Room",
    "STUDY ROOM":   "Study Room",
    "GUEST":        "Guest Room",
    "GUEST ROOM":   "Guest Room",
    "SIT OUT":      "Sit Out",
    "SITOUT":       "Sit Out",
}


def _normalize_local(label: str) -> str | None:
    """Try to match a label against the local abbreviation map.

    Returns the normalized name or None if no match is found.
    """
    key = re.sub(r"[^A-Z0-9. ]", "", label.upper()).strip()
    key = re.sub(r"\s+", " ", key)
    if key in _LOCAL_MAP:
        return _LOCAL_MAP[key]
    # Fuzzy: strip trailing digits / numbers ("BED ROOM 1" → "BED ROOM")
    stripped = re.sub(r"\s*\d+$", "", key).strip()
    if stripped in _LOCAL_MAP:
        return _LOCAL_MAP[stripped]
    return None


class LabelNormalizer:
    def __init__(self):
        from app.config import settings
        self.api_key = (settings.gemini_api_key or "").strip()
        self.model: Optional[Any] = None
        self._client = None
        
        if self.api_key and self.api_key != "your_key_here":
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                self.model = "gemini-2.0-flash"
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
        else:
            logger.info("Gemini API key not set. Using local normalization only.")

    def normalize_labels(self, labels: List[str]) -> Dict[str, str]:
        """
        Takes a list of raw CAD labels and returns a dictionary mapping 
        raw label -> normalized label.
        Example: ["BED RM", "LIV"] -> {"BED RM": "Bedroom", "LIV": "Living Room"}
        """
        if not labels:
            return {}

        unique_labels = list(set([lbl for lbl in labels if lbl]))
        if not unique_labels:
            return {}

        # Phase 1: local normalization (instant, no API cost)
        result: Dict[str, str] = {}
        remaining: list[str] = []
        for lbl in unique_labels:
            local = _normalize_local(lbl)
            if local:
                result[lbl] = local
            else:
                remaining.append(lbl)

        if not remaining:
            return result

        # Phase 2: AI normalization for labels the local map couldn't handle
        if not self.model:
            # Fallback: title-case
            for lbl in remaining:
                result[lbl] = lbl.title()
            return result

        prompt = f"""
        Convert CAD room labels into standardized room names. 
        Sometimes they are abbreviations.
        
        Input labels: {json.dumps(remaining)}
        
        Return exactly a JSON dictionary mapping the original input label to the standardized room name.
        Do not include any Markdown blocks, just the raw JSON.
        """
        
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=[prompt]
            )
            text = response.text
            if text.startswith("```json"):
                text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
            elif text.startswith("```"):
                text = text[3:-3]
                
            normalized_dict = json.loads(text.strip())
            
            for lbl in remaining:
                result[lbl] = normalized_dict.get(lbl, lbl.title())
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to normalize labels via Gemini: {e}")
            for lbl in remaining:
                result[lbl] = lbl.title()
            return result
