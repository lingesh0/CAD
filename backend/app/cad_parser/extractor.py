import math
import ezdxf
import ezdxf.tools.text
import logging
from typing import Any

from shapely.geometry import Polygon as ShapelyPolygon

logger = logging.getLogger(__name__)

# Keywords that indicate a door-related block name.
_DOOR_BLOCK_KEYWORDS = {"DOOR", "DR", "SWING", "ENTRY", "GATE"}


class DXFExtractor:
    """Extracts all geometry, text, and door arcs from a DXF file."""

    # Closed polylines with area below this threshold are treated as
    # hatch fill / annotation noise and skipped during segment extraction.
    MIN_CLOSED_POLY_AREA = 0.04  # in drawing units²

    # Open 2-vertex polylines (single segments) shorter than this are
    # typically text character strokes, dimension ticks, or hatch fills.
    MIN_OPEN_SEG_LENGTH = 0.05  # in drawing units

    # ARCs whose sweep angle (°) is in this range are candidate door swings.
    _DOOR_ARC_MIN_ANGLE = 70.0
    _DOOR_ARC_MAX_ANGLE = 100.0

    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            self.doc = ezdxf.readfile(file_path)
            self.msp = self.doc.modelspace()
        except IOError:
            logger.error(f"Not a DXF file or a generic I/O error: {file_path}")
            raise
        except ezdxf.DXFStructureError:
            logger.error(f"Invalid or corrupted DXF file: {file_path}")
            raise
        self._visited_blocks: set[str] = set()

    @staticmethod
    def _insunits_to_feet_factor(insunits: int) -> float:
        mapping = {
            0: 3.28084,     # Unitless (fallback assumption: meters)
            1: 1.0 / 12.0,  # Inches
            2: 1.0,         # Feet
            4: 0.00328084,  # Millimeters
            5: 0.0328084,   # Centimeters
            6: 3.28084,     # Meters
        }
        return mapping.get(insunits, 1.0)

    def extract_geometry(self) -> dict:
        """Return extracted geometry, labels, doors, and unit metadata."""
        insunits = int(self.doc.header.get("$INSUNITS", 0) or 0)
        linear_to_feet = self._insunits_to_feet_factor(insunits)
        geometry: dict[str, list[Any]] = {
            "segments": [],  # [{start:[x,y], end:[x,y], layer:str}, ...]
            "texts": [],     # [{text:str, position:[x,y], layer:str}, ...]
            "doors": [],     # [{center:[x,y], radius:float, layer:str}, ...]
        }
        metadata = {
            "insunits": insunits,
            "linear_to_feet": linear_to_feet,
            "area_to_sqft": linear_to_feet * linear_to_feet,
        }
        self._visited_blocks.clear()
        self._seen_text_keys: set[tuple] = set()  # dedup (text, rx, ry)
        self._process_layout(self.msp, geometry, offset_x=0.0, offset_y=0.0, is_block_def=False)
        logger.info(
            "Extracted %d segments, %d texts, %d doors from %s (INSUNITS=%d)",
            len(geometry["segments"]), len(geometry["texts"]),
            len(geometry["doors"]), self.file_path, insunits,
        )
        geometry["metadata"] = metadata
        return geometry

    def _add_text(self, geometry: dict, text: str, x: float, y: float, layer: str) -> None:
        """Add a text entry only if it hasn't been seen at this position before."""
        key = (text.upper().strip(), round(x, 2), round(y, 2))
        if key in self._seen_text_keys:
            return
        self._seen_text_keys.add(key)
        geometry["texts"].append({"text": text, "position": [x, y], "layer": layer})

    # ------------------------------------------------------------------
    def _process_layout(
        self, layout: Any, geometry: dict, *,
        offset_x: float, offset_y: float,
        is_block_def: bool = False,
    ) -> None:
        # --- LINE entities ---
        for entity in layout.query("LINE"):
            try:
                geometry["segments"].append({
                    "start": [entity.dxf.start.x + offset_x, entity.dxf.start.y + offset_y],
                    "end":   [entity.dxf.end.x + offset_x,   entity.dxf.end.y + offset_y],
                    "layer": entity.dxf.layer,
                })
            except Exception:
                pass

        # --- LWPOLYLINE / POLYLINE → decompose into segments ---
        for entity in layout.query("LWPOLYLINE POLYLINE"):
            try:
                if entity.dxftype() == "LWPOLYLINE":
                    pts = [(p[0] + offset_x, p[1] + offset_y) for p in entity.get_points()]
                    is_closed = entity.closed
                else:
                    pts = [(p.x + offset_x, p.y + offset_y) for p in entity.points()]
                    is_closed = entity.is_closed
                if len(pts) < 2:
                    continue

                # --- Short open segment filter ---
                if not is_closed and len(pts) == 2:
                    dx = pts[1][0] - pts[0][0]
                    dy = pts[1][1] - pts[0][1]
                    if (dx * dx + dy * dy) < self.MIN_OPEN_SEG_LENGTH ** 2:
                        continue

                # --- Hatch / annotation filter ---
                if is_closed and len(pts) >= 3:
                    try:
                        poly = ShapelyPolygon(pts)
                        if poly.is_valid and poly.area < self.MIN_CLOSED_POLY_AREA:
                            continue
                    except Exception:
                        pass

                for i in range(len(pts) - 1):
                    geometry["segments"].append({
                        "start": list(pts[i]),
                        "end":   list(pts[i + 1]),
                        "layer": entity.dxf.layer,
                    })
                if is_closed and len(pts) >= 3 and pts[0] != pts[-1]:
                    geometry["segments"].append({
                        "start": list(pts[-1]),
                        "end":   list(pts[0]),
                        "layer": entity.dxf.layer,
                    })
            except Exception as e:
                logger.debug("Failed to extract polyline: %s", e)

        # --- ARC entities (door swing detection) ---
        for entity in layout.query("ARC"):
            try:
                cx = entity.dxf.center.x + offset_x
                cy = entity.dxf.center.y + offset_y
                r = entity.dxf.radius
                start_angle = entity.dxf.start_angle
                end_angle = entity.dxf.end_angle
                sweep = (end_angle - start_angle) % 360.0
                if self._DOOR_ARC_MIN_ANGLE <= sweep <= self._DOOR_ARC_MAX_ANGLE:
                    geometry["doors"].append({
                        "center": [cx, cy],
                        "radius": r,
                        "start_angle": start_angle,
                        "end_angle": end_angle,
                        "sweep": sweep,
                        "layer": entity.dxf.layer,
                    })
            except Exception as e:
                logger.debug("Failed to extract arc: %s", e)

        # --- CIRCLE entities (small circles can also be door markers) ---
        for entity in layout.query("CIRCLE"):
            try:
                r = entity.dxf.radius
                layer = entity.dxf.layer.upper()
                if any(kw in layer for kw in _DOOR_BLOCK_KEYWORDS):
                    cx = entity.dxf.center.x + offset_x
                    cy = entity.dxf.center.y + offset_y
                    geometry["doors"].append({
                        "center": [cx, cy],
                        "radius": r,
                        "start_angle": 0.0,
                        "end_angle": 360.0,
                        "sweep": 360.0,
                        "layer": entity.dxf.layer,
                    })
            except Exception as e:
                logger.debug("Failed to extract circle: %s", e)

        # --- TEXT / MTEXT ---
        for entity in layout.query("TEXT MTEXT"):
            try:
                if entity.dxftype() == "MTEXT":
                    raw = entity.text
                else:
                    raw = entity.dxf.text
                pos = entity.dxf.insert
                text_val = ezdxf.tools.text.plain_text(raw).strip()
                if text_val:
                    self._add_text(
                        geometry, text_val,
                        pos.x + offset_x, pos.y + offset_y,
                        entity.dxf.layer,
                    )
            except Exception as e:
                logger.debug("Failed to extract text: %s", e)

        # --- standalone ATTRIB (actual values) and ATTDEF (templates) ---
        # Skip ATTDEF inside block definitions — they are tag templates, not values.
        entity_types = "ATTRIB" if is_block_def else "ATTRIB ATTDEF"
        for entity in layout.query(entity_types):
            try:
                txt = ezdxf.tools.text.plain_text(entity.dxf.text).strip()
                pos = entity.dxf.insert
                if txt:
                    self._add_text(
                        geometry, txt,
                        pos.x + offset_x, pos.y + offset_y,
                        entity.dxf.layer,
                    )
            except Exception as e:
                logger.debug("Failed to extract attrib/attdef: %s", e)

        # --- INSERT (block references) – recurse with offset ---
        for entity in layout.query("INSERT"):
            try:
                block_name = entity.dxf.name
                ins = entity.dxf.insert

                # Detect door blocks by block name heuristic
                upper_name = block_name.upper()
                if any(kw in upper_name for kw in _DOOR_BLOCK_KEYWORDS):
                    geometry["doors"].append({
                        "center": [ins.x + offset_x, ins.y + offset_y],
                        "radius": 0.0,  # unknown from block reference
                        "start_angle": 0.0,
                        "end_angle": 90.0,
                        "sweep": 90.0,
                        "layer": entity.dxf.layer,
                        "block_name": block_name,
                    })

                # Extract ATTRIB values directly from the INSERT reference.
                # Use the ATTRIB's own insert position; fall back to the INSERT
                # position if the attribute has no explicit placement.
                if hasattr(entity, "attribs"):
                    for attrib in entity.attribs:
                        try:
                            txt = ezdxf.tools.text.plain_text(attrib.dxf.text).strip()
                            if not txt:
                                continue
                            # Prefer attrib's own position; fall back to INSERT position
                            try:
                                apos = attrib.dxf.insert
                                ax, ay = apos.x + offset_x, apos.y + offset_y
                            except Exception:
                                ax, ay = ins.x + offset_x, ins.y + offset_y
                            self._add_text(geometry, txt, ax, ay, attrib.dxf.layer)
                        except Exception:
                            pass

                # Recurse into block definition (geometry only — ATTDEF skipped there)
                if block_name in self._visited_blocks:
                    continue
                self._visited_blocks.add(block_name)
                block = self.doc.blocks.get(block_name)
                if block:
                    self._process_layout(
                        block, geometry,
                        offset_x=offset_x + ins.x,
                        offset_y=offset_y + ins.y,
                        is_block_def=True,
                    )
                self._visited_blocks.discard(block_name)
            except Exception as e:
                logger.debug("Failed to process block: %s", e)
