"""
SVG Converter
=============
Converts DXF/DWG files to SVG format for downstream vector parsing and
raster segmentation.

Conversion paths (tried in priority order)
------------------------------------------
1. ezdxf drawing addon → SVGBackend          (modern ezdxf ≥ 1.0, pure-Python)
2. Inkscape CLI --export-filename             (if ``inkscape`` is on PATH)
3. LibreCAD CLI dxf2svg                       (if ``librecad`` is on PATH)
4. Pure-Python fallback: ezdxf entity walk   (always works, no external CLI)

DWG support
-----------
If the input is a DWG file the module first attempts conversion via
ODA/Teigha File Converter (``ODAFileConverter`` on PATH) to produce a DXF,
then proceeds with the DXF → SVG path above.

Coordinate convention
---------------------
DXF uses a Y-up coordinate system; SVG uses Y-down.  The generated SVG
wraps all geometry in::

    <g transform="scale(1,-1) translate(0,-<total_height>)">

so that numeric values in the SVG elements remain identical to DXF drawing
units.  Callers that read back the SVG (e.g. ``SVGParser``) must account for
this single transform when extracting coordinates.

Usage
-----
>>> from app.pipeline.svg_converter import convert_to_svg
>>> svg_path = convert_to_svg("floor_plan.dxf", "floor_plan.svg")
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def convert_to_svg(
    input_file: str,
    output_svg: str,
    *,
    segments: list[dict] | None = None,
) -> str:
    """
    Convert a DXF or DWG file to an SVG file.

    Parameters
    ----------
    input_file : absolute path to the DXF/DWG file.
    output_svg : desired output path for the SVG file.
    segments   : pre-extracted segment dicts from ``DXFParser`` used as
                 a last-resort fallback when ezdxf cannot open the file.

    Returns
    -------
    Absolute path to the saved SVG file.
    """
    input_path = Path(input_file).resolve()
    output_path = Path(output_svg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # DWG → DXF if needed
    dxf_path = str(input_path)
    if input_path.suffix.lower() == ".dwg":
        dxf_path = _convert_dwg_to_dxf(str(input_path), str(output_path.parent))

    # Try modern ezdxf SVGBackend
    if _try_ezdxf_svg_backend(dxf_path, str(output_path)):
        return str(output_path.resolve())

    # Try Inkscape CLI
    if _try_inkscape(dxf_path, str(output_path)):
        return str(output_path.resolve())

    # Try LibreCAD CLI
    if _try_librecad(dxf_path, str(output_path)):
        return str(output_path.resolve())

    # Always-available fallback: build SVG from ezdxf entity traversal
    _fallback_ezdxf_to_svg(dxf_path, str(output_path), precomputed_segments=segments)
    return str(output_path.resolve())


# ---------------------------------------------------------------------------
# DWG → DXF helper
# ---------------------------------------------------------------------------


def _convert_dwg_to_dxf(dwg_path: str, output_dir: str) -> str:
    """Try ODA/Teigha File Converter to produce a DXF from a DWG.

    Returns the DXF path on success, the original DWG path on failure
    (ezdxf can sometimes read newer DWG directly).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(dwg_path).stem
    expected_dxf = out_dir / f"{stem}.dxf"

    for oda in ("ODAFileConverter", "TeighaFileConverter"):
        if not shutil.which(oda):
            continue
        cmd = [
            oda,
            str(Path(dwg_path).parent),
            str(out_dir),
            "ACAD2018",
            "DXF",
            "0",   # no recurse
            "0",   # no audit
            Path(dwg_path).name,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60, check=True)
            if expected_dxf.exists():
                logger.info("DWG → DXF via %s: %s", oda, expected_dxf)
                return str(expected_dxf)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass

    logger.warning("ODA File Converter unavailable – proceeding with %s as-is.", dwg_path)
    return dwg_path


# ---------------------------------------------------------------------------
# Conversion strategies
# ---------------------------------------------------------------------------


def _try_ezdxf_svg_backend(dxf_path: str, svg_path: str) -> bool:
    """Use ezdxf drawing addon with SVGBackend (ezdxf ≥ 1.0)."""
    try:
        import ezdxf  # type: ignore[import-untyped]
        from ezdxf.addons.drawing import RenderContext, Frontend  # type: ignore[import-untyped]
        from ezdxf.addons.drawing.svg import SVGBackend  # type: ignore[import-untyped]

        doc = ezdxf.readfile(dxf_path)  # type: ignore[attr-defined]
        msp = doc.modelspace()
        ctx = RenderContext(doc)
        backend = SVGBackend()
        Frontend(ctx, backend).draw_layout(msp, finalize=True)
        svg_str: str = backend.get_string()  # type: ignore[call-arg,attr-defined]
        with open(svg_path, "w", encoding="utf-8") as fh:
            fh.write(svg_str)
        logger.info("SVG exported via ezdxf SVGBackend → %s", svg_path)
        return True
    except Exception as exc:
        logger.debug("ezdxf SVGBackend unavailable: %s", exc)
        return False


def _try_inkscape(dxf_path: str, svg_path: str) -> bool:
    """Convert DXF → SVG using Inkscape CLI."""
    inkscape = shutil.which("inkscape") or shutil.which("inkscape.exe")
    if not inkscape:
        return False
    cmd = [inkscape, "--export-filename", svg_path, dxf_path]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        if Path(svg_path).exists():
            logger.info("SVG exported via Inkscape → %s", svg_path)
            return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.debug("Inkscape failed: %s", exc)
    return False


def _try_librecad(dxf_path: str, svg_path: str) -> bool:
    """Convert DXF → SVG using LibreCAD CLI."""
    lc = shutil.which("librecad") or shutil.which("LibreCAD")
    if not lc:
        return False
    cmd = [lc, "dxf2svg", "-o", svg_path, dxf_path]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        if Path(svg_path).exists():
            logger.info("SVG exported via LibreCAD → %s", svg_path)
            return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.debug("LibreCAD failed: %s", exc)
    return False


# ---------------------------------------------------------------------------
# Pure-Python fallback: ezdxf entity traversal → hand-crafted SVG
# ---------------------------------------------------------------------------


def _fallback_ezdxf_to_svg(
    dxf_path: str,
    svg_path: str,
    *,
    precomputed_segments: list[dict] | None = None,
) -> None:
    """
    Build an SVG by walking ezdxf modelspace entities.

    Exports LINE / LWPOLYLINE / POLYLINE as ``<line>`` / ``<polyline>``,
    ARC as ``<path>``, TEXT / MTEXT as ``<text>``.

    Falls back to ``precomputed_segments`` when ezdxf cannot open the file.
    """
    # (x1,y1,x2,y2,layer)
    lines:   list[tuple[float, float, float, float, str]] = []
    # ([(x,y),...], layer, closed)
    polys:   list[tuple[list[tuple[float, float]], str, bool]] = []
    # (cx,cy,r,angle_start_deg,angle_end_deg,layer)
    arcs:    list[tuple[float, float, float, float, float, str]] = []
    # (x,y,text,layer)
    texts:   list[tuple[float, float, str, str]] = []

    try:
        import ezdxf  # type: ignore[import-untyped]

        doc = ezdxf.readfile(dxf_path)  # type: ignore[attr-defined]
        for ent in doc.modelspace():
            layer = str(ent.dxf.layer)
            etype = ent.dxftype()
            try:
                if etype == "LINE":
                    s, e = ent.dxf.start, ent.dxf.end
                    lines.append((float(s.x), float(s.y), float(e.x), float(e.y), layer))

                elif etype == "LWPOLYLINE":
                    pts = [(float(p[0]), float(p[1])) for p in ent.get_points("xy")]  # type: ignore[attr-defined]
                    if len(pts) >= 2:
                        closed = bool(ent.closed)  # type: ignore[attr-defined]
                        polys.append((pts, layer, closed))

                elif etype == "POLYLINE":
                    pts = [
                        (float(v.dxf.location.x), float(v.dxf.location.y))
                        for v in ent.vertices  # type: ignore[attr-defined]
                    ]
                    if len(pts) >= 2:
                        polys.append((pts, layer, bool(getattr(ent, "is_closed", False))))

                elif etype == "ARC":
                    cx, cy = float(ent.dxf.center.x), float(ent.dxf.center.y)
                    arcs.append((
                        cx, cy,
                        float(ent.dxf.radius),
                        float(ent.dxf.start_angle),
                        float(ent.dxf.end_angle),
                        layer,
                    ))

                elif etype in ("TEXT", "MTEXT"):
                    pos = ent.dxf.insert
                    raw = ent.plain_mtext() if etype == "MTEXT" else ent.dxf.text  # type: ignore[attr-defined]
                    if raw:
                        texts.append((float(pos.x), float(pos.y), str(raw), layer))

            except Exception:
                pass

    except Exception as exc:
        if precomputed_segments is None:
            logger.error("Cannot open DXF %s with ezdxf: %s", dxf_path, exc)
            Path(svg_path).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
            return
        lines = [
            (
                float(s["start"][0]), float(s["start"][1]),
                float(s["end"][0]),   float(s["end"][1]),
                str(s.get("layer", "0")),
            )
            for s in precomputed_segments
        ]

    if not lines and not polys and not arcs:
        logger.warning("SVG fallback: no geometry found – writing empty SVG.")
        Path(svg_path).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
        return

    # Compute drawing extents
    all_x: list[float] = []
    all_y: list[float] = []
    for x1, y1, x2, y2, _ in lines:
        all_x += [x1, x2];  all_y += [y1, y2]
    for pts, *_ in polys:
        all_x += [p[0] for p in pts];  all_y += [p[1] for p in pts]
    for cx, cy, r, *_ in arcs:
        all_x += [cx - r, cx + r];  all_y += [cy - r, cy + r]
    for tx, ty, *_ in texts:
        all_x.append(tx);  all_y.append(ty)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    W  = max(max_x - min_x, 1.0)
    H  = max(max_y - min_y, 1.0)
    mg = max(W, H) * 0.02           # 2 % margin on each side

    vb_x = min_x - mg
    vb_y = min_y - mg
    vb_w = W + 2 * mg
    vb_h = H + 2 * mg

    # Build SVG element tree
    svg_ns = "http://www.w3.org/2000/svg"
    root = Element(f"{{{svg_ns}}}svg")
    root.set("xmlns", svg_ns)
    root.set("viewBox", f"{vb_x:.4f} {vb_y:.4f} {vb_w:.4f} {vb_h:.4f}")
    root.set("width",   f"{vb_w:.4f}")
    root.set("height",  f"{vb_h:.4f}")

    # Y-flip group: DXF Y-up → SVG Y-down
    # After this transform, element coordinates == DXF drawing units.
    ty_flip = -(vb_y * 2 + vb_h)
    g = SubElement(root, f"{{{svg_ns}}}g")
    g.set("transform", f"scale(1,-1) translate(0,{ty_flip:.4f})")
    g.set("stroke", "#000000")
    g.set("stroke-width", "0.1")
    g.set("fill", "none")

    for x1, y1, x2, y2, layer in lines:
        el = SubElement(g, f"{{{svg_ns}}}line")
        el.set("x1", f"{x1:.4f}");  el.set("y1", f"{y1:.4f}")
        el.set("x2", f"{x2:.4f}");  el.set("y2", f"{y2:.4f}")
        el.set("data-layer", layer)

    for pts, layer, closed in polys:
        tag = f"{{{svg_ns}}}polygon" if closed else f"{{{svg_ns}}}polyline"
        el = SubElement(g, tag)
        el.set("points", " ".join(f"{p[0]:.4f},{p[1]:.4f}" for p in pts))
        el.set("data-layer", layer)

    for cx, cy, r, a1, a2, layer in arcs:
        d = _arc_path_d(cx, cy, r, a1, a2)
        el = SubElement(g, f"{{{svg_ns}}}path")
        el.set("d", d)
        el.set("data-layer", layer)

    for tx, ty, text, layer in texts:
        el = SubElement(g, f"{{{svg_ns}}}text")
        el.set("x", f"{tx:.4f}")
        el.set("y", f"{ty:.4f}")
        el.set("data-layer", layer)
        el.set("font-size", "1")
        el.set("fill",   "#333333")
        el.set("stroke", "none")
        el.text = text

    # Pretty-print (Python ≥ 3.9) – silently skipped on older versions
    try:
        from xml.etree.ElementTree import indent as _indent
        _indent(root, space="  ")
    except (ImportError, TypeError):
        pass

    tree = ElementTree(root)
    tree.write(svg_path, encoding="unicode", xml_declaration=True)
    logger.info(
        "SVG fallback export → %s  (%d lines, %d polys, %d arcs, %d texts)",
        svg_path, len(lines), len(polys), len(arcs), len(texts),
    )


# ---------------------------------------------------------------------------
# SVG arc path helper
# ---------------------------------------------------------------------------


def _arc_path_d(
    cx: float, cy: float, r: float, a1_deg: float, a2_deg: float
) -> str:
    """Return SVG path ``d`` attribute for a circular arc."""
    a1 = math.radians(a1_deg)
    a2 = math.radians(a2_deg)
    if a2 <= a1:
        a2 += 2.0 * math.pi
    sx = cx + r * math.cos(a1);  sy = cy + r * math.sin(a1)
    ex = cx + r * math.cos(a2);  ey = cy + r * math.sin(a2)
    large = 1 if (a2 - a1) > math.pi else 0
    return f"M {sx:.4f} {sy:.4f} A {r:.4f} {r:.4f} 0 {large} 1 {ex:.4f} {ey:.4f}"
