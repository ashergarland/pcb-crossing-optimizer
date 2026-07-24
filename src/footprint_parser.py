"""Footprint geometry parser for KiCad .kicad_mod files.

Extracts physical dimensions (courtyard bounding box, pad positions) from
footprint files. Used by the placement engine to compute component sizes
for overlap detection and spacing.

Supports both modern (footprint ...) and legacy (module ...) formats.
"""

from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# =========================================================================
# Data model
# =========================================================================

@dataclass
class PadGeometry:
    """A single pad's position and size within a footprint."""
    number: str
    x: float  # mm, relative to footprint origin
    y: float
    width: float  # mm
    height: float
    shape: str  # "rect", "roundrect", "oval", "circle", "custom"
    layers: list[str] = field(default_factory=list)


@dataclass
class FootprintGeometry:
    """Physical geometry of a footprint, extracted from .kicad_mod."""
    library: str
    name: str
    width_mm: float   # courtyard or bounding box width
    height_mm: float  # courtyard or bounding box height
    pads: list[PadGeometry] = field(default_factory=list)
    courtyard: Optional[tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y)
    description: str = ""


# =========================================================================
# Parsing
# =========================================================================

# Regex patterns for extracting data from .kicad_mod files
_PAD_MODERN_RE = re.compile(
    r'\(pad\s+"?([^"\s)]+)"?\s+\w+\s+\w+'
    r'\s+\(at\s+([-\d.]+)\s+([-\d.]+)(?:\s+[-\d.]+)?\)'
    r'\s+\(size\s+([-\d.]+)\s+([-\d.]+)\)'
    r'(?:\s+\(layers\s+([^)]+)\))?',
    re.DOTALL,
)

# Legacy format: (pad N type shape (at x y [rot]) (size w h) ...)
_PAD_LEGACY_RE = re.compile(
    r'\(pad\s+(\S+)\s+\w+\s+\w+'
    r'\s+\(at\s+([-\d.]+)\s+([-\d.]+)(?:\s+[-\d.]+)?\)'
    r'\s+\(size\s+([-\d.]+)\s+([-\d.]+)\)'
    r'(?:\s+\(layers\s+([^)]+)\))?',
    re.DOTALL,
)

# Courtyard lines: (fp_line (start x y) (end x y) (layer "F.CrtYd"|F.CrtYd) ...)
_CRTYD_LINE_RE = re.compile(
    r'\(fp_line\s+\(start\s+([-\d.]+)\s+([-\d.]+)\)\s+\(end\s+([-\d.]+)\s+([-\d.]+)\)\s+\(layer\s+"?(?:F\.CrtYd|B\.CrtYd)"?\)',
)

# Courtyard rect (KiCad 8+): (fp_rect (start x y) (end x y) (layer F.CrtYd) ...)
_CRTYD_RECT_RE = re.compile(
    r'\(fp_rect\s+\(start\s+([-\d.]+)\s+([-\d.]+)\)\s+\(end\s+([-\d.]+)\s+([-\d.]+)\)\s+\(layer\s+"?(?:F\.CrtYd|B\.CrtYd)"?\)',
)

_FP_NAME_RE = re.compile(r'\((?:footprint|module)\s+"?([^"\s)]+)"?')
_FP_DESCR_RE = re.compile(r'\(descr?\s+"([^"]*)"')


def parse_footprint_geometry(path: Path) -> Optional[FootprintGeometry]:
    """Parse a .kicad_mod file and extract physical geometry.

    Returns None if the file cannot be parsed.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    # Extract name
    m = _FP_NAME_RE.search(text[:500])
    if not m:
        return None
    fp_name = m.group(1)
    lib_name = path.parent.stem.replace(".pretty", "")

    # Description
    dm = _FP_DESCR_RE.search(text[:2000])
    description = dm.group(1) if dm else ""

    # Extract pads
    pads: list[PadGeometry] = []
    for match in _PAD_MODERN_RE.finditer(text):
        layers_str = match.group(6) or ""
        layers = [l.strip().strip('"') for l in layers_str.split() if l.strip()]
        pads.append(PadGeometry(
            number=match.group(1),
            x=float(match.group(2)),
            y=float(match.group(3)),
            width=float(match.group(4)),
            height=float(match.group(5)),
            shape="smd",
            layers=layers,
        ))

    # If modern regex didn't match, try legacy
    if not pads:
        for match in _PAD_LEGACY_RE.finditer(text):
            layers_str = match.group(6) or ""
            layers = [l.strip().strip('"') for l in layers_str.split() if l.strip()]
            pads.append(PadGeometry(
                number=match.group(1),
                x=float(match.group(2)),
                y=float(match.group(3)),
                width=float(match.group(4)),
                height=float(match.group(5)),
                shape="smd",
                layers=layers,
            ))

    # Extract courtyard bounding box from courtyard lines
    courtyard = _extract_courtyard(text)

    # If no courtyard found, compute bounding box from pads
    if courtyard is None and pads:
        courtyard = _bbox_from_pads(pads)

    if courtyard:
        min_x, min_y, max_x, max_y = courtyard
        width = max_x - min_x
        height = max_y - min_y
    elif pads:
        # Fallback: use pad extents + margin
        width = max(p.x + p.width / 2 for p in pads) - min(p.x - p.width / 2 for p in pads) + 0.5
        height = max(p.y + p.height / 2 for p in pads) - min(p.y - p.height / 2 for p in pads) + 0.5
    else:
        # No pads, no courtyard — use a default small size
        width = 1.0
        height = 1.0

    return FootprintGeometry(
        library=lib_name,
        name=fp_name,
        width_mm=round(width, 3),
        height_mm=round(height, 3),
        pads=pads,
        courtyard=courtyard,
        description=description,
    )


def _extract_courtyard(text: str) -> Optional[tuple[float, float, float, float]]:
    """Extract courtyard bounding box from fp_line or fp_rect on CrtYd layer."""
    # Try fp_rect first (newer KiCad)
    rect_matches = _CRTYD_RECT_RE.findall(text)
    if rect_matches:
        xs = []
        ys = []
        for x1, y1, x2, y2 in rect_matches:
            xs.extend([float(x1), float(x2)])
            ys.extend([float(y1), float(y2)])
        return (min(xs), min(ys), max(xs), max(ys))

    # Try fp_line on courtyard layer
    line_matches = _CRTYD_LINE_RE.findall(text)
    if line_matches:
        xs = []
        ys = []
        for x1, y1, x2, y2 in line_matches:
            xs.extend([float(x1), float(x2)])
            ys.extend([float(y1), float(y2)])
        return (min(xs), min(ys), max(xs), max(ys))

    return None


def _bbox_from_pads(pads: list[PadGeometry]) -> tuple[float, float, float, float]:
    """Compute bounding box from pad positions and sizes, with 0.25mm margin."""
    margin = 0.25
    min_x = min(p.x - p.width / 2 for p in pads) - margin
    min_y = min(p.y - p.height / 2 for p in pads) - margin
    max_x = max(p.x + p.width / 2 for p in pads) + margin
    max_y = max(p.y + p.height / 2 for p in pads) + margin
    return (min_x, min_y, max_x, max_y)


# =========================================================================
# Library discovery
# =========================================================================

def detect_footprint_dir(override: str = "") -> Optional[Path]:
    """Find the KiCad footprint library directory.

    Checks environment variables, then standard install locations.
    """
    if override:
        p = Path(override)
        if p.is_dir():
            return p
        return None

    for var in ("KICAD9_FOOTPRINT_DIR", "KICAD8_FOOTPRINT_DIR",
                "KICAD7_FOOTPRINT_DIR", "KICAD_FOOTPRINT_DIR"):
        val = os.environ.get(var)
        if val:
            p = Path(val)
            if p.is_dir():
                return p

    # Derive from symbol dir env var
    for var in ("KICAD9_SYMBOL_DIR", "KICAD8_SYMBOL_DIR",
                "KICAD7_SYMBOL_DIR", "KICAD_SYMBOL_DIR"):
        val = os.environ.get(var)
        if val:
            fp_dir = Path(val).parent / "footprints"
            if fp_dir.is_dir():
                return fp_dir

    system = platform.system()
    candidates: list[Path] = []
    if system == "Windows":
        pf = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        for ver in ("10.0", "9.0", "8.0", "7.0"):
            candidates.append(pf / "KiCad" / ver / "share" / "kicad" / "footprints")
    elif system == "Darwin":
        candidates.append(Path("/Applications/KiCad/KiCad.app/Contents/SharedSupport/footprints"))
    else:
        candidates.append(Path("/usr/share/kicad/footprints"))

    for c in candidates:
        if c.is_dir():
            return c
    return None


def get_footprint_geometry(
    library: str,
    name: str,
    footprint_dir: Optional[Path] = None,
) -> Optional[FootprintGeometry]:
    """Load geometry for a specific footprint from the KiCad library.

    Args:
        library: Library name (e.g. "Resistor_SMD")
        name: Footprint name (e.g. "R_0805_2012Metric")
        footprint_dir: Override path to footprints directory. Auto-detected if None.

    Returns:
        FootprintGeometry or None if not found.
    """
    if footprint_dir is None:
        footprint_dir = detect_footprint_dir()
    if footprint_dir is None:
        return None

    pretty_dir = footprint_dir / f"{library}.pretty"
    if not pretty_dir.is_dir():
        return None

    mod_file = pretty_dir / f"{name}.kicad_mod"
    if not mod_file.is_file():
        return None

    return parse_footprint_geometry(mod_file)


def resolve_footprint_geometry(
    footprint_str: str,
    footprint_dir: Optional[Path] = None,
) -> Optional[FootprintGeometry]:
    """Resolve a footprint string like "Resistor_SMD:R_0805_2012Metric" to geometry.

    Args:
        footprint_str: Colon-separated "library:footprint" string from a netlist.
        footprint_dir: Override path to footprints directory.
    """
    if ":" not in footprint_str:
        return None
    library, name = footprint_str.split(":", 1)
    return get_footprint_geometry(library, name, footprint_dir)
