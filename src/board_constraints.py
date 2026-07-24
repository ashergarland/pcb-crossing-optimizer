"""Board constraints model for PCB placement.

Defines the physical constraints that the placement engine must respect:
board outline, mounting holes, fixed component positions, keepout zones,
and clearance rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .footprint_parser import FootprintGeometry


# =========================================================================
# Data model
# =========================================================================

@dataclass
class MountingHole:
    """A mounting hole with fixed position and diameter."""
    x: float  # mm
    y: float  # mm
    diameter: float  # mm


@dataclass
class FixedPosition:
    """A component locked to a specific position."""
    ref: str
    x: float  # mm
    y: float  # mm
    rotation: float = 0.0  # degrees
    layer: str = "F.Cu"


@dataclass
class KeepoutZone:
    """A rectangular keepout area where no components may be placed."""
    x: float  # mm (top-left)
    y: float  # mm (top-left)
    width: float  # mm
    height: float  # mm


@dataclass
class BoardConstraints:
    """All physical constraints for component placement."""
    # Board outline (width x height in mm). None = auto-size.
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None

    # Edge clearance — minimum distance from component courtyard to board edge
    edge_clearance_mm: float = 0.5

    # Mounting holes — components cannot overlap these
    mounting_holes: list[MountingHole] = field(default_factory=list)

    # Fixed component positions (connectors, switches, etc.)
    fixed_positions: list[FixedPosition] = field(default_factory=list)

    # Keepout zones
    keepout_zones: list[KeepoutZone] = field(default_factory=list)

    # Minimum clearance between component courtyards
    component_clearance_mm: float = 0.2


# =========================================================================
# Parsing / construction helpers
# =========================================================================

def parse_board_constraints(data: dict) -> BoardConstraints:
    """Parse a JSON-style dict into BoardConstraints.

    Expected format:
    {
        "width_mm": 50.0,
        "height_mm": 30.0,
        "edge_clearance_mm": 0.5,
        "component_clearance_mm": 0.2,
        "mounting_holes": [{"x": 3, "y": 3, "diameter": 3.2}, ...],
        "fixed_positions": [{"ref": "J1", "x": 0, "y": 15, "rotation": 0, "layer": "F.Cu"}, ...],
        "keepout_zones": [{"x": 10, "y": 10, "width": 5, "height": 5}, ...]
    }
    """
    holes = [
        MountingHole(x=h["x"], y=h["y"], diameter=h["diameter"])
        for h in data.get("mounting_holes", [])
    ]
    fixed = [
        FixedPosition(
            ref=f["ref"],
            x=f["x"],
            y=f["y"],
            rotation=f.get("rotation", 0.0),
            layer=f.get("layer", "F.Cu"),
        )
        for f in data.get("fixed_positions", [])
    ]
    keepouts = [
        KeepoutZone(x=k["x"], y=k["y"], width=k["width"], height=k["height"])
        for k in data.get("keepout_zones", [])
    ]

    return BoardConstraints(
        width_mm=data.get("width_mm"),
        height_mm=data.get("height_mm"),
        edge_clearance_mm=data.get("edge_clearance_mm", 0.5),
        component_clearance_mm=data.get("component_clearance_mm", 0.2),
        mounting_holes=holes,
        fixed_positions=fixed,
        keepout_zones=keepouts,
    )


def infer_board_size(
    geometries: dict[str, FootprintGeometry],
    constraints: BoardConstraints,
    margin_factor: float = 1.8,
) -> tuple[float, float]:
    """Estimate a reasonable board size from component areas.

    Uses total component area * margin_factor to estimate required board area,
    then produces a roughly square board (biased slightly wider).

    Args:
        geometries: {ref: FootprintGeometry} for all components.
        constraints: Board constraints (may already have width/height set).
        margin_factor: Multiplier for total component area (default 1.8 = 80% extra).

    Returns:
        (width_mm, height_mm)
    """
    if constraints.width_mm and constraints.height_mm:
        return (constraints.width_mm, constraints.height_mm)

    # Sum courtyard areas
    total_area = 0.0
    for geom in geometries.values():
        total_area += geom.width_mm * geom.height_mm

    # Apply margin
    target_area = total_area * margin_factor

    # Aim for ~4:3 aspect ratio
    width = (target_area * (4 / 3)) ** 0.5
    height = target_area / width

    # Enforce minimums
    width = max(width, 10.0)
    height = max(height, 10.0)

    # If one dimension is specified, compute the other
    if constraints.width_mm:
        return (constraints.width_mm, target_area / constraints.width_mm)
    if constraints.height_mm:
        return (target_area / constraints.height_mm, constraints.height_mm)

    return (round(width, 1), round(height, 1))
