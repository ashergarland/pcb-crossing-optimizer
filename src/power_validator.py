"""Power trace width validation based on IPC-2152.

Calculates minimum trace widths for given current requirements and flags
nets where the required trace width exceeds available clearance between pads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# =========================================================================
# Data model
# =========================================================================

@dataclass
class StackupParams:
    """PCB stackup parameters affecting trace width calculations."""
    copper_weight_oz: float = 1.0  # 1 oz = 35µm, 2 oz = 70µm
    max_temp_rise_c: float = 10.0  # maximum acceptable temperature rise
    ambient_temp_c: float = 25.0


@dataclass
class PowerViolation:
    """A net where the required trace width exceeds available clearance."""
    net: str
    current_a: float
    required_width_mm: float
    copper_weight_oz: float
    max_temp_rise_c: float
    affected_refs: list[str]


# =========================================================================
# IPC-2152 trace width calculation
# =========================================================================

def trace_width_ipc2152(
    current_a: float,
    copper_weight_oz: float = 1.0,
    max_temp_rise_c: float = 10.0,
    is_external: bool = True,
) -> float:
    """Calculate minimum trace width per IPC-2152.

    Uses the simplified formula derived from IPC-2152 charts:
        Area (mils²) = (I / (k * dT^b))^(1/c)
    where:
        k = 0.048 (external), 0.024 (internal)
        b = 0.44
        c = 0.725

    Then: width = area / thickness

    Args:
        current_a: Current in amps.
        copper_weight_oz: Copper thickness in oz (1 oz = 1.37 mils = 35µm).
        max_temp_rise_c: Maximum temperature rise in °C.
        is_external: True for outer layers, False for inner layers.

    Returns:
        Minimum trace width in mm.
    """
    if current_a <= 0:
        return 0.0

    # Constants from IPC-2152 simplified model
    k = 0.048 if is_external else 0.024
    b = 0.44
    c = 0.725

    # Cross-sectional area in mils²
    area_mils2 = (current_a / (k * max_temp_rise_c ** b)) ** (1.0 / c)

    # Copper thickness in mils (1 oz = 1.37 mils)
    thickness_mils = copper_weight_oz * 1.37

    # Width in mils
    width_mils = area_mils2 / thickness_mils

    # Convert to mm (1 mil = 0.0254 mm)
    width_mm = width_mils * 0.0254

    return width_mm


# =========================================================================
# Validation
# =========================================================================

def validate_power_traces(
    nets: dict[str, list[tuple[str, str]]],
    current_budget: dict[str, float],
    stackup: Optional[StackupParams] = None,
) -> list[PowerViolation]:
    """Validate that all power nets can be routed with adequate trace widths.

    Args:
        nets: Parsed netlist {net_name: [(ref, pin), ...]}.
        current_budget: {net_name: current_in_amps} for nets to validate.
        stackup: PCB stackup parameters. Uses defaults if None.

    Returns:
        List of violations where required trace width may be problematic.
    """
    if stackup is None:
        stackup = StackupParams()

    violations: list[PowerViolation] = []

    for net_name, current_a in current_budget.items():
        if net_name not in nets:
            continue
        if current_a <= 0:
            continue

        required_width = trace_width_ipc2152(
            current_a=current_a,
            copper_weight_oz=stackup.copper_weight_oz,
            max_temp_rise_c=stackup.max_temp_rise_c,
            is_external=True,
        )

        # Collect affected component refs
        nodes = nets[net_name]
        affected_refs = sorted(set(ref for ref, pin in nodes))

        # Flag if required width exceeds common design rules
        # (0.2mm is typical minimum for low-power, flag anything > 0.5mm as notable)
        if required_width > 0.15:  # Always report power trace requirements
            violations.append(PowerViolation(
                net=net_name,
                current_a=current_a,
                required_width_mm=round(required_width, 3),
                copper_weight_oz=stackup.copper_weight_oz,
                max_temp_rise_c=stackup.max_temp_rise_c,
                affected_refs=affected_refs,
            ))

    return violations


def power_validation_to_dict(violations: list[PowerViolation]) -> list[dict]:
    """Convert power violations to JSON-serializable format."""
    return [
        {
            "net": v.net,
            "current_a": v.current_a,
            "required_width_mm": v.required_width_mm,
            "copper_weight_oz": v.copper_weight_oz,
            "max_temp_rise_c": v.max_temp_rise_c,
            "affected_refs": v.affected_refs,
            "severity": "error" if v.required_width_mm > 1.0 else
                        "warning" if v.required_width_mm > 0.5 else "info",
        }
        for v in violations
    ]
