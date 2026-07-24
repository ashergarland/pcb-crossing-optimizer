"""Tests for the placement engine, footprint parser, and power validator."""

from pathlib import Path

from crossing_analyzer import parse_netlist
from crossing_analyzer.footprint_parser import (
    FootprintGeometry,
    PadGeometry,
    parse_footprint_geometry,
    resolve_footprint_geometry,
    detect_footprint_dir,
)
from crossing_analyzer.board_constraints import (
    BoardConstraints,
    FixedPosition,
    infer_board_size,
    parse_board_constraints,
)
from crossing_analyzer.placement_engine import (
    PlacementResult,
    place_components,
    placement_to_dict,
)
from crossing_analyzer.power_validator import (
    StackupParams,
    trace_width_ipc2152,
    validate_power_traces,
    power_validation_to_dict,
)


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


# =========================================================================
# Footprint parser tests
# =========================================================================

def test_parse_footprint_geometry_from_fixture():
    """Parse a real .kicad_mod file if KiCad footprints are available."""
    fp_dir = detect_footprint_dir()
    if fp_dir is None:
        # No KiCad installed, test with a synthetic geometry
        return

    # Try to find a common footprint
    r0805 = fp_dir / "Resistor_SMD.pretty" / "R_0805_2012Metric.kicad_mod"
    if not r0805.exists():
        return

    geom = parse_footprint_geometry(r0805)
    assert geom is not None
    assert geom.name == "R_0805_2012Metric"
    assert geom.library == "Resistor_SMD"
    assert geom.width_mm > 0
    assert geom.height_mm > 0
    assert len(geom.pads) >= 2


def test_resolve_footprint_geometry():
    """resolve_footprint_geometry splits library:name correctly."""
    fp_dir = detect_footprint_dir()
    if fp_dir is None:
        return

    geom = resolve_footprint_geometry("Resistor_SMD:R_0805_2012Metric", fp_dir)
    if geom:  # Only assert if KiCad has this footprint
        assert geom.name == "R_0805_2012Metric"


# =========================================================================
# Board constraints tests
# =========================================================================

def test_parse_board_constraints():
    """Parse a constraints dict."""
    data = {
        "width_mm": 50.0,
        "height_mm": 30.0,
        "edge_clearance_mm": 1.0,
        "mounting_holes": [{"x": 3, "y": 3, "diameter": 3.2}],
        "fixed_positions": [{"ref": "J1", "x": 0, "y": 15, "rotation": 90}],
        "keepout_zones": [{"x": 20, "y": 10, "width": 5, "height": 5}],
    }
    c = parse_board_constraints(data)
    assert c.width_mm == 50.0
    assert c.height_mm == 30.0
    assert len(c.mounting_holes) == 1
    assert len(c.fixed_positions) == 1
    assert c.fixed_positions[0].ref == "J1"
    assert len(c.keepout_zones) == 1


def test_infer_board_size():
    """infer_board_size produces reasonable dimensions from component areas."""
    geometries = {
        "J1": FootprintGeometry("Connector", "1x04", 2.54, 10.0),
        "J2": FootprintGeometry("Connector", "1x04", 2.54, 10.0),
        "R1": FootprintGeometry("Resistor_SMD", "R_0805", 2.0, 1.5),
        "R2": FootprintGeometry("Resistor_SMD", "R_0805", 2.0, 1.5),
        "C1": FootprintGeometry("Capacitor_SMD", "C_0805", 2.0, 1.5),
    }
    constraints = BoardConstraints()
    w, h = infer_board_size(geometries, constraints)
    assert w >= 10.0
    assert h >= 10.0
    # Total component area is ~57 mm², board should be ~100+ mm²
    assert w * h > 60


def test_infer_board_size_respects_fixed_dimensions():
    """If width is set, only height is inferred."""
    geometries = {"R1": FootprintGeometry("R", "R_0805", 2.0, 1.5)}
    constraints = BoardConstraints(width_mm=40.0)
    w, h = infer_board_size(geometries, constraints)
    assert w == 40.0


# =========================================================================
# Placement engine tests
# =========================================================================

def test_place_components_synthetic():
    """Place synthetic components and verify basic properties."""
    nets = {
        "SDA": [("J1", "1"), ("J2", "3")],
        "SCL": [("J1", "2"), ("J2", "4")],
        "VCC": [("J1", "3"), ("J2", "1"), ("C1", "1")],
        "GND": [("J1", "4"), ("J2", "2"), ("C1", "2")],
    }
    component_info = {
        "J1": {"value": "HOST", "part": "Conn_01x04"},
        "J2": {"value": "DEVICE", "part": "Conn_01x04"},
        "C1": {"value": "100n", "part": "C"},
    }
    geometries = {
        "J1": FootprintGeometry("Connector", "1x04", 2.54, 10.0),
        "J2": FootprintGeometry("Connector", "1x04", 2.54, 10.0),
        "C1": FootprintGeometry("Capacitor_SMD", "C_0805", 2.0, 1.5),
    }
    constraints = BoardConstraints(width_mm=30.0, height_mm=20.0)

    result = place_components(
        nets, component_info, geometries, constraints,
        annealing_iterations=1000,
    )

    assert isinstance(result, PlacementResult)
    assert len(result.positions) == 3
    assert result.metrics.overlap_count == 0
    assert result.metrics.out_of_bounds_count == 0

    # All components should be within board bounds
    for ref, pos in result.positions.items():
        assert 0 <= pos["x"] <= 30.0
        assert 0 <= pos["y"] <= 20.0


def test_place_components_with_fixed():
    """Fixed components stay at their specified positions."""
    nets = {"NET1": [("J1", "1"), ("R1", "1")]}
    component_info = {"J1": {"value": "X", "part": "Conn"}, "R1": {"value": "10k", "part": "R"}}
    geometries = {
        "J1": FootprintGeometry("C", "1x04", 2.54, 10.0),
        "R1": FootprintGeometry("R", "R_0805", 2.0, 1.5),
    }
    constraints = BoardConstraints(
        width_mm=30.0,
        height_mm=20.0,
        fixed_positions=[FixedPosition(ref="J1", x=2.0, y=10.0, rotation=0)],
    )

    result = place_components(nets, component_info, geometries, constraints, annealing_iterations=500)

    assert result.positions["J1"]["x"] == 2.0
    assert result.positions["J1"]["y"] == 10.0


def test_place_components_i2c_breakout():
    """End-to-end placement test using the i2c_breakout example netlist."""
    netlist_path = EXAMPLES_DIR / "i2c_breakout.net"
    if not netlist_path.exists():
        return

    parsed = parse_netlist(str(netlist_path))
    nets = parsed["nets"]
    components = parsed["components"]

    # Create synthetic geometries (since we may not have KiCad installed)
    geometries = {}
    for ref, info in components.items():
        if ref.startswith("J"):
            geometries[ref] = FootprintGeometry("Connector", "1x04", 2.54, 10.0)
        elif ref.startswith("R"):
            geometries[ref] = FootprintGeometry("Resistor_SMD", "R_0805", 2.0, 1.5)
        elif ref.startswith("C"):
            geometries[ref] = FootprintGeometry("Capacitor_SMD", "C_0805", 2.0, 1.5)
        elif ref.startswith("TP"):
            geometries[ref] = FootprintGeometry("TestPoint", "TP_1.5mm", 1.5, 1.5)
        else:
            geometries[ref] = FootprintGeometry("Generic", ref, 3.0, 3.0)

    constraints = BoardConstraints(width_mm=35.0, height_mm=25.0)

    result = place_components(
        nets, components, geometries, constraints,
        annealing_iterations=5000,
    )

    assert result.metrics.overlap_count == 0
    assert result.metrics.out_of_bounds_count == 0
    assert result.metrics.total_wire_length_mm > 0

    # Verify serialization
    d = placement_to_dict(result)
    assert "positions" in d
    assert "metrics" in d
    assert "board" in d
    assert len(d["positions"]) == len(components)


# =========================================================================
# Power validator tests
# =========================================================================

def test_trace_width_basic():
    """IPC-2152 trace width for 1A at 1oz copper should be reasonable."""
    w = trace_width_ipc2152(1.0, copper_weight_oz=1.0, max_temp_rise_c=10.0)
    # 1A at 1oz/10°C rise should be roughly 0.5-1.0mm
    assert 0.2 < w < 2.0


def test_trace_width_zero_current():
    """Zero current should give zero width."""
    assert trace_width_ipc2152(0.0) == 0.0


def test_trace_width_scales_with_current():
    """Higher current requires wider trace."""
    w1 = trace_width_ipc2152(0.5)
    w2 = trace_width_ipc2152(2.0)
    assert w2 > w1


def test_validate_power_traces():
    """Validate power traces with a current budget."""
    nets = {
        "VCC": [("J1", "3"), ("U1", "1"), ("C1", "1")],
        "GND": [("J1", "4"), ("U1", "2"), ("C1", "2")],
        "SDA": [("J1", "1"), ("U1", "3")],
    }
    budget = {"VCC": 0.5, "GND": 1.0}

    violations = validate_power_traces(nets, budget)
    assert len(violations) >= 1  # At least GND at 1A should flag

    # Verify structure
    for v in violations:
        assert v.net in budget
        assert v.required_width_mm > 0
        assert len(v.affected_refs) > 0


def test_power_validation_to_dict():
    """JSON serialization of power violations."""
    nets = {"VCC": [("J1", "1"), ("U1", "1")]}
    violations = validate_power_traces(nets, {"VCC": 2.0})
    d = power_validation_to_dict(violations)
    assert len(d) >= 1
    assert "severity" in d[0]
    assert d[0]["net"] == "VCC"


# =========================================================================
# Placement JSON output structure tests
# =========================================================================

def test_placement_to_dict_structure():
    """Verify the JSON output structure matches what AI agents expect."""
    nets = {"NET1": [("R1", "1"), ("R2", "1")]}
    component_info = {"R1": {"value": "10k", "part": "R"}, "R2": {"value": "10k", "part": "R"}}
    geometries = {
        "R1": FootprintGeometry("R", "R_0805", 2.0, 1.5),
        "R2": FootprintGeometry("R", "R_0805", 2.0, 1.5),
    }
    constraints = BoardConstraints(width_mm=20.0, height_mm=15.0)

    result = place_components(nets, component_info, geometries, constraints, annealing_iterations=100)
    d = placement_to_dict(result)

    # Top-level keys
    assert set(d.keys()) == {"board", "positions", "metrics", "decoupling_issues", "groups"}

    # Board
    assert d["board"]["width_mm"] == 20.0
    assert d["board"]["height_mm"] == 15.0

    # Each position has x, y, rotation, layer
    for ref, pos in d["positions"].items():
        assert "x" in pos
        assert "y" in pos
        assert "rotation" in pos
        assert "layer" in pos

    # Metrics
    assert "total_wire_length_mm" in d["metrics"]
    assert "overlap_count" in d["metrics"]
