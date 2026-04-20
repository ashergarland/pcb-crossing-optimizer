"""Tests for the crossing analyzer core logic."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crossing_analyzer import (
    PinColumn,
    Edge,
    CrossingPair,
    count_crossings,
    compute_optimal_order,
    analyze_connectors,
    parse_netlist,
    extract_edges,
    infer_pin_order,
)


# =========================================================================
# Unit tests: crossing counter
# =========================================================================

def test_no_crossings_parallel():
    """Parallel edges (same order on both sides) produce zero crossings."""
    fixed = PinColumn(ref="J1", pin_order=["1", "2", "3"])
    reorderable = PinColumn(ref="J2", pin_order=["1", "2", "3"])
    edges = [
        Edge("A", "1", "1"),
        Edge("B", "2", "2"),
        Edge("C", "3", "3"),
    ]
    crossings = count_crossings(fixed, reorderable, edges)
    assert len(crossings) == 0


def test_single_crossing():
    """Two edges that cross: (1->2) and (2->1)."""
    fixed = PinColumn(ref="J1", pin_order=["1", "2"])
    reorderable = PinColumn(ref="J2", pin_order=["1", "2"])
    edges = [
        Edge("A", "1", "2"),
        Edge("B", "2", "1"),
    ]
    crossings = count_crossings(fixed, reorderable, edges)
    assert len(crossings) == 1


def test_multiple_crossings():
    """Fully reversed order: 3 edges produce 3 crossing pairs."""
    fixed = PinColumn(ref="J1", pin_order=["1", "2", "3"])
    reorderable = PinColumn(ref="J2", pin_order=["1", "2", "3"])
    edges = [
        Edge("A", "1", "3"),
        Edge("B", "2", "2"),
        Edge("C", "3", "1"),
    ]
    crossings = count_crossings(fixed, reorderable, edges)
    # A x B, A x C, B x C
    assert len(crossings) == 3


def test_microsd_crossings():
    """Reproduce the microSD breakout crossing scenario (original pin order)."""
    # J1 pins 2-7 (excluding DAT2/DAT1 which don't go to J2)
    fixed = PinColumn(ref="J1", pin_order=["2", "3", "4", "5", "6", "7"])
    # J2 original order: CS, MOSI, VDD, GND, MISO, SCK
    reorderable = PinColumn(ref="J2", pin_order=["1", "2", "3", "4", "5", "6"])
    edges = [
        Edge("CS",   "2", "1"),
        Edge("MOSI", "3", "2"),
        Edge("VDD",  "4", "3"),
        Edge("SCK",  "5", "6"),  # SCK jumps to pin 6
        Edge("GND",  "6", "4"),
        Edge("MISO", "7", "5"),
    ]
    crossings = count_crossings(fixed, reorderable, edges)
    assert len(crossings) == 2
    # Verify which pairs cross
    net_pairs = {(c.edge_a.net_name, c.edge_b.net_name) for c in crossings}
    assert ("GND", "SCK") in net_pairs or ("SCK", "GND") in net_pairs
    assert ("MISO", "SCK") in net_pairs or ("SCK", "MISO") in net_pairs


# =========================================================================
# Unit tests: optimizer
# =========================================================================

def test_optimal_order_simple():
    """Optimizer produces monotonic order for simple 1:1 mapping."""
    fixed = PinColumn(ref="J1", pin_order=["1", "2", "3"])
    edges = [
        Edge("A", "1", "a"),
        Edge("B", "2", "b"),
        Edge("C", "3", "c"),
    ]
    optimal = compute_optimal_order(fixed, edges)
    assert optimal == ["a", "b", "c"]


def test_optimal_order_reversal():
    """Optimizer corrects a fully reversed mapping."""
    fixed = PinColumn(ref="J1", pin_order=["1", "2", "3"])
    edges = [
        Edge("A", "1", "c"),
        Edge("B", "2", "b"),
        Edge("C", "3", "a"),
    ]
    optimal = compute_optimal_order(fixed, edges)
    assert optimal == ["c", "b", "a"]


def test_optimal_order_microsd():
    """Optimizer produces the correct fix for the microSD breakout."""
    fixed = PinColumn(ref="J1", pin_order=["2", "3", "4", "5", "6", "7"])
    edges = [
        Edge("CS",   "2", "1"),
        Edge("MOSI", "3", "2"),
        Edge("VDD",  "4", "3"),
        Edge("SCK",  "5", "6"),
        Edge("GND",  "6", "4"),
        Edge("MISO", "7", "5"),
    ]
    optimal = compute_optimal_order(fixed, edges)
    # Optimal should put SCK(pos 3) before GND(pos 4) before MISO(pos 5)
    # So: 1, 2, 3, 6, 4, 5
    assert optimal == ["1", "2", "3", "6", "4", "5"]


def test_optimal_eliminates_crossings():
    """Verify that the optimal order produces zero crossings."""
    fixed = PinColumn(ref="J1", pin_order=["2", "3", "4", "5", "6", "7"])
    edges = [
        Edge("CS",   "2", "1"),
        Edge("MOSI", "3", "2"),
        Edge("VDD",  "4", "3"),
        Edge("SCK",  "5", "6"),
        Edge("GND",  "6", "4"),
        Edge("MISO", "7", "5"),
    ]
    optimal_order = compute_optimal_order(fixed, edges)
    optimized = PinColumn(ref="J2", pin_order=optimal_order)
    crossings = count_crossings(fixed, optimized, edges)
    assert len(crossings) == 0


# =========================================================================
# Unit tests: full analysis
# =========================================================================

def test_analyze_connectors_report():
    """Full analysis returns correct report structure."""
    fixed = PinColumn(ref="J1", pin_order=["1", "2", "3"])
    reorderable = PinColumn(ref="J2", pin_order=["1", "2", "3"])
    edges = [
        Edge("A", "1", "2"),
        Edge("B", "2", "1"),
        Edge("C", "3", "3"),
    ]
    report = analyze_connectors(fixed, reorderable, edges)
    assert report.crossing_count == 1
    assert report.optimal_crossing_count == 0
    assert report.fixed_ref == "J1"
    assert report.reorderable_ref == "J2"


# =========================================================================
# Integration tests: netlist parsing
# =========================================================================

def test_parse_microsd_netlist():
    """Parse the microSD example netlist and verify structure."""
    netlist_path = Path(__file__).resolve().parent.parent / "examples" / "microsd_breakout.net"
    if not netlist_path.exists():
        return  # skip if example not available

    data = parse_netlist(str(netlist_path))

    # Check components
    assert "J1" in data["components"]
    assert "J2" in data["components"]
    assert "R1" in data["components"]
    assert "C1" in data["components"]

    # Check nets
    assert "CS" in data["nets"]
    assert "GND" in data["nets"]
    assert "VDD" in data["nets"]
    assert "MOSI" in data["nets"]
    assert "MISO" in data["nets"]
    assert "SCK" in data["nets"]


def test_parse_and_analyze_microsd():
    """End-to-end: parse microSD netlist and verify 0 crossings (fixed version)."""
    netlist_path = Path(__file__).resolve().parent.parent / "examples" / "microsd_breakout.net"
    if not netlist_path.exists():
        return

    data = parse_netlist(str(netlist_path))

    fixed_pins = [p for p in infer_pin_order("J1", data["nets"]) if p != "SH"]
    reorderable_pins = infer_pin_order("J2", data["nets"])

    fixed = PinColumn(ref="J1", pin_order=fixed_pins)
    reorderable = PinColumn(ref="J2", pin_order=reorderable_pins)

    edges = [
        e for e in extract_edges(data["nets"], "J1", "J2")
        if e.fixed_pin != "SH"
    ]

    report = analyze_connectors(fixed, reorderable, edges)
    # The current netlist has the fixed pin order (0 crossings)
    assert report.crossing_count == 0


def test_parse_and_analyze_i2c():
    """End-to-end: parse I2C netlist and verify 0 crossings."""
    netlist_path = Path(__file__).resolve().parent.parent / "examples" / "i2c_breakout.net"
    if not netlist_path.exists():
        return

    data = parse_netlist(str(netlist_path))

    fixed_pins = infer_pin_order("J1", data["nets"])
    reorderable_pins = infer_pin_order("J2", data["nets"])

    fixed = PinColumn(ref="J1", pin_order=fixed_pins)
    reorderable = PinColumn(ref="J2", pin_order=reorderable_pins)

    edges = extract_edges(data["nets"], "J1", "J2")

    report = analyze_connectors(fixed, reorderable, edges)
    assert report.crossing_count == 0


# =========================================================================
# Run
# =========================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
