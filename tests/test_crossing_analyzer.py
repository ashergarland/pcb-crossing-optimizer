"""Tests for the PCB Crossing Optimizer (sweep-only API)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crossing_analyzer import (
    PinColumn,
    LayerEdge,
    LayerPairCrossing,
    MultilayerReport,
    layer_global_positions,
    count_layer_pair_crossings,
    extract_layer_pair_edges,
    sweep_optimize,
    format_multilayer_report,
    report_to_dict,
    _format_pin_ref,
    parse_netlist,
    infer_pin_order,
    PinAssignment,
    FootprintPlan,
    parse_pin_locks,
    build_pin_map,
    plan_footprint,
    format_footprint_plan,
    plan_to_dict,
)


# =========================================================================
# Unit tests: position mapping
# =========================================================================

def test_layer_global_positions_single():
    """Single component: positions are 0..n-1."""
    comps = [PinColumn(ref="J1", pin_order=["1", "2", "3"])]
    pos = layer_global_positions(comps)
    assert pos == {("J1", "1"): 0, ("J1", "2"): 1, ("J1", "3"): 2}


def test_layer_global_positions_multi():
    """Multiple components: positions are sequential across components."""
    comps = [
        PinColumn(ref="R1", pin_order=["1", "2"]),
        PinColumn(ref="R2", pin_order=["1", "2"]),
    ]
    pos = layer_global_positions(comps)
    assert pos == {
        ("R1", "1"): 0, ("R1", "2"): 1,
        ("R2", "1"): 2, ("R2", "2"): 3,
    }


# =========================================================================
# Unit tests: crossing counter
# =========================================================================

def test_layer_pair_no_crossings():
    """Parallel edges between single-component layers produce no crossings."""
    src = [PinColumn(ref="J1", pin_order=["1", "2", "3"])]
    tgt = [PinColumn(ref="J2", pin_order=["1", "2", "3"])]
    edges = [
        LayerEdge("A", "J1", "1", "J2", "1"),
        LayerEdge("B", "J1", "2", "J2", "2"),
        LayerEdge("C", "J1", "3", "J2", "3"),
    ]
    crossings = count_layer_pair_crossings(src, tgt, edges)
    assert len(crossings) == 0


def test_layer_pair_with_crossings():
    """Two crossing edges between single-component layers."""
    src = [PinColumn(ref="J1", pin_order=["1", "2"])]
    tgt = [PinColumn(ref="J2", pin_order=["1", "2"])]
    edges = [
        LayerEdge("A", "J1", "1", "J2", "2"),
        LayerEdge("B", "J1", "2", "J2", "1"),
    ]
    crossings = count_layer_pair_crossings(src, tgt, edges)
    assert len(crossings) == 1


def test_layer_pair_multi_component():
    """Crossings between layers with multiple components."""
    src = [PinColumn(ref="J1", pin_order=["1", "2", "3"])]
    tgt = [
        PinColumn(ref="R1", pin_order=["1", "2"]),
        PinColumn(ref="R2", pin_order=["1", "2"]),
    ]
    # J1.1 -> R2.1 (pos 0 -> pos 2), J1.3 -> R1.1 (pos 2 -> pos 0): crossing
    edges = [
        LayerEdge("A", "J1", "1", "R2", "1"),
        LayerEdge("B", "J1", "3", "R1", "1"),
    ]
    crossings = count_layer_pair_crossings(src, tgt, edges)
    assert len(crossings) == 1


# =========================================================================
# Unit tests: edge extraction between layer pairs
# =========================================================================

def test_extract_layer_pair_edges():
    """Extract edges from parsed net data between two component sets."""
    nets = {
        "VDD": [("J1", "1"), ("R1", "1"), ("J2", "1")],
        "GND": [("J1", "2"), ("J2", "2")],
        "SIG": [("R1", "2"), ("J2", "3")],
    }
    edges = extract_layer_pair_edges(nets, {"J1"}, {"R1"})
    # Only VDD connects J1 and R1
    assert len(edges) == 1
    assert edges[0].net_name == "VDD"
    assert edges[0].source_ref == "J1"
    assert edges[0].target_ref == "R1"


def test_extract_layer_pair_edges_multi_target():
    """Net connecting one source to multiple targets."""
    nets = {
        "VDD": [("J1", "1"), ("R1", "1"), ("R2", "1")],
    }
    edges = extract_layer_pair_edges(nets, {"J1"}, {"R1", "R2"})
    assert len(edges) == 2
    targets = {(e.target_ref, e.target_pin) for e in edges}
    assert ("R1", "1") in targets
    assert ("R2", "1") in targets


# =========================================================================
# Unit tests: sweep optimizer
# =========================================================================

def test_sweep_two_layers_no_crossings():
    """Two layers, parallel edges: no optimization needed."""
    layers = [
        [PinColumn(ref="J1", pin_order=["1", "2", "3"])],
        [PinColumn(ref="J2", pin_order=["1", "2", "3"])],
    ]
    nets = {
        "A": [("J1", "1"), ("J2", "1")],
        "B": [("J1", "2"), ("J2", "2")],
        "C": [("J1", "3"), ("J2", "3")],
    }
    report = sweep_optimize(layers, {"J2"}, nets)
    assert report.total_crossings == 0
    assert report.total_crossings_after == 0


def test_sweep_two_layers_with_crossings():
    """Two layers with crossings: sweep should fix them."""
    layers = [
        [PinColumn(ref="J1", pin_order=["1", "2"])],
        [PinColumn(ref="J2", pin_order=["1", "2"])],
    ]
    nets = {
        "A": [("J1", "1"), ("J2", "2")],
        "B": [("J1", "2"), ("J2", "1")],
    }
    report = sweep_optimize(layers, {"J2"}, nets)
    assert report.total_crossings == 1
    assert report.total_crossings_after == 0
    assert report.optimized_orders["J2"] == ["2", "1"]


def test_sweep_two_layers_matches_pair_scenario():
    """Two-layer sweep correctly resolves the microSD 2-connector scenario."""
    layers = [
        [PinColumn(ref="J1", pin_order=["2", "3", "4", "5", "6", "7"])],
        [PinColumn(ref="J2", pin_order=["1", "2", "3", "4", "5", "6"])],
    ]
    nets = {
        "CS":   [("J1", "2"), ("J2", "1")],
        "MOSI": [("J1", "3"), ("J2", "2")],
        "VDD":  [("J1", "4"), ("J2", "3")],
        "SCK":  [("J1", "5"), ("J2", "6")],
        "GND":  [("J1", "6"), ("J2", "4")],
        "MISO": [("J1", "7"), ("J2", "5")],
    }
    report = sweep_optimize(layers, {"J2"}, nets)
    assert report.total_crossings == 2
    assert report.total_crossings_after == 0
    # Optimal order should put SCK before GND before MISO
    opt = report.optimized_orders["J2"]
    assert opt.index("6") < opt.index("4")  # SCK pin before GND pin
    assert opt.index("4") < opt.index("5")  # GND pin before MISO pin


def test_sweep_three_layers_passives():
    """Three layers: J1 -> [R1, R2] -> J2, with crossings through passives."""
    layers = [
        [PinColumn(ref="J1", pin_order=["1", "2", "3"])],
        [
            PinColumn(ref="R1", pin_order=["1", "2"]),
            PinColumn(ref="R2", pin_order=["1", "2"]),
        ],
        [PinColumn(ref="J2", pin_order=["1", "2", "3"])],
    ]
    nets = {
        "SIG_A": [("J1", "1"), ("R1", "1")],
        "SIG_A_OUT": [("R1", "2"), ("J2", "3")],
        "SIG_B": [("J1", "3"), ("R2", "1")],
        "SIG_B_OUT": [("R2", "2"), ("J2", "1")],
        "PASS": [("J1", "2"), ("J2", "2")],
    }
    report = sweep_optimize(layers, {"J2"}, nets)
    assert report.total_crossings >= 1
    assert report.total_crossings_after == 0


def test_sweep_fixed_components_unchanged():
    """Components not in reorderable_refs should keep their pin order."""
    layers = [
        [PinColumn(ref="J1", pin_order=["1", "2"])],
        [PinColumn(ref="J2", pin_order=["1", "2"])],
    ]
    nets = {
        "A": [("J1", "1"), ("J2", "2")],
        "B": [("J1", "2"), ("J2", "1")],
    }
    report = sweep_optimize(layers, set(), nets)  # nothing reorderable
    assert report.total_crossings == 1
    assert report.total_crossings_after == 1  # can't fix without reordering
    assert report.optimized_orders["J1"] == ["1", "2"]
    assert report.optimized_orders["J2"] == ["1", "2"]


def test_sweep_no_mutation():
    """sweep_optimize should not mutate the caller's layer data."""
    original = [
        [PinColumn(ref="J1", pin_order=["1", "2"])],
        [PinColumn(ref="J2", pin_order=["1", "2"])],
    ]
    nets = {
        "A": [("J1", "1"), ("J2", "2")],
        "B": [("J1", "2"), ("J2", "1")],
    }
    sweep_optimize(original, {"J2"}, nets)
    assert original[1][0].pin_order == ["1", "2"]


def test_sweep_component_reordering():
    """Sweep should reorder components within a layer to reduce crossings."""
    layers = [
        [PinColumn(ref="J1", pin_order=["1", "2", "3", "4"])],
        [
            PinColumn(ref="R2", pin_order=["1", "2"]),  # Initially first
            PinColumn(ref="R1", pin_order=["1", "2"]),  # Initially second
        ],
        [PinColumn(ref="J2", pin_order=["1", "2"])],
    ]
    nets = {
        "SIG_1": [("J1", "1"), ("R1", "1")],
        "SIG_1_OUT": [("R1", "2"), ("J2", "1")],
        "SIG_2": [("J1", "4"), ("R2", "1")],
        "SIG_2_OUT": [("R2", "2"), ("J2", "2")],
    }
    report = sweep_optimize(layers, {"J2"}, nets)
    assert report.total_crossings_after == 0


def test_sweep_long_edge_via_virtual_nodes():
    """Long edges spanning non-adjacent layers are handled via virtual nodes.

    Without virtual nodes: the A->C edge would be invisible to the layer 0->1
    crossing counter, missing a real crossing.
    With virtual nodes: a pass-through pin is inserted in layer 1, and the
    crossing is detected.
    """
    layers = [
        [PinColumn(ref="J1", pin_order=["1", "2"])],
        [PinColumn(ref="R1", pin_order=["1", "2"])],
        [PinColumn(ref="J2", pin_order=["1", "2"])],
    ]
    nets = {
        "A": [("J1", "1"), ("J2", "2")],
        "B_in": [("J1", "2"), ("R1", "1")],
        "B_out": [("R1", "2"), ("J2", "1")],
    }
    report = sweep_optimize(layers, {"J2"}, nets)
    assert report.total_crossings >= 1
    assert report.total_crossings_after <= report.total_crossings


# =========================================================================
# Unit tests: report formatting
# =========================================================================

def test_format_multilayer_report_no_crossings():
    """Format report when there are no crossings."""
    report = MultilayerReport(
        total_crossings=0,
        total_crossings_after=0,
        layer_pair_reports=[],
        original_orders={"J1": ["1", "2"], "J2": ["1", "2"]},
        optimized_orders={"J1": ["1", "2"], "J2": ["1", "2"]},
        iterations=1,
    )
    text = format_multilayer_report(report, {})
    assert "Total crossings (before): 0" in text
    assert "No reordering needed." in text


def test_format_multilayer_report_with_changes():
    """Format report when reordering was applied."""
    report = MultilayerReport(
        total_crossings=1,
        total_crossings_after=0,
        layer_pair_reports=[],
        original_orders={"J2": ["1", "2"]},
        optimized_orders={"J2": ["2", "1"]},
        iterations=1,
    )
    text = format_multilayer_report(report, {})
    assert "Recommended pin reorderings:" in text
    assert "J2:" in text


# =========================================================================
# Integration: netlist parsing
# =========================================================================

def test_parse_microsd_netlist():
    """Parse the microSD example netlist and verify structure."""
    netlist_path = Path(__file__).resolve().parent.parent / "examples" / "microsd_breakout.net"
    if not netlist_path.exists():
        return  # skip if example not available

    data = parse_netlist(str(netlist_path))

    assert "J1" in data["components"]
    assert "J2" in data["components"]
    assert "R1" in data["components"]
    assert "C1" in data["components"]

    assert "CS" in data["nets"]
    assert "GND" in data["nets"]
    assert "VDD" in data["nets"]
    assert "MOSI" in data["nets"]
    assert "MISO" in data["nets"]
    assert "SCK" in data["nets"]


# =========================================================================
# Integration: sweep with real netlists
# =========================================================================

def test_sweep_microsd_netlist():
    """Sweep on microSD netlist with 3 layers: J1 -> [R1,R2,C1] -> J2.

    The 3-layer model adds virtual pass-through nodes for signals that
    span layers 0 and 2, revealing crossings between signal highways
    and passive component connections.
    The sweep minimizes total crossings but cannot eliminate all of them
    because passive pin orientations create unavoidable inversions.
    """
    netlist_path = Path(__file__).resolve().parent.parent / "examples" / "microsd_breakout.net"
    if not netlist_path.exists():
        return

    data = parse_netlist(str(netlist_path))
    exclude = {("J1", "SH")}

    layers = []
    for refs in [["J1"], ["R1", "R2", "C1"], ["J2"]]:
        layer = []
        for ref in refs:
            pins = [
                p for p in infer_pin_order(ref, data["nets"])
                if (ref, p) not in exclude
            ]
            layer.append(PinColumn(ref=ref, pin_order=pins))
        layers.append(layer)

    report = sweep_optimize(layers, {"J2"}, data["nets"])

    # 3-layer model detects more crossings than 2-layer pair analysis
    assert report.total_crossings > 0
    # Sweep should reduce total crossings
    assert report.total_crossings_after <= report.total_crossings
    # J2 should still appear in optimized orders
    assert "J2" in report.optimized_orders
    # Virtual refs should NOT appear in optimized orders
    assert all(not ref.startswith("_virt_") for ref in report.optimized_orders)


def test_sweep_i2c_netlist():
    """Sweep on I2C netlist: should handle 2-layer case correctly."""
    netlist_path = Path(__file__).resolve().parent.parent / "examples" / "i2c_breakout.net"
    if not netlist_path.exists():
        return

    data = parse_netlist(str(netlist_path))

    layers = [
        [PinColumn(ref="J1", pin_order=infer_pin_order("J1", data["nets"]))],
        [PinColumn(ref="J2", pin_order=infer_pin_order("J2", data["nets"]))],
    ]

    report = sweep_optimize(layers, {"J2"}, data["nets"])
    assert report.total_crossings_after <= report.total_crossings
    assert "J2" in report.optimized_orders


# =========================================================================
# Virtual node display cleanup
# =========================================================================

def test_format_pin_ref_real():
    """Real refs display as ref.pin."""
    assert _format_pin_ref("J1", "6") == "J1.6"
    assert _format_pin_ref("R2", "1") == "R2.1"


def test_format_pin_ref_virtual():
    """Virtual refs display as [pass-through]."""
    assert _format_pin_ref("_virt_L1", "_v3") == "[pass-through]"
    assert _format_pin_ref("_virt_L0", "_v0") == "[pass-through]"


def test_format_virtual_nodes_cleaned():
    """format_multilayer_report hides virtual node internals in crossing descriptions."""
    # Build a report with a crossing that includes a virtual node edge
    edge_a = LayerEdge("net_A", "J1", "1", "_virt_L1", "_v0")
    edge_b = LayerEdge("net_B", "J1", "2", "_virt_L1", "_v1")
    from crossing_analyzer import LayerPairReport
    pr = LayerPairReport(
        source_layer_idx=0, target_layer_idx=1,
        source_refs=["J1"], target_refs=["_virt_L1"],
        crossing_count=1,
        crossings=[LayerPairCrossing(edge_a, edge_b)],
    )
    report = MultilayerReport(
        total_crossings=1, total_crossings_after=1,
        layer_pair_reports=[pr],
        original_orders={}, optimized_orders={},
        iterations=1,
    )
    text = format_multilayer_report(report, {})
    assert "_virt_" not in text
    assert "[pass-through]" in text


# =========================================================================
# report_to_dict
# =========================================================================

def test_report_to_dict_structure():
    """report_to_dict returns expected top-level keys."""
    report = MultilayerReport(
        total_crossings=5, total_crossings_after=2,
        layer_pair_reports=[],
        original_orders={"J1": ["1", "2"]},
        optimized_orders={"J1": ["2", "1"]},
        iterations=3,
    )
    d = report_to_dict(report, {})
    assert d["total_crossings_before"] == 5
    assert d["total_crossings_after"] == 2
    assert d["iterations"] == 3
    assert "J1" in d["reorderings"]
    assert d["reorderings"]["J1"]["original"] == ["1", "2"]
    assert d["reorderings"]["J1"]["optimized"] == ["2", "1"]


def test_report_to_dict_no_virtual_refs():
    """report_to_dict uses [pass-through] for virtual node refs."""
    edge_a = LayerEdge("net_X", "J1", "1", "_virt_L1", "_v0")
    edge_b = LayerEdge("net_Y", "J1", "2", "_virt_L1", "_v1")
    from crossing_analyzer import LayerPairReport
    pr = LayerPairReport(
        source_layer_idx=0, target_layer_idx=1,
        source_refs=["J1"], target_refs=["_virt_L1"],
        crossing_count=1,
        crossings=[LayerPairCrossing(edge_a, edge_b)],
    )
    report = MultilayerReport(
        total_crossings=1, total_crossings_after=1,
        layer_pair_reports=[pr],
        original_orders={}, optimized_orders={},
        iterations=1,
    )
    d = report_to_dict(report, {})
    import json
    text = json.dumps(d)
    assert "_virt_" not in text
    assert "[pass-through]" in text


# =========================================================================
# CLI tests (subprocess)
# =========================================================================

import subprocess

def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run pcb-crossing-optimizer CLI via python -m style."""
    src = str(Path(__file__).resolve().parent.parent / "src" / "crossing_analyzer.py")
    return subprocess.run(
        [sys.executable, src, *args],
        capture_output=True, text=True,
    )


def test_cli_json_output():
    """--json flag produces valid JSON output."""
    netlist = str(Path(__file__).resolve().parent.parent / "examples" / "i2c_breakout.net")
    if not Path(netlist).exists():
        return
    result = _run_cli(netlist, "--layers", "J1 | J2", "--reorderable", "J2", "--json")
    import json
    data = json.loads(result.stdout)
    assert "total_crossings_before" in data
    assert "total_crossings_after" in data
    assert "reorderings" in data
    assert "_virt_" not in result.stdout


def test_cli_quiet_exit_code_zero():
    """--quiet suppresses output; exit 0 when no crossings remain."""
    netlist = str(Path(__file__).resolve().parent.parent / "examples" / "i2c_breakout.net")
    if not Path(netlist).exists():
        return
    result = _run_cli(netlist, "--layers", "J1 | J2", "--reorderable", "J2", "--quiet")
    # i2c 2-layer case should be fully resolvable -> exit 0
    assert result.stdout.strip() == ""
    assert result.returncode == 0


def test_cli_quiet_exit_code_one():
    """--quiet with unsolvable crossings returns exit code 1."""
    netlist = str(Path(__file__).resolve().parent.parent / "examples" / "microsd_breakout.net")
    if not Path(netlist).exists():
        return
    # microSD 3-layer case has unavoidable crossings
    result = _run_cli(netlist, "--layers", "J1 | R1,R2,C1 | J2", "--reorderable", "J2", "--exclude", "J1:SH", "--quiet")
    assert result.stdout.strip() == ""
    assert result.returncode == 1


# =========================================================================
# parse_pin_locks tests
# =========================================================================

def test_parse_pin_locks_basic():
    """Parse simple PIN=NET locks."""
    locks = parse_pin_locks(["1=NC", "3=GND_EARLY_A", "4=GND_EARLY_B"])
    assert locks == {"1": None, "3": "GND_EARLY_A", "4": "GND_EARLY_B"}


def test_parse_pin_locks_empty():
    """Empty list returns empty dict."""
    assert parse_pin_locks([]) == {}


def test_parse_pin_locks_nc_case_insensitive():
    """NC detection is case-insensitive."""
    locks = parse_pin_locks(["1=nc", "2=Nc"])
    assert locks["1"] is None
    assert locks["2"] is None


def test_parse_pin_locks_invalid():
    """Invalid format raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Invalid --lock format"):
        parse_pin_locks(["bad"])


def test_parse_pin_locks_empty_pin():
    """Empty pin raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Empty pin"):
        parse_pin_locks(["=NET"])


# =========================================================================
# build_pin_map tests
# =========================================================================

def test_build_pin_map_basic():
    """build_pin_map merges locked, optimized, and unmatched pins."""
    all_pins = ["1", "2", "3", "4", "5"]
    locks = {"1": None, "5": None}  # NC
    optimized_signal = ["3", "4"]   # two signal pins in sweep order
    pin_to_net = {"3": "NET_A", "4": "NET_B"}
    nets = {"NET_A": [("J1", "3"), ("J2", "1")], "NET_B": [("J1", "4"), ("J2", "2")]}

    result = build_pin_map(
        target_ref="J1",
        all_pins=all_pins,
        locks=locks,
        optimized_signal_pins=optimized_signal,
        pin_to_net=pin_to_net,
        nets=nets,
        unmatched_mode="end",
    )

    assert len(result) == 5
    assert result[0].status == "locked"
    assert result[0].net is None  # NC
    assert result[4].status == "locked"
    # Positions 1-3 (indices 1-3): 2 optimized + 1 unmatched
    statuses = [a.status for a in result]
    assert statuses.count("locked") == 2
    assert statuses.count("optimized") == 2
    assert statuses.count("unmatched") == 1


def test_build_pin_map_unmatched_start():
    """unmatched_mode='start' places unmatched pins before signal pins."""
    all_pins = ["1", "2", "3", "4"]
    locks = {}
    optimized_signal = ["2"]
    pin_to_net = {"2": "NET_A"}
    nets = {"NET_A": [("J1", "2"), ("J2", "1")]}

    result = build_pin_map(
        target_ref="J1",
        all_pins=all_pins,
        locks=locks,
        optimized_signal_pins=optimized_signal,
        pin_to_net=pin_to_net,
        nets=nets,
        unmatched_mode="start",
    )

    # With 4 pins, 0 locked, 1 optimized, 3 unmatched
    # start mode: unmatched first, then optimized
    statuses = [a.status for a in result]
    assert statuses[:3] == ["unmatched", "unmatched", "unmatched"]
    assert statuses[3] == "optimized"


# =========================================================================
# plan_footprint unit tests
# =========================================================================

def test_plan_footprint_simple():
    """plan_footprint with a simple 2-layer topology returns a FootprintPlan."""
    # Minimal nets: J2 pin 1 -> NET_A, J2 pin 2 -> NET_B
    # Target J1 with 4 pins: want to find optimal pin assignment
    nets = {
        "NET_A": [("J2", "1"), ("J1", "1")],
        "NET_B": [("J2", "2"), ("J1", "2")],
    }
    anchor_layers = [[PinColumn(ref="J2", pin_order=["1", "2"])]]
    locks = {"3": None, "4": None}

    plan = plan_footprint(
        target_ref="J1",
        target_pins=["1", "2", "3", "4"],
        anchor_layers=anchor_layers,
        nets=nets,
        locks=locks,
        unmatched="end",
    )

    assert isinstance(plan, FootprintPlan)
    assert plan.target_ref == "J1"
    assert len(plan.pin_map) == 4
    assert plan.crossings_after == 0  # trivial case, no crossings


def test_plan_footprint_excludes_nets():
    """plan_footprint excludes specified nets from analysis."""
    nets = {
        "GND": [("J2", "1"), ("J1", "1"), ("C1", "2")],
        "NET_A": [("J2", "2"), ("J1", "2")],
    }
    anchor_layers = [[PinColumn(ref="J2", pin_order=["1", "2"])]]

    plan = plan_footprint(
        target_ref="J1",
        target_pins=["1", "2"],
        anchor_layers=anchor_layers,
        nets=nets,
        locks={},
        exclude_nets={"GND"},
    )

    # GND should not appear in the signal optimization
    signal_nets = [a.net for a in plan.pin_map if a.status == "optimized"]
    assert "GND" not in signal_nets


# =========================================================================
# format_footprint_plan / plan_to_dict tests
# =========================================================================

def test_format_footprint_plan_output():
    """format_footprint_plan produces human-readable output."""
    plan = FootprintPlan(
        target_ref="J1",
        pin_map=[
            PinAssignment(pin="1", net=None, status="locked"),
            PinAssignment(pin="2", net="NET_A", status="optimized", routes_to="J2.1"),
            PinAssignment(pin="3", net=None, status="unmatched"),
        ],
        crossings_before=2,
        crossings_after=0,
        iterations=3,
        passive_reorderings={},
    )
    text = format_footprint_plan(plan)
    assert "J1" in text
    assert "Locked pins" in text
    assert "Optimized signal assignment" in text
    assert "Unmatched pins" in text
    assert "2 before -> 0 after" in text


def test_plan_to_dict_structure():
    """plan_to_dict produces expected JSON structure."""
    plan = FootprintPlan(
        target_ref="J1",
        pin_map=[
            PinAssignment(pin="1", net=None, status="locked"),
            PinAssignment(pin="2", net="NET_A", status="optimized", routes_to="J2.1"),
        ],
        crossings_before=1,
        crossings_after=0,
        iterations=2,
        passive_reorderings={"R1": ["2", "1"]},
    )
    d = plan_to_dict(plan)
    assert d["target"] == "J1"
    assert d["total_pins"] == 2
    assert d["crossings_before"] == 1
    assert d["crossings_after"] == 0
    assert len(d["pin_map"]) == 2
    assert d["pin_map"][0]["status"] == "locked"
    assert d["pin_map"][1]["routes_to"] == "J2.1"
    assert d["passive_reorderings"]["R1"] == ["2", "1"]


# =========================================================================
# plan-footprint CLI tests
# =========================================================================

def test_cli_plan_footprint_json():
    """plan-footprint --json produces valid JSON."""
    netlist = str(Path(__file__).resolve().parent.parent / "examples" / "microsd_breakout.net")
    if not Path(netlist).exists():
        return
    result = _run_cli(
        "plan-footprint", netlist,
        "--target", "J1",
        "--anchors", "J2",
        "--lock", "1=NC",
        "--exclude-nets", "GND",
        "--json",
    )
    assert result.returncode in (0, 1)  # may or may not have crossings
    import json
    data = json.loads(result.stdout)
    assert "target" in data
    assert "pin_map" in data
    assert data["target"] == "J1"


def test_cli_plan_footprint_text():
    """plan-footprint without --json produces readable text."""
    netlist = str(Path(__file__).resolve().parent.parent / "examples" / "microsd_breakout.net")
    if not Path(netlist).exists():
        return
    result = _run_cli(
        "plan-footprint", netlist,
        "--target", "J1",
        "--anchors", "J2",
        "--lock", "1=NC",
        "--exclude-nets", "GND",
    )
    assert result.returncode in (0, 1)
    assert "Footprint Pin Map Proposal" in result.stdout


def test_cli_plan_footprint_quiet():
    """plan-footprint --quiet suppresses output."""
    netlist = str(Path(__file__).resolve().parent.parent / "examples" / "microsd_breakout.net")
    if not Path(netlist).exists():
        return
    result = _run_cli(
        "plan-footprint", netlist,
        "--target", "J1",
        "--anchors", "J2",
        "--exclude-nets", "GND",
        "--quiet",
    )
    assert result.stdout.strip() == ""
    assert result.returncode in (0, 1)


def test_cli_analyze_explicit_subcommand():
    """Explicit 'analyze' subcommand works."""
    netlist = str(Path(__file__).resolve().parent.parent / "examples" / "i2c_breakout.net")
    if not Path(netlist).exists():
        return
    result = _run_cli(
        "analyze", netlist,
        "--layers", "J1 | J2",
        "--reorderable", "J2",
        "--json",
    )
    import json
    data = json.loads(result.stdout)
    assert "total_crossings_before" in data
