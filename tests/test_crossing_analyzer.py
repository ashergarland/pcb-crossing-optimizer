"""Tests for the crossing analyzer (sweep-only API)."""

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
    """Run crossing-analyzer CLI via python -m style."""
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
