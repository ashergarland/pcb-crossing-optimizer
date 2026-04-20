"""Crossing analyzer for SKiDL-generated netlists.

Detects trace crossings between connector pairs and computes optimal
pin orderings for reorderable connectors (headers) to minimize or
eliminate crossings for single-layer routing.

Algorithm: bipartite graph crossing minimization via inversion counting
and barycenter heuristic for multi-fan nets.

Usage (CLI):
    python tools/crossing_analyzer.py <netlist.net> <fixed_ref> <reorderable_ref>

    Example:
    python tools/crossing_analyzer.py output/netlists/microsd_breakout.net J1 J2

Programmatic usage from SKiDL scripts:
    from tools.crossing_analyzer import analyze_connectors, PinColumn, Edge
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional


# =========================================================================
# Data model
# =========================================================================

@dataclass
class PinColumn:
    """A connector with pins in physical order (top-to-bottom or left-to-right).

    pin_order is a list of pin IDs (strings) in physical position order.
    Position 0 is the first physical pin.
    """
    ref: str
    pin_order: list[str]

    def position_of(self, pin_id: str) -> int:
        return self.pin_order.index(pin_id)


@dataclass
class Edge:
    """A net connecting one pin on the fixed column to one pin on the
    reorderable column."""
    net_name: str
    fixed_pin: str
    reorderable_pin: str


@dataclass
class CrossingPair:
    """Two edges that cross each other."""
    edge_a: Edge
    edge_b: Edge


@dataclass
class CrossingReport:
    """Result of a crossing analysis between two connectors."""
    fixed_ref: str
    reorderable_ref: str
    crossing_count: int
    crossings: list[CrossingPair]
    current_order: list[str]
    optimal_order: list[str]
    optimal_crossing_count: int


# =========================================================================
# Core analysis (pure data in / data out)
# =========================================================================

def count_crossings(
    fixed: PinColumn,
    reorderable: PinColumn,
    edges: list[Edge],
) -> list[CrossingPair]:
    """Count crossings between edges connecting two pin columns.

    Two edges (i->j) and (k->l) cross iff (i < k and j > l) or
    (i > k and j < l), where i,k are positions on the fixed column
    and j,l are positions on the reorderable column.
    """
    crossings: list[CrossingPair] = []
    for a, b in combinations(edges, 2):
        fi = fixed.position_of(a.fixed_pin)
        fk = fixed.position_of(b.fixed_pin)
        rj = reorderable.position_of(a.reorderable_pin)
        rl = reorderable.position_of(b.reorderable_pin)
        if (fi < fk and rj > rl) or (fi > fk and rj < rl):
            crossings.append(CrossingPair(a, b))
    return crossings


def compute_optimal_order(
    fixed: PinColumn,
    edges: list[Edge],
) -> list[str]:
    """Compute the optimal pin order for the reorderable column.

    For 1:1 pin mappings: sort reorderable pins by the position of their
    connected fixed pin. This guarantees zero crossings.

    For multi-fan nets (one reorderable pin connected to multiple fixed
    pins, or vice versa): use the barycenter heuristic, assigning each
    reorderable pin the average position of all its connected fixed pins,
    then sorting by that value.
    """
    # Build map: reorderable_pin -> list of fixed positions
    pin_positions: dict[str, list[int]] = {}
    for edge in edges:
        pos = fixed.position_of(edge.fixed_pin)
        pin_positions.setdefault(edge.reorderable_pin, []).append(pos)

    # Barycenter: average connected fixed position
    barycenters: dict[str, float] = {}
    for pin, positions in pin_positions.items():
        barycenters[pin] = sum(positions) / len(positions)

    return sorted(barycenters.keys(), key=lambda p: barycenters[p])


def analyze_connectors(
    fixed: PinColumn,
    reorderable: PinColumn,
    edges: list[Edge],
) -> CrossingReport:
    """Full crossing analysis: count current crossings, compute optimal order,
    count crossings after optimization."""
    current_crossings = count_crossings(fixed, reorderable, edges)

    optimal_order = compute_optimal_order(fixed, edges)

    # Build an optimized PinColumn and count remaining crossings
    optimized = PinColumn(ref=reorderable.ref, pin_order=optimal_order)
    optimal_crossings = count_crossings(fixed, optimized, edges)

    return CrossingReport(
        fixed_ref=fixed.ref,
        reorderable_ref=reorderable.ref,
        crossing_count=len(current_crossings),
        crossings=current_crossings,
        current_order=list(reorderable.pin_order),
        optimal_order=optimal_order,
        optimal_crossing_count=len(optimal_crossings),
    )


# =========================================================================
# KiCad netlist parser (.net S-expression format)
# =========================================================================

def _tokenize_sexp(text: str) -> list[str]:
    """Tokenize an S-expression into a flat list of tokens."""
    tokens: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in " \t\n\r":
            i += 1
        elif c == "(":
            tokens.append("(")
            i += 1
        elif c == ")":
            tokens.append(")")
            i += 1
        elif c == '"':
            # Quoted string
            j = i + 1
            while j < len(text) and text[j] != '"':
                if text[j] == "\\":
                    j += 1  # skip escaped char
                j += 1
            tokens.append(text[i + 1 : j])
            i = j + 1
        else:
            # Unquoted atom
            j = i
            while j < len(text) and text[j] not in " \t\n\r()\"":
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _parse_sexp(tokens: list[str], pos: int = 0) -> tuple:
    """Parse tokenized S-expression into nested tuples."""
    if tokens[pos] == "(":
        items = []
        pos += 1
        while tokens[pos] != ")":
            item, pos = _parse_sexp(tokens, pos)
            items.append(item)
        return tuple(items), pos + 1
    else:
        return tokens[pos], pos + 1


def _find_nodes(sexp, tag: str):
    """Recursively find all sub-expressions starting with the given tag."""
    results = []
    if isinstance(sexp, tuple):
        if len(sexp) > 0 and sexp[0] == tag:
            results.append(sexp)
        for child in sexp:
            results.extend(_find_nodes(child, tag))
    return results


def parse_netlist(filepath: str) -> dict:
    """Parse a KiCad .net file, returning components and net connectivity.

    Returns:
        {
            "components": {ref: {"value": str, "part": str}},
            "nets": {net_name: [(ref, pin_id), ...]},
        }
    """
    text = Path(filepath).read_text(encoding="utf-8")
    tokens = _tokenize_sexp(text)
    tree, _ = _parse_sexp(tokens)

    components: dict[str, dict] = {}
    for comp in _find_nodes(tree, "comp"):
        ref = value = part = None
        for item in comp:
            if isinstance(item, tuple):
                if item[0] == "ref" and len(item) > 1:
                    ref = item[1]
                elif item[0] == "value" and len(item) > 1:
                    value = item[1]
                elif item[0] == "libsource":
                    for sub in item:
                        if isinstance(sub, tuple) and sub[0] == "part" and len(sub) > 1:
                            part = sub[1]
        if ref:
            components[ref] = {"value": value, "part": part}

    nets: dict[str, list[tuple[str, str]]] = {}
    for net in _find_nodes(tree, "net"):
        net_name = None
        nodes = []
        for item in net:
            if isinstance(item, tuple):
                if item[0] == "name" and len(item) > 1:
                    net_name = item[1]
                elif item[0] == "node":
                    ref = pin = None
                    for sub in item:
                        if isinstance(sub, tuple):
                            if sub[0] == "ref" and len(sub) > 1:
                                ref = sub[1]
                            elif sub[0] == "pin" and len(sub) > 1:
                                pin = sub[1]
                    if ref and pin:
                        nodes.append((ref, pin))
        if net_name:
            nets[net_name] = nodes

    return {"components": components, "nets": nets}


def extract_edges(
    nets: dict[str, list[tuple[str, str]]],
    fixed_ref: str,
    reorderable_ref: str,
) -> list[Edge]:
    """Extract edges between two connectors from parsed net data.

    Only nets that connect both connectors produce edges.
    """
    edges: list[Edge] = []
    for net_name, nodes in nets.items():
        fixed_pins = [pin for ref, pin in nodes if ref == fixed_ref]
        reorder_pins = [pin for ref, pin in nodes if ref == reorderable_ref]
        # Create an edge for each pair (handles multi-pin nets)
        for fp in fixed_pins:
            for rp in reorder_pins:
                edges.append(Edge(net_name=net_name, fixed_pin=fp, reorderable_pin=rp))
    return edges


def infer_pin_order(ref: str, nets: dict[str, list[tuple[str, str]]]) -> list[str]:
    """Infer pin order for a component by collecting all pin IDs seen in the
    netlist and sorting them numerically (with non-numeric pins like 'SH' last)."""
    pins: set[str] = set()
    for nodes in nets.values():
        for node_ref, pin in nodes:
            if node_ref == ref:
                pins.add(pin)

    def sort_key(p: str):
        try:
            return (0, int(p))
        except ValueError:
            return (1, p)

    return sorted(pins, key=sort_key)


# =========================================================================
# Report formatting
# =========================================================================

def format_report(report: CrossingReport) -> str:
    """Format a CrossingReport as human-readable text."""
    lines: list[str] = []
    lines.append(f"Crossing Analysis: {report.fixed_ref} -> {report.reorderable_ref}")
    lines.append("=" * 60)
    lines.append("")

    if report.crossing_count == 0:
        lines.append("No crossings detected. Pin ordering is optimal.")
        lines.append("")
        lines.append(f"Current {report.reorderable_ref} pin order: "
                      f"{', '.join(report.current_order)}")
        return "\n".join(lines)

    lines.append(f"Crossings found: {report.crossing_count}")
    lines.append("")

    for i, cp in enumerate(report.crossings, 1):
        lines.append(
            f"  {i}. {cp.edge_a.net_name} ({report.fixed_ref}.{cp.edge_a.fixed_pin}"
            f" -> {report.reorderable_ref}.{cp.edge_a.reorderable_pin})"
            f"  X  {cp.edge_b.net_name} ({report.fixed_ref}.{cp.edge_b.fixed_pin}"
            f" -> {report.reorderable_ref}.{cp.edge_b.reorderable_pin})"
        )

    lines.append("")
    lines.append(f"Current {report.reorderable_ref} pin order:  "
                  f"{', '.join(report.current_order)}")
    lines.append(f"Optimal {report.reorderable_ref} pin order:  "
                  f"{', '.join(report.optimal_order)}")
    lines.append(f"Crossings after optimization: {report.optimal_crossing_count}")

    if report.optimal_crossing_count > 0:
        lines.append("")
        lines.append(
            "WARNING: Not all crossings can be eliminated by reordering alone."
        )
        lines.append(
            "Remaining crossings will require vias or a second routing layer."
        )

    # Generate the net-to-pin mapping for the optimal order
    lines.append("")
    lines.append("Recommended pin reassignment:")
    # We need to figure out which net goes to which pin.
    # Build a map from current pin -> net from the edges in crossings + report
    # Actually, we can rebuild from the edges. But we don't have edges in the report.
    # Instead, show the optimal order with position numbers.
    for pos, pin in enumerate(report.optimal_order, 1):
        lines.append(f"  {report.reorderable_ref} pin {pos} = (currently pin {pin})")

    return "\n".join(lines)


def format_report_with_nets(
    report: CrossingReport,
    edges: list[Edge],
) -> str:
    """Format a CrossingReport with net name annotations."""
    lines: list[str] = []
    lines.append(f"Crossing Analysis: {report.fixed_ref} -> {report.reorderable_ref}")
    lines.append("=" * 60)
    lines.append("")

    # Build pin-to-net map for the reorderable connector
    pin_to_net: dict[str, str] = {}
    for edge in edges:
        pin_to_net[edge.reorderable_pin] = edge.net_name

    if report.crossing_count == 0:
        lines.append("No crossings detected. Pin ordering is optimal.")
        lines.append("")
        lines.append(f"Current {report.reorderable_ref} pin order:")
        for i, pin in enumerate(report.current_order, 1):
            net = pin_to_net.get(pin, "?")
            lines.append(f"  Pin {i} = {net} (pin {pin})")
        return "\n".join(lines)

    lines.append(f"Crossings found: {report.crossing_count}")
    lines.append("")

    for i, cp in enumerate(report.crossings, 1):
        lines.append(
            f"  {i}. {cp.edge_a.net_name} ({report.fixed_ref}.{cp.edge_a.fixed_pin}"
            f" -> {report.reorderable_ref}.{cp.edge_a.reorderable_pin})"
            f"  X  {cp.edge_b.net_name} ({report.fixed_ref}.{cp.edge_b.fixed_pin}"
            f" -> {report.reorderable_ref}.{cp.edge_b.reorderable_pin})"
        )

    lines.append("")
    lines.append(f"Current {report.reorderable_ref} pin order:")
    for i, pin in enumerate(report.current_order, 1):
        net = pin_to_net.get(pin, "?")
        lines.append(f"  Pin {i} = {net}")

    lines.append("")
    lines.append(f"Optimal {report.reorderable_ref} pin order:")
    for i, pin in enumerate(report.optimal_order, 1):
        net = pin_to_net.get(pin, "?")
        lines.append(f"  Pin {i} = {net}")

    lines.append("")
    lines.append(f"Crossings after optimization: {report.optimal_crossing_count}")

    if report.optimal_crossing_count > 0:
        lines.append("")
        lines.append(
            "WARNING: Not all crossings can be eliminated by reordering alone."
        )
        lines.append(
            "Remaining crossings will require vias or a second routing layer."
        )

    return "\n".join(lines)


# =========================================================================
# Visualization
# =========================================================================

def format_connection_matrix(
    fixed: PinColumn,
    reorderable: PinColumn,
    edges: list[Edge],
    pin_to_net: dict[str, str],
    label: str = "",
) -> str:
    """Format a connection matrix showing fixed pins as rows and
    reorderable pins as columns, with * marking each connection.

    A crossing-free layout has all * marks forming a monotonic
    top-left to bottom-right diagonal. Any backward jump indicates
    a crossing."""
    # Build set of (fixed_pos, reorderable_pos) for quick lookup
    connections: set[tuple[int, int]] = set()
    for edge in edges:
        fp = fixed.position_of(edge.fixed_pin)
        rp = reorderable.position_of(edge.reorderable_pin)
        connections.add((fp, rp))

    # Column headers: net names for reorderable pins
    col_labels = [pin_to_net.get(p, p) for p in reorderable.pin_order]
    col_width = max(len(lbl) for lbl in col_labels) + 1

    # Row labels: [pin] net for fixed pins
    fixed_pin_net: dict[str, str] = {}
    for edge in edges:
        fixed_pin_net[edge.fixed_pin] = edge.net_name
    row_labels = []
    for p in fixed.pin_order:
        net = fixed_pin_net.get(p, "")
        row_labels.append(f"[{p}] {net}")
    row_label_width = max(len(lbl) for lbl in row_labels) + 1

    lines: list[str] = []
    if label:
        lines.append(label)
        lines.append("")

    # Header row
    header = " " * row_label_width
    for i, lbl in enumerate(col_labels):
        header += f"[{reorderable.pin_order[i]}]".center(col_width)
    lines.append(header)

    net_header = " " * row_label_width
    for lbl in col_labels:
        net_header += lbl.center(col_width)
    lines.append(net_header)

    lines.append(" " * row_label_width + "-" * (col_width * len(col_labels)))

    # Data rows
    for fi, fp in enumerate(fixed.pin_order):
        row = row_labels[fi].ljust(row_label_width)
        for ri in range(len(reorderable.pin_order)):
            if (fi, ri) in connections:
                row += "*".center(col_width)
            else:
                row += ".".center(col_width)
        lines.append(row)

    return "\n".join(lines)


def format_routing_diagram(
    fixed: PinColumn,
    reorderable: PinColumn,
    edges: list[Edge],
    pin_to_net: dict[str, str],
    label: str = "",
) -> str:
    """Format an ASCII routing diagram showing trace paths between two
    connectors.

    Uses a grid-based channel router: fixed pins on the left, reorderable
    pins on the right, traces drawn through the routing channel between them.
    Crossings are marked with X."""
    n_fixed = len(fixed.pin_order)
    n_reorder = len(reorderable.pin_order)
    n_rows = max(n_fixed, n_reorder)

    # Map each edge to (fixed_position, reorderable_position)
    edge_positions: list[tuple[int, int, str]] = []
    for edge in edges:
        fp = fixed.position_of(edge.fixed_pin)
        rp = reorderable.position_of(edge.reorderable_pin)
        edge_positions.append((fp, rp, edge.net_name))

    # Sort by fixed position for consistent rendering
    edge_positions.sort(key=lambda x: x[0])

    # Channel width scales with number of traces to allow room for routing
    channel_width = max(20, len(edges) * 3 + 4)

    # Build the routing grid: each cell is a character
    grid = [[" "] * channel_width for _ in range(n_rows)]

    # For each trace, interpolate from left row (fixed_pos) to right row (reorderable_pos)
    # Track which cells are occupied and by which net
    cell_owner: dict[tuple[int, int], str] = {}

    for fp, rp, net_name in edge_positions:
        # Trace goes from row=fp on the left to row=rp on the right
        # Interpolate through the channel columns
        for col in range(channel_width):
            # Linear interpolation
            t = col / max(channel_width - 1, 1)
            row_f = fp + t * (rp - fp)
            row = int(round(row_f))
            row = max(0, min(n_rows - 1, row))

            key = (row, col)
            if key in cell_owner and cell_owner[key] != net_name:
                # Crossing detected
                grid[row][col] = "X"
                cell_owner[key] = "CROSSING"
            elif key not in cell_owner:
                grid[row][col] = "-"
                cell_owner[key] = net_name

    # Build fixed-side labels
    fixed_labels = []
    fixed_pin_net: dict[str, str] = {}
    for edge in edges:
        fixed_pin_net[edge.fixed_pin] = edge.net_name
    for i, p in enumerate(fixed.pin_order):
        net = fixed_pin_net.get(p, "")
        fixed_labels.append(f"{net:>5} [{p}]")
    # Pad if fewer fixed than rows
    while len(fixed_labels) < n_rows:
        fixed_labels.append(" " * 9)

    # Build reorderable-side labels
    reorder_labels = []
    for i, p in enumerate(reorderable.pin_order):
        net = pin_to_net.get(p, "")
        reorder_labels.append(f"[{p}] {net}")
    while len(reorder_labels) < n_rows:
        reorder_labels.append("")

    # Assemble output
    lines: list[str] = []
    if label:
        lines.append(label)
        lines.append("")

    left_header = f"{fixed.ref:>9}"
    right_header = f"{reorderable.ref}"
    lines.append(f"{left_header}  {'Routing Channel':^{channel_width}}  {right_header}")
    lines.append(f"{'':>9}  {'=' * channel_width}")

    for row_idx in range(n_rows):
        left = fixed_labels[row_idx] if row_idx < len(fixed_labels) else " " * 9
        right = reorder_labels[row_idx] if row_idx < len(reorder_labels) else ""
        channel = "".join(grid[row_idx])
        lines.append(f"{left} |{channel}| {right}")

    lines.append(f"{'':>9}  {'=' * channel_width}")

    # Legend
    crossings = sum(1 for row in grid for c in row if c == "X")
    if crossings > 0:
        lines.append(f"  X = crossing point ({crossings} found)")
    else:
        lines.append("  All traces parallel, no crossings.")

    return "\n".join(lines)


def format_before_after(
    fixed: PinColumn,
    current_reorderable: PinColumn,
    optimal_reorderable: PinColumn,
    edges: list[Edge],
    pin_to_net: dict[str, str],
) -> str:
    """Format a before/after comparison showing both the current
    and optimal layouts."""
    lines: list[str] = []

    lines.append("CURRENT LAYOUT")
    lines.append(format_routing_diagram(
        fixed, current_reorderable, edges, pin_to_net,
    ))
    lines.append("")
    lines.append(format_connection_matrix(
        fixed, current_reorderable, edges, pin_to_net,
    ))

    lines.append("")
    lines.append("")
    lines.append("OPTIMAL LAYOUT")
    lines.append(format_routing_diagram(
        fixed, optimal_reorderable, edges, pin_to_net,
    ))
    lines.append("")
    lines.append(format_connection_matrix(
        fixed, optimal_reorderable, edges, pin_to_net,
    ))

    return "\n".join(lines)


# =========================================================================
# CLI entry point
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze trace crossings between two connectors in a KiCad netlist.",
        epilog="Example: python tools/crossing_analyzer.py "
               "output/netlists/microsd_breakout.net J1 J2 --exclude J1:SH",
    )
    parser.add_argument("netlist", help="Path to a KiCad .net file generated by SKiDL")
    parser.add_argument("fixed_ref", help="Reference designator of the fixed connector (e.g. J1)")
    parser.add_argument("reorderable_ref", help="Reference designator of the reorderable connector (e.g. J2)")
    parser.add_argument(
        "--exclude", nargs="*", default=[], metavar="REF:PIN",
        help="Exclude pins from analysis (e.g. J1:SH). "
             "Useful for shield/mounting pins that are not in the signal routing channel.",
    )
    parser.add_argument(
        "--diagram", action="store_true",
        help="Show ASCII routing diagrams and connection matrices (before/after).",
    )
    args = parser.parse_args()

    if not Path(args.netlist).exists():
        print(f"Error: file not found: {args.netlist}")
        sys.exit(1)

    # Parse exclusions into a set of (ref, pin) tuples
    exclude_set: set[tuple[str, str]] = set()
    for exc in args.exclude:
        if ":" not in exc:
            print(f"Error: exclusion must be REF:PIN format, got '{exc}'")
            sys.exit(1)
        ref, pin = exc.split(":", 1)
        exclude_set.add((ref, pin))

    # Parse netlist
    data = parse_netlist(args.netlist)

    fixed_ref = args.fixed_ref
    reorderable_ref = args.reorderable_ref

    # Validate refs exist
    for ref in (fixed_ref, reorderable_ref):
        if ref not in data["components"]:
            print(f"Error: component '{ref}' not found in netlist.")
            print(f"Available components: {', '.join(sorted(data['components'].keys()))}")
            sys.exit(1)

    # Infer pin orders from netlist, applying exclusions
    fixed_pins = [
        p for p in infer_pin_order(fixed_ref, data["nets"])
        if (fixed_ref, p) not in exclude_set
    ]
    reorderable_pins = [
        p for p in infer_pin_order(reorderable_ref, data["nets"])
        if (reorderable_ref, p) not in exclude_set
    ]

    fixed = PinColumn(ref=fixed_ref, pin_order=fixed_pins)
    reorderable = PinColumn(ref=reorderable_ref, pin_order=reorderable_pins)

    # Extract edges, filtering out excluded pins
    edges = [
        e for e in extract_edges(data["nets"], fixed_ref, reorderable_ref)
        if (fixed_ref, e.fixed_pin) not in exclude_set
        and (reorderable_ref, e.reorderable_pin) not in exclude_set
    ]

    if not edges:
        print(f"No nets connect {fixed_ref} and {reorderable_ref}.")
        sys.exit(0)

    if exclude_set:
        excluded_str = ", ".join(f"{r}:{p}" for r, p in sorted(exclude_set))
        print(f"(Excluding pins: {excluded_str})")
        print()

    # Analyze
    report = analyze_connectors(fixed, reorderable, edges)

    # Print report
    print(format_report_with_nets(report, edges))

    # Optional diagram output
    if args.diagram:
        pin_to_net: dict[str, str] = {}
        for edge in edges:
            pin_to_net[edge.reorderable_pin] = edge.net_name

        optimal = PinColumn(ref=reorderable_ref, pin_order=report.optimal_order)

        if report.crossing_count > 0:
            # Show before/after comparison
            print()
            print(format_before_after(fixed, reorderable, optimal, edges, pin_to_net))
        else:
            # Already optimal, show current layout
            print()
            print(format_routing_diagram(
                fixed, reorderable, edges, pin_to_net,
                label="CURRENT LAYOUT (optimal)",
            ))
            print()
            print(format_connection_matrix(
                fixed, reorderable, edges, pin_to_net,
                label="CONNECTION MATRIX",
            ))


if __name__ == "__main__":
    main()
