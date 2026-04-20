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
# Multi-layer data model
# =========================================================================

@dataclass
class LayerEdge:
    """An edge between pins on components in adjacent layers."""
    net_name: str
    source_ref: str
    source_pin: str
    target_ref: str
    target_pin: str


@dataclass
class LayerPairCrossing:
    """Two edges between adjacent layers that cross."""
    edge_a: LayerEdge
    edge_b: LayerEdge


@dataclass
class LayerPairReport:
    """Crossing report for one pair of adjacent layers."""
    source_layer_idx: int
    target_layer_idx: int
    source_refs: list[str]
    target_refs: list[str]
    crossing_count: int
    crossings: list[LayerPairCrossing]


@dataclass
class MultilayerReport:
    """Full crossing analysis across all layer pairs."""
    total_crossings: int
    total_crossings_after: int
    layer_pair_reports: list[LayerPairReport]
    original_orders: dict[str, list[str]]
    optimized_orders: dict[str, list[str]]
    iterations: int


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
# Multi-layer analysis (Sugiyama-style barycenter sweep)
# =========================================================================

def layer_global_positions(
    components: list[PinColumn],
) -> dict[tuple[str, str], int]:
    """Map (ref, pin) to a global position index across a layer.

    Components are laid out sequentially: if component A has 3 pins and
    component B has 2 pins, B's first pin is at position 3.
    """
    pos = 0
    result: dict[tuple[str, str], int] = {}
    for comp in components:
        for pin in comp.pin_order:
            result[(comp.ref, pin)] = pos
            pos += 1
    return result


def count_layer_pair_crossings(
    source_layer: list[PinColumn],
    target_layer: list[PinColumn],
    edges: list[LayerEdge],
) -> list[LayerPairCrossing]:
    """Count crossings between edges connecting two adjacent layers.

    Generalizes count_crossings() to multi-component layers with
    global position indices.
    """
    source_pos = layer_global_positions(source_layer)
    target_pos = layer_global_positions(target_layer)

    # Filter to edges that have valid positions in both layers
    valid_edges: list[LayerEdge] = []
    for e in edges:
        sk = (e.source_ref, e.source_pin)
        tk = (e.target_ref, e.target_pin)
        if sk in source_pos and tk in target_pos:
            valid_edges.append(e)

    crossings: list[LayerPairCrossing] = []
    for a, b in combinations(valid_edges, 2):
        si = source_pos[(a.source_ref, a.source_pin)]
        sk = source_pos[(b.source_ref, b.source_pin)]
        tj = target_pos[(a.target_ref, a.target_pin)]
        tl = target_pos[(b.target_ref, b.target_pin)]
        if (si < sk and tj > tl) or (si > sk and tj < tl):
            crossings.append(LayerPairCrossing(a, b))
    return crossings


def extract_layer_pair_edges(
    nets: dict[str, list[tuple[str, str]]],
    source_refs: set[str],
    target_refs: set[str],
) -> list[LayerEdge]:
    """Extract edges between two sets of components from parsed net data.

    A net produces edges if it has pins on components in both the source
    and target sets.
    """
    edges: list[LayerEdge] = []
    for net_name, nodes in nets.items():
        source_pins = [(ref, pin) for ref, pin in nodes if ref in source_refs]
        target_pins = [(ref, pin) for ref, pin in nodes if ref in target_refs]
        for sref, spin in source_pins:
            for tref, tpin in target_pins:
                edges.append(LayerEdge(
                    net_name=net_name,
                    source_ref=sref, source_pin=spin,
                    target_ref=tref, target_pin=tpin,
                ))
    return edges


def _compute_multilayer_barycenters(
    component_ref: str,
    adjacent_layer: list[PinColumn],
    edges: list[LayerEdge],
) -> dict[str, float]:
    """Compute barycenter for each pin of a component based on its
    connections to an adjacent layer.

    Works regardless of whether the component is on the source or target
    side of the edges.
    """
    adj_pos = layer_global_positions(adjacent_layer)

    pin_positions: dict[str, list[int]] = {}
    for edge in edges:
        if edge.source_ref == component_ref:
            pin = edge.source_pin
            adj_key = (edge.target_ref, edge.target_pin)
        elif edge.target_ref == component_ref:
            pin = edge.target_pin
            adj_key = (edge.source_ref, edge.source_pin)
        else:
            continue

        if adj_key in adj_pos:
            pin_positions.setdefault(pin, []).append(adj_pos[adj_key])

    barycenters: dict[str, float] = {}
    for pin, positions in pin_positions.items():
        barycenters[pin] = sum(positions) / len(positions)

    return barycenters


def _reorder_pins_by_barycenter(
    component: PinColumn,
    barycenters: dict[str, float],
) -> PinColumn:
    """Reorder a component's pins by their barycenter values.

    Pins without a barycenter (unconnected to the adjacent layer)
    keep their relative order and are placed at the end.
    """
    connected = [p for p in component.pin_order if p in barycenters]
    unconnected = [p for p in component.pin_order if p not in barycenters]
    connected.sort(key=lambda p: barycenters[p])
    return PinColumn(ref=component.ref, pin_order=connected + unconnected)


def _reorder_components_in_layer(
    layer: list[PinColumn],
    adjacent_layer: list[PinColumn],
    edges: list[LayerEdge],
) -> list[PinColumn]:
    """Reorder components within a layer by their aggregate barycenter.

    Each component's aggregate barycenter is the average of all its
    pin barycenters relative to the adjacent layer.
    """
    adj_pos = layer_global_positions(adjacent_layer)

    comp_barycenters: dict[str, float] = {}
    for comp in layer:
        positions: list[int] = []
        for edge in edges:
            if edge.source_ref == comp.ref:
                adj_key = (edge.target_ref, edge.target_pin)
            elif edge.target_ref == comp.ref:
                adj_key = (edge.source_ref, edge.source_pin)
            else:
                continue
            if adj_key in adj_pos:
                positions.append(adj_pos[adj_key])

        if positions:
            comp_barycenters[comp.ref] = sum(positions) / len(positions)
        else:
            comp_barycenters[comp.ref] = float("inf")

    return sorted(layer, key=lambda c: comp_barycenters[c.ref])


def _build_component_layer_map(
    layers: list[list[PinColumn]],
) -> dict[str, int]:
    """Map component ref -> layer index."""
    result: dict[str, int] = {}
    for i, layer in enumerate(layers):
        for comp in layer:
            result[comp.ref] = i
    return result


def _expand_with_virtual_nodes(
    layers: list[list[PinColumn]],
    nets: dict[str, list[tuple[str, str]]],
) -> tuple[list[list[PinColumn]], list[list[LayerEdge]], set[str]]:
    """Insert virtual nodes for long edges spanning non-adjacent layers.

    For a net connecting components in layers 0 and 2, a virtual pin is
    inserted in layer 1 so the sweep can account for all routing paths.

    Only pins present in the layers' pin_order lists are considered,
    so caller exclusions are automatically respected.

    Returns:
        expanded_layers: layers with virtual PinColumn added where needed.
        layer_pair_edges: pre-computed edge list for each adjacent pair.
        virtual_refs: set of virtual component refs (always reorderable).
    """
    comp_layer = _build_component_layer_map(layers)
    n_layers = len(layers)

    # Build set of active (ref, pin) from the layer data
    active_pins: set[tuple[str, str]] = set()
    for layer in layers:
        for comp in layer:
            for pin in comp.pin_order:
                active_pins.add((comp.ref, pin))

    # Collect virtual pins needed per intermediate layer
    virt_pins_per_layer: dict[int, list[str]] = {}
    virt_counter = 0

    # Build edge lists for each layer pair
    pair_edges: list[list[LayerEdge]] = [[] for _ in range(n_layers - 1)]

    for net_name, nodes in nets.items():
        # Group nodes by layer, filtering to active pins only
        nodes_by_layer: dict[int, list[tuple[str, str]]] = {}
        for ref, pin in nodes:
            if ref not in comp_layer:
                continue
            if (ref, pin) not in active_pins:
                continue
            li = comp_layer[ref]
            nodes_by_layer.setdefault(li, []).append((ref, pin))

        # For each pair of layer groups, create edges
        sorted_layers = sorted(nodes_by_layer.keys())
        for a_idx in range(len(sorted_layers)):
            for b_idx in range(a_idx + 1, len(sorted_layers)):
                src_li = sorted_layers[a_idx]
                tgt_li = sorted_layers[b_idx]
                gap = tgt_li - src_li

                for sref, spin in nodes_by_layer[src_li]:
                    for tref, tpin in nodes_by_layer[tgt_li]:
                        if gap == 1:
                            # Direct edge: no virtual nodes needed
                            pair_edges[src_li].append(
                                LayerEdge(net_name, sref, spin, tref, tpin)
                            )
                        else:
                            # Long edge: create virtual pins in intermediate layers
                            chain: list[tuple[str, str]] = [(sref, spin)]
                            for mid_li in range(src_li + 1, tgt_li):
                                vref = f"_virt_L{mid_li}"
                                vid = f"_v{virt_counter}"
                                virt_counter += 1
                                virt_pins_per_layer.setdefault(mid_li, []).append(vid)
                                chain.append((vref, vid))
                            chain.append((tref, tpin))

                            # Create edge segments along the chain
                            for seg in range(len(chain) - 1):
                                sr, sp = chain[seg]
                                tr, tp = chain[seg + 1]
                                edge_li = src_li + seg
                                pair_edges[edge_li].append(
                                    LayerEdge(net_name, sr, sp, tr, tp)
                                )

    # Build expanded layers with virtual PinColumns
    expanded: list[list[PinColumn]] = []
    virtual_refs: set[str] = set()
    for i, layer in enumerate(layers):
        new_layer = [
            PinColumn(ref=c.ref, pin_order=list(c.pin_order)) for c in layer
        ]
        if i in virt_pins_per_layer:
            vref = f"_virt_L{i}"
            virtual_refs.add(vref)
            new_layer.append(PinColumn(ref=vref, pin_order=virt_pins_per_layer[i]))
        expanded.append(new_layer)

    return expanded, pair_edges, virtual_refs


def sweep_optimize(
    layers: list[list[PinColumn]],
    reorderable_refs: set[str],
    nets: dict[str, list[tuple[str, str]]],
    max_iterations: int = 10,
) -> MultilayerReport:
    """Minimize crossings across all layer pairs using Sugiyama-style
    barycenter sweep.

    Handles long edges (nets spanning non-adjacent layers) by inserting
    virtual nodes in intermediate layers so the sweep accounts for all
    routing paths.

    Args:
        layers: List of layers, each layer is a list of PinColumn objects.
                layers[0] is the leftmost/topmost layer.
        reorderable_refs: Set of component refs whose pin order can change.
        nets: Parsed netlist data (net_name -> [(ref, pin), ...]).
        max_iterations: Maximum forward+backward sweep iterations.

    Returns:
        MultilayerReport with crossing counts before/after and optimized orders.
    """
    # Save original orders before any modification
    original_orders: dict[str, list[str]] = {}
    for layer in layers:
        for comp in layer:
            original_orders[comp.ref] = list(comp.pin_order)

    # Expand layers with virtual nodes for long edges
    layers, layer_pair_edges, virtual_refs = _expand_with_virtual_nodes(layers, nets)

    # Virtual nodes are always reorderable
    all_reorderable = reorderable_refs | virtual_refs

    # Count initial crossings
    def total_crossing_count() -> int:
        total = 0
        for i, edges in enumerate(layer_pair_edges):
            crossings = count_layer_pair_crossings(
                layers[i], layers[i + 1], edges,
            )
            total += len(crossings)
        return total

    initial_total = total_crossing_count()
    best_total = initial_total
    iterations = 0

    for iteration in range(max_iterations):
        improved = False

        # Forward sweep: fix layer i, optimize layer i+1
        for i in range(len(layers) - 1):
            edges = layer_pair_edges[i]
            new_layer: list[PinColumn] = []
            for comp in layers[i + 1]:
                if comp.ref in all_reorderable:
                    bc = _compute_multilayer_barycenters(
                        comp.ref, layers[i], edges,
                    )
                    new_layer.append(_reorder_pins_by_barycenter(comp, bc))
                else:
                    new_layer.append(comp)
            layers[i + 1] = _reorder_components_in_layer(
                new_layer, layers[i], edges,
            )

        # Backward sweep: fix layer i+1, optimize layer i
        for i in range(len(layers) - 2, -1, -1):
            edges = layer_pair_edges[i]
            new_layer = []
            for comp in layers[i]:
                if comp.ref in all_reorderable:
                    bc = _compute_multilayer_barycenters(
                        comp.ref, layers[i + 1], edges,
                    )
                    new_layer.append(_reorder_pins_by_barycenter(comp, bc))
                else:
                    new_layer.append(comp)
            layers[i] = _reorder_components_in_layer(
                new_layer, layers[i + 1], edges,
            )

        current_total = total_crossing_count()
        iterations = iteration + 1

        if current_total < best_total:
            best_total = current_total
            improved = True

        if not improved:
            break

    # Build layer pair reports for final state (filter out virtual refs from display)
    pair_reports: list[LayerPairReport] = []
    for i, edges in enumerate(layer_pair_edges):
        crossings = count_layer_pair_crossings(
            layers[i], layers[i + 1], edges,
        )
        pair_reports.append(LayerPairReport(
            source_layer_idx=i,
            target_layer_idx=i + 1,
            source_refs=[c.ref for c in layers[i] if c.ref not in virtual_refs],
            target_refs=[c.ref for c in layers[i + 1] if c.ref not in virtual_refs],
            crossing_count=len(crossings),
            crossings=crossings,
        ))

    # Collect optimized orders (real components only)
    optimized_orders: dict[str, list[str]] = {}
    for layer in layers:
        for comp in layer:
            if comp.ref not in virtual_refs:
                optimized_orders[comp.ref] = list(comp.pin_order)

    return MultilayerReport(
        total_crossings=initial_total,
        total_crossings_after=best_total,
        layer_pair_reports=pair_reports,
        original_orders=original_orders,
        optimized_orders=optimized_orders,
        iterations=iterations,
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
# Multi-layer report formatting
# =========================================================================

def format_multilayer_report(
    report: MultilayerReport,
    nets: dict[str, list[tuple[str, str]]],
) -> str:
    """Format a MultilayerReport as human-readable text."""
    lines: list[str] = []
    lines.append("Multi-Layer Crossing Analysis")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total crossings (before): {report.total_crossings}")
    lines.append(f"Total crossings (after):  {report.total_crossings_after}")
    lines.append(f"Sweep iterations:         {report.iterations}")
    lines.append("")

    for pair_report in report.layer_pair_reports:
        src = ", ".join(pair_report.source_refs)
        tgt = ", ".join(pair_report.target_refs)
        lines.append(
            f"Layer {pair_report.source_layer_idx} [{src}] -> "
            f"Layer {pair_report.target_layer_idx} [{tgt}]"
        )
        lines.append(f"  Crossings: {pair_report.crossing_count}")
        for i, cp in enumerate(pair_report.crossings, 1):
            lines.append(
                f"    {i}. {cp.edge_a.net_name} "
                f"({cp.edge_a.source_ref}.{cp.edge_a.source_pin} -> "
                f"{cp.edge_a.target_ref}.{cp.edge_a.target_pin})  X  "
                f"{cp.edge_b.net_name} "
                f"({cp.edge_b.source_ref}.{cp.edge_b.source_pin} -> "
                f"{cp.edge_b.target_ref}.{cp.edge_b.target_pin})"
            )
        lines.append("")

    # Show reordering recommendations
    changed = {
        ref for ref in report.optimized_orders
        if report.optimized_orders[ref] != report.original_orders.get(ref)
    }
    if changed:
        lines.append("Recommended pin reorderings:")
        for ref in sorted(changed):
            original = ", ".join(report.original_orders[ref])
            optimized = ", ".join(report.optimized_orders[ref])
            lines.append(f"  {ref}: [{original}] -> [{optimized}]")
    else:
        lines.append("No reordering needed.")

    if report.total_crossings_after > 0:
        lines.append("")
        lines.append(
            "WARNING: Not all crossings can be eliminated by reordering alone."
        )
        lines.append(
            "Remaining crossings will require vias or a second routing layer."
        )

    return "\n".join(lines)


# =========================================================================
# CLI entry point
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze trace crossings in a KiCad netlist.",
        epilog=(
            "Pair mode:  crossing_analyzer.py net.net J1 J2 --exclude J1:SH\n"
            "Sweep mode: crossing_analyzer.py net.net --layers 'J1 | R1,C1 | J2' "
            "--reorderable J2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("netlist", help="Path to a KiCad .net file generated by SKiDL")
    parser.add_argument(
        "fixed_ref", nargs="?", default=None,
        help="(Pair mode) Reference designator of the fixed connector (e.g. J1)",
    )
    parser.add_argument(
        "reorderable_ref", nargs="?", default=None,
        help="(Pair mode) Reference designator of the reorderable connector (e.g. J2)",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[], metavar="REF:PIN",
        help="Exclude pins from analysis (e.g. J1:SH).",
    )
    parser.add_argument(
        "--diagram", action="store_true",
        help="Show ASCII routing diagrams and connection matrices.",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help=(
            "(Sweep mode) Layer specification: components per layer separated by |. "
            "Multiple components in a layer separated by commas. "
            "Example: 'J1 | R1,R2,C1 | J2'"
        ),
    )
    parser.add_argument(
        "--reorderable", nargs="*", default=[], metavar="REF",
        help="(Sweep mode) Components whose pin order can be changed.",
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

    if exclude_set:
        excluded_str = ", ".join(f"{r}:{p}" for r, p in sorted(exclude_set))
        print(f"(Excluding pins: {excluded_str})")
        print()

    # Dispatch to the appropriate mode
    if args.layers is not None:
        _main_sweep(args, data, exclude_set)
    elif args.fixed_ref and args.reorderable_ref:
        _main_pair(args, data, exclude_set)
    else:
        print("Error: provide either (fixed_ref, reorderable_ref) for pair mode")
        print("       or --layers for sweep mode.")
        sys.exit(1)


def _main_pair(args, data: dict, exclude_set: set[tuple[str, str]]):
    """Pair mode: analyze crossings between two connectors."""
    fixed_ref = args.fixed_ref
    reorderable_ref = args.reorderable_ref

    for ref in (fixed_ref, reorderable_ref):
        if ref not in data["components"]:
            print(f"Error: component '{ref}' not found in netlist.")
            print(f"Available: {', '.join(sorted(data['components'].keys()))}")
            sys.exit(1)

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

    edges = [
        e for e in extract_edges(data["nets"], fixed_ref, reorderable_ref)
        if (fixed_ref, e.fixed_pin) not in exclude_set
        and (reorderable_ref, e.reorderable_pin) not in exclude_set
    ]

    if not edges:
        print(f"No nets connect {fixed_ref} and {reorderable_ref}.")
        sys.exit(0)

    report = analyze_connectors(fixed, reorderable, edges)
    print(format_report_with_nets(report, edges))

    if args.diagram:
        pin_to_net: dict[str, str] = {}
        for edge in edges:
            pin_to_net[edge.reorderable_pin] = edge.net_name

        optimal = PinColumn(ref=reorderable_ref, pin_order=report.optimal_order)

        if report.crossing_count > 0:
            print()
            print(format_before_after(fixed, reorderable, optimal, edges, pin_to_net))
        else:
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


def _main_sweep(args, data: dict, exclude_set: set[tuple[str, str]]):
    """Sweep mode: multi-layer crossing analysis."""
    # Parse layer specification: "J1 | R1,R2,C1 | J2"
    layer_specs = [
        [ref.strip() for ref in group.split(",")]
        for group in args.layers.split("|")
    ]

    # Validate all refs exist
    all_refs = [ref for group in layer_specs for ref in group]
    for ref in all_refs:
        if ref not in data["components"]:
            print(f"Error: component '{ref}' not found in netlist.")
            print(f"Available: {', '.join(sorted(data['components'].keys()))}")
            sys.exit(1)

    # Build reorderable set
    reorderable_refs = set(args.reorderable)
    if not reorderable_refs:
        print("Warning: no --reorderable refs specified; no optimization possible.")

    # Build layer PinColumn lists, applying exclusions
    layers: list[list[PinColumn]] = []
    for group in layer_specs:
        layer: list[PinColumn] = []
        for ref in group:
            pins = [
                p for p in infer_pin_order(ref, data["nets"])
                if (ref, p) not in exclude_set
            ]
            layer.append(PinColumn(ref=ref, pin_order=pins))
        layers.append(layer)

    # Print layer summary
    print("Layer assignment:")
    for i, layer in enumerate(layers):
        refs = ", ".join(c.ref for c in layer)
        pins = sum(len(c.pin_order) for c in layer)
        print(f"  Layer {i}: [{refs}] ({pins} pins)")
    print()

    # Run sweep
    report = sweep_optimize(layers, reorderable_refs, data["nets"])
    print(format_multilayer_report(report, data["nets"]))


if __name__ == "__main__":
    main()
