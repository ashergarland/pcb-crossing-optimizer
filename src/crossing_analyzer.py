"""PCB Crossing Optimizer – crossing minimization for SKiDL-generated netlists.

Detects trace crossings across component layers and computes optimal
pin orderings for reorderable connectors to minimize or eliminate
crossings for single-layer routing.

Algorithm: Sugiyama-style barycenter sweep with virtual node insertion
for long edges spanning non-adjacent layers.

Usage (CLI):
    python crossing_analyzer.py <netlist.net> --layers "J1 | R1,C1 | J2" --reorderable J2

Programmatic usage from SKiDL scripts:
    from crossing_analyzer import PinColumn, sweep_optimize, format_multilayer_report
"""

from __future__ import annotations

import json
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


@dataclass
class PinAssignment:
    """A single pin's assignment in a footprint plan."""
    pin: str
    net: Optional[str]   # None = NC
    status: str          # "locked" | "optimized" | "unmatched"
    routes_to: Optional[str] = None  # e.g. "J2.1" for optimized pins


@dataclass
class FootprintPlan:
    """Result of a plan-footprint analysis."""
    target_ref: str
    pin_map: list[PinAssignment]
    crossings_before: int
    crossings_after: int
    iterations: int
    passive_reorderings: dict[str, list[str]]


# =========================================================================
# Formatting helpers
# =========================================================================

def _format_pin_ref(ref: str, pin: str) -> str:
    """Format a pin reference for display, hiding virtual node internals."""
    if ref.startswith("_virt_"):
        return "[pass-through]"
    return f"{ref}.{pin}"


# =========================================================================
# Core analysis
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

    Two edges (i->j) and (k->l) cross iff (i < k and j > l) or
    (i > k and j < l), where i,k are positions on the source layer
    and j,l are positions on the target layer.
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
        src = ", ".join(r for r in pair_report.source_refs if not r.startswith("_virt_"))
        tgt = ", ".join(r for r in pair_report.target_refs if not r.startswith("_virt_"))
        src_label = src or "[pass-through]"
        tgt_label = tgt or "[pass-through]"
        lines.append(
            f"Layer {pair_report.source_layer_idx} [{src_label}] -> "
            f"Layer {pair_report.target_layer_idx} [{tgt_label}]"
        )
        lines.append(f"  Crossings: {pair_report.crossing_count}")
        for i, cp in enumerate(pair_report.crossings, 1):
            a_src = _format_pin_ref(cp.edge_a.source_ref, cp.edge_a.source_pin)
            a_tgt = _format_pin_ref(cp.edge_a.target_ref, cp.edge_a.target_pin)
            b_src = _format_pin_ref(cp.edge_b.source_ref, cp.edge_b.source_pin)
            b_tgt = _format_pin_ref(cp.edge_b.target_ref, cp.edge_b.target_pin)
            lines.append(
                f"    {i}. {cp.edge_a.net_name} "
                f"({a_src} -> {a_tgt})  X  "
                f"{cp.edge_b.net_name} "
                f"({b_src} -> {b_tgt})"
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


def report_to_dict(
    report: MultilayerReport,
    nets: dict[str, list[tuple[str, str]]],
) -> dict:
    """Convert a MultilayerReport to a JSON-serializable dict."""
    layer_pairs = []
    for pr in report.layer_pair_reports:
        crossings = []
        for cp in pr.crossings:
            crossings.append({
                "edge_a": {
                    "net": cp.edge_a.net_name,
                    "source": _format_pin_ref(cp.edge_a.source_ref, cp.edge_a.source_pin),
                    "target": _format_pin_ref(cp.edge_a.target_ref, cp.edge_a.target_pin),
                },
                "edge_b": {
                    "net": cp.edge_b.net_name,
                    "source": _format_pin_ref(cp.edge_b.source_ref, cp.edge_b.source_pin),
                    "target": _format_pin_ref(cp.edge_b.target_ref, cp.edge_b.target_pin),
                },
            })
        layer_pairs.append({
            "source_layer": pr.source_layer_idx,
            "target_layer": pr.target_layer_idx,
            "source_refs": [r for r in pr.source_refs if not r.startswith("_virt_")],
            "target_refs": [r for r in pr.target_refs if not r.startswith("_virt_")],
            "crossing_count": pr.crossing_count,
            "crossings": crossings,
        })

    reorderings = {}
    for ref in sorted(report.optimized_orders):
        orig = report.original_orders.get(ref, [])
        opt = report.optimized_orders[ref]
        if orig != opt:
            reorderings[ref] = {"original": orig, "optimized": opt}

    return {
        "total_crossings_before": report.total_crossings,
        "total_crossings_after": report.total_crossings_after,
        "iterations": report.iterations,
        "layer_pairs": layer_pairs,
        "reorderings": reorderings,
    }


# =========================================================================
# Footprint planning
# =========================================================================

def parse_pin_locks(lock_args: list[str]) -> dict[str, str | None]:
    """Parse --lock arguments into a pin-to-net mapping.

    Each arg is 'PIN=NET' or 'PIN=NC'.
    Returns dict mapping pin ID to net name (or None for NC).
    """
    locks: dict[str, str | None] = {}
    for arg in lock_args:
        if "=" not in arg:
            raise ValueError(
                f"Invalid --lock format: '{arg}'. Expected PIN=NET or PIN=NC."
            )
        pin, net = arg.split("=", 1)
        pin = pin.strip()
        net = net.strip()
        if not pin:
            raise ValueError(f"Empty pin in --lock: '{arg}'")
        if net.upper() == "NC":
            locks[pin] = None
        else:
            locks[pin] = net
    return locks


def _find_primary_route(
    target_ref: str,
    target_pin: str,
    net_name: str,
    nets: dict[str, list[tuple[str, str]]],
) -> str | None:
    """Find the primary non-target endpoint for a net (for 'routes to' display)."""
    nodes = nets.get(net_name, [])
    for ref, pin in nodes:
        if ref != target_ref and not ref.startswith("TP"):
            return f"{ref}.{pin}"
    return None


def build_pin_map(
    target_ref: str,
    all_pins: list[str],
    locks: dict[str, str | None],
    optimized_signal_pins: list[str],
    pin_to_net: dict[str, str],
    nets: dict[str, list[tuple[str, str]]],
    unmatched_mode: str,
) -> list[PinAssignment]:
    """Build the final pin map by merging locked, optimized, and unmatched pins.

    Args:
        target_ref: Component ref (for routes_to lookup).
        all_pins: All pin IDs in physical position order.
        locks: Pin-to-net locks from --lock (None = NC).
        optimized_signal_pins: Pin IDs in sweep-optimized order (signal pins only).
        pin_to_net: Mapping of pin ID to net name (from netlist).
        nets: Full net connectivity for routes_to lookup.
        unmatched_mode: 'start', 'end', or 'split'.

    Returns:
        List of PinAssignment in physical position order.
    """
    # Categorize pins
    locked_pins = set(locks.keys())
    signal_pins = set(optimized_signal_pins)
    unmatched_pins = [
        p for p in all_pins
        if p not in locked_pins and p not in signal_pins
    ]

    # Build the assignment slots
    n = len(all_pins)
    result: list[PinAssignment | None] = [None] * n
    pin_to_idx = {p: i for i, p in enumerate(all_pins)}

    # 1. Place locked pins at their positions
    for pin, net in locks.items():
        if pin in pin_to_idx:
            idx = pin_to_idx[pin]
            result[idx] = PinAssignment(
                pin=pin, net=net, status="locked",
                routes_to=_find_primary_route(target_ref, pin, net, nets) if net else None,
            )

    # 2. Collect open slots (not locked)
    open_slots = [i for i in range(n) if result[i] is None]

    # 3. Place unmatched pins per mode, filling from the open slots
    if unmatched_mode == "start":
        unmatched_slots = open_slots[:len(unmatched_pins)]
        signal_slots = open_slots[len(unmatched_pins):]
    elif unmatched_mode == "split":
        half = len(unmatched_pins) // 2
        unmatched_start = unmatched_pins[:half]
        unmatched_end = unmatched_pins[half:]
        signal_slot_count = len(open_slots) - len(unmatched_pins)
        unmatched_slots = open_slots[:len(unmatched_start)] + open_slots[len(unmatched_start) + signal_slot_count:]
        signal_slots = open_slots[len(unmatched_start):len(unmatched_start) + signal_slot_count]
        # Rebuild unmatched_pins to match the split order
        unmatched_pins = unmatched_start + unmatched_end
    else:  # "end" (default)
        signal_slots = open_slots[:len(optimized_signal_pins)]
        unmatched_slots = open_slots[len(optimized_signal_pins):]

    # 4. Place optimized signal pins in sweep order
    for sig_pin, slot_idx in zip(optimized_signal_pins, signal_slots):
        net = pin_to_net.get(sig_pin)
        result[slot_idx] = PinAssignment(
            pin=all_pins[slot_idx], net=net, status="optimized",
            routes_to=_find_primary_route(target_ref, sig_pin, net, nets) if net else None,
        )

    # 5. Place unmatched pins
    for um_pin, slot_idx in zip(unmatched_pins, unmatched_slots):
        net = pin_to_net.get(um_pin)
        result[slot_idx] = PinAssignment(
            pin=all_pins[slot_idx], net=net, status="unmatched",
            routes_to=None,
        )

    # Fill any remaining Nones (shouldn't happen but defensive)
    for i in range(n):
        if result[i] is None:
            result[i] = PinAssignment(pin=all_pins[i], net=None, status="unmatched")

    return result


def plan_footprint(
    target_ref: str,
    target_pins: list[str],
    anchor_layers: list[list[PinColumn]],
    nets: dict[str, list[tuple[str, str]]],
    locks: dict[str, str | None],
    unmatched: str = "end",
    exclude_nets: set[str] | None = None,
) -> FootprintPlan:
    """Compute an optimal pin map for a custom footprint.

    Args:
        target_ref: Ref of the component being designed.
        target_pins: All pin IDs in physical position order.
        anchor_layers: Fixed components organized in layers (outermost first).
        nets: Parsed netlist connectivity.
        locks: Pin-to-net locks (from parse_pin_locks).
        unmatched: Placement mode for unconnected pins.
        exclude_nets: Net names to exclude from analysis.

    Returns:
        FootprintPlan with complete pin map proposal.
    """
    exclude = exclude_nets or set()

    # Filter nets
    filtered_nets = {
        name: nodes for name, nodes in nets.items()
        if name not in exclude
    }

    # Build pin-to-net mapping for the target component
    pin_to_net: dict[str, str] = {}
    for net_name, nodes in filtered_nets.items():
        for ref, pin in nodes:
            if ref == target_ref:
                pin_to_net[pin] = net_name

    # Identify locked, signal, and unmatched pins
    locked_pins = set(locks.keys())
    signal_pins = [
        p for p in target_pins
        if p not in locked_pins and p in pin_to_net
    ]
    # Unmatched: not locked and not connected to any analyzed net
    # (will be placed by build_pin_map)

    # Build layers for sweep: anchor_layers + [target with signal pins only]
    # Detect passives: 2-pin components in anchor layers are reorderable
    reorderable_refs = {target_ref}
    for layer in anchor_layers:
        for comp in layer:
            if len(comp.pin_order) == 2:
                reorderable_refs.add(comp.ref)

    target_column = PinColumn(ref=target_ref, pin_order=list(signal_pins))
    all_layers = list(anchor_layers) + [[target_column]]

    # Run sweep
    report = sweep_optimize(all_layers, reorderable_refs, filtered_nets)

    # Extract optimized signal pin order from sweep result
    optimized_signal_order = report.optimized_orders.get(target_ref, signal_pins)

    # Collect passive reorderings
    passive_reorderings: dict[str, list[str]] = {}
    for ref, order in report.optimized_orders.items():
        if ref != target_ref and order != report.original_orders.get(ref):
            passive_reorderings[ref] = order

    # Build final pin map
    pin_map = build_pin_map(
        target_ref=target_ref,
        all_pins=target_pins,
        locks=locks,
        optimized_signal_pins=optimized_signal_order,
        pin_to_net=pin_to_net,
        nets=filtered_nets,
        unmatched_mode=unmatched,
    )

    return FootprintPlan(
        target_ref=target_ref,
        pin_map=pin_map,
        crossings_before=report.total_crossings,
        crossings_after=report.total_crossings_after,
        iterations=report.iterations,
        passive_reorderings=passive_reorderings,
    )


def format_footprint_plan(plan: FootprintPlan) -> str:
    """Format a FootprintPlan as human-readable text."""
    lines: list[str] = []
    total = len(plan.pin_map)
    lines.append(f"Footprint Pin Map Proposal for {plan.target_ref} ({total} positions)")
    lines.append("=" * 60)
    lines.append("")

    # Group by status
    locked = [a for a in plan.pin_map if a.status == "locked"]
    optimized = [a for a in plan.pin_map if a.status == "optimized"]
    unmatched = [a for a in plan.pin_map if a.status == "unmatched"]

    if locked:
        lines.append("Locked pins:")
        for a in locked:
            net_str = a.net if a.net else "NC"
            lines.append(f"  {a.pin:>3}: {net_str}")
        lines.append("")

    if optimized:
        lines.append("Optimized signal assignment:")
        for a in optimized:
            net_str = a.net if a.net else "NC"
            route = f"  (routes to {a.routes_to})" if a.routes_to else ""
            lines.append(f"  {a.pin:>3}: {net_str:<20}{route}")
        lines.append("")

    if unmatched:
        lines.append("Unmatched pins:")
        for a in unmatched:
            net_str = a.net if a.net else "NC"
            lines.append(f"  {a.pin:>3}: {net_str}")
        lines.append("")

    if plan.passive_reorderings:
        lines.append("Passive reorderings:")
        for ref in sorted(plan.passive_reorderings):
            order = ", ".join(plan.passive_reorderings[ref])
            lines.append(f"  {ref}: [{order}]")
        lines.append("")

    lines.append(
        f"Crossings: {plan.crossings_before} before -> "
        f"{plan.crossings_after} after ({plan.iterations} iterations)"
    )

    if plan.crossings_after > 0:
        lines.append("")
        lines.append(
            "WARNING: Not all crossings can be eliminated by reordering alone."
        )
        lines.append(
            "Remaining crossings will require vias or a second routing layer."
        )

    return "\n".join(lines)


def plan_to_dict(plan: FootprintPlan) -> dict:
    """Convert a FootprintPlan to a JSON-serializable dict."""
    pin_map = []
    for a in plan.pin_map:
        entry: dict = {
            "pin": a.pin,
            "net": a.net,
            "status": a.status,
        }
        if a.routes_to:
            entry["routes_to"] = a.routes_to
        pin_map.append(entry)

    return {
        "target": plan.target_ref,
        "total_pins": len(plan.pin_map),
        "crossings_before": plan.crossings_before,
        "crossings_after": plan.crossings_after,
        "iterations": plan.iterations,
        "pin_map": pin_map,
        "passive_reorderings": plan.passive_reorderings,
    }


# =========================================================================
# CLI entry point
# =========================================================================

def _cmd_analyze(args):
    """Handle the 'analyze' subcommand (original behavior)."""
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

    verbose = not args.quiet and not args.json_output

    if exclude_set and verbose:
        excluded_str = ", ".join(f"{r}:{p}" for r, p in sorted(exclude_set))
        print(f"(Excluding pins: {excluded_str})")
        print()

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
    if not reorderable_refs and verbose:
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
    if verbose:
        print("Layer assignment:")
        for i, layer in enumerate(layers):
            refs = ", ".join(c.ref for c in layer)
            pins = sum(len(c.pin_order) for c in layer)
            print(f"  Layer {i}: [{refs}] ({pins} pins)")
        print()

    # Run sweep
    report = sweep_optimize(layers, reorderable_refs, data["nets"])

    if args.json_output:
        print(json.dumps(report_to_dict(report, data["nets"]), indent=2))
    elif not args.quiet:
        print(format_multilayer_report(report, data["nets"]))

    sys.exit(0 if report.total_crossings_after == 0 else 1)


def _cmd_plan_footprint(args):
    """Handle the 'plan-footprint' subcommand."""
    if not Path(args.netlist).exists():
        print(f"Error: file not found: {args.netlist}")
        sys.exit(1)

    data = parse_netlist(args.netlist)

    # Validate target component
    if args.target not in data["components"]:
        print(f"Error: target component '{args.target}' not found in netlist.")
        print(f"Available: {', '.join(sorted(data['components'].keys()))}")
        sys.exit(1)

    # Parse --anchors: "J2,U1 | R1,R2,C1"
    anchor_layers: list[list[PinColumn]] = []
    for group_str in args.anchors.split("|"):
        layer: list[PinColumn] = []
        for ref in group_str.split(","):
            ref = ref.strip()
            if not ref:
                continue
            if ref not in data["components"]:
                print(f"Error: anchor component '{ref}' not found in netlist.")
                print(f"Available: {', '.join(sorted(data['components'].keys()))}")
                sys.exit(1)
            pins = infer_pin_order(ref, data["nets"])
            layer.append(PinColumn(ref=ref, pin_order=pins))
        if layer:
            anchor_layers.append(layer)

    # Parse locks
    try:
        locks = parse_pin_locks(args.lock or [])
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Infer target pin count from component or default to max pin in netlist
    target_pins = infer_pin_order(args.target, data["nets"])
    # Include locked pins that may not appear in the netlist (e.g. NC pins)
    for pin in locks:
        if pin not in target_pins:
            target_pins.append(pin)
    # Re-sort
    def _sort_key(p: str):
        try:
            return (0, int(p))
        except ValueError:
            return (1, p)
    target_pins.sort(key=_sort_key)

    exclude_nets = set(args.exclude_nets or [])
    verbose = not args.quiet and not args.json_output

    if verbose:
        print(f"Planning footprint for {args.target} ({len(target_pins)} pins)")
        print(f"Anchor layers: {args.anchors}")
        if locks:
            locked_str = ", ".join(
                f"{p}={'NC' if n is None else n}"
                for p, n in sorted(locks.items(), key=lambda x: _sort_key(x[0]))
            )
            print(f"Locked pins: {locked_str}")
        if exclude_nets:
            print(f"Excluded nets: {', '.join(sorted(exclude_nets))}")
        print()

    plan = plan_footprint(
        target_ref=args.target,
        target_pins=target_pins,
        anchor_layers=anchor_layers,
        nets=data["nets"],
        locks=locks,
        unmatched=args.unmatched,
        exclude_nets=exclude_nets,
    )

    if args.json_output:
        print(json.dumps(plan_to_dict(plan), indent=2))
    elif not args.quiet:
        print(format_footprint_plan(plan))

    sys.exit(0 if plan.crossings_after == 0 else 1)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze trace crossings in a KiCad netlist.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- analyze subcommand (default) ---
    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze crossings between component layers.",
        epilog=(
            "Example: pcb-crossing-optimizer analyze net.net "
            "--layers 'J1 | R1,C1 | J2' --reorderable J2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    analyze.add_argument("netlist", help="Path to a KiCad .net file generated by SKiDL")
    analyze.add_argument(
        "--layers", type=str, required=True,
        help=(
            "Layer specification: components per layer separated by |. "
            "Multiple components in a layer separated by commas. "
            "Example: 'J1 | R1,R2,C1 | J2'"
        ),
    )
    analyze.add_argument(
        "--reorderable", nargs="*", default=[], metavar="REF",
        help="Components whose pin order can be changed.",
    )
    analyze.add_argument(
        "--exclude", nargs="*", default=[], metavar="REF:PIN",
        help="Exclude pins from analysis (e.g. J1:SH).",
    )
    analyze.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output results as JSON instead of human-readable text.",
    )
    analyze.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress all output; exit code 0 = no crossings, 1 = crossings remain.",
    )
    analyze.set_defaults(func=_cmd_analyze)

    # --- plan-footprint subcommand ---
    plan = subparsers.add_parser(
        "plan-footprint",
        help="Compute an optimal pin map for a custom footprint.",
        epilog=(
            "Example: pcb-crossing-optimizer plan-footprint net.net "
            "--target J1 --anchors 'J2,U1 | R1,R2,C1' "
            "--lock 1=NC 2=NC 3=GND_EARLY_A"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plan.add_argument("netlist", help="Path to a KiCad .net file generated by SKiDL")
    plan.add_argument(
        "--target", required=True, metavar="REF",
        help="Component ref whose footprint is being designed.",
    )
    plan.add_argument(
        "--anchors", required=True,
        help=(
            "Fixed components organized in layers separated by |. "
            "Example: 'J2,U1 | R1,R2,R3,C1'"
        ),
    )
    plan.add_argument(
        "--lock", nargs="*", default=[], metavar="PIN=NET",
        help="Lock pins to specific nets. Use PIN=NC for no-connect. Example: 1=NC 3=GND_EARLY_A",
    )
    plan.add_argument(
        "--unmatched", choices=["start", "end", "split"], default="end",
        help="Where to place unmatched pins: start, end (default), or split.",
    )
    plan.add_argument(
        "--exclude-nets", nargs="*", default=[], metavar="NET",
        help="Net names to exclude from analysis (e.g. GND).",
    )
    plan.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output results as JSON instead of human-readable text.",
    )
    plan.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress all output; exit code 0 = no crossings, 1 = crossings remain.",
    )
    plan.set_defaults(func=_cmd_plan_footprint)

    # Backward compatibility: if first arg is not a known subcommand,
    # assume "analyze" mode (legacy CLI: pcb-crossing-optimizer file.net --layers ...)
    known_commands = {"analyze", "plan-footprint", "-h", "--help"}
    argv = sys.argv[1:]
    if argv and argv[0] not in known_commands:
        argv = ["analyze"] + argv

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
