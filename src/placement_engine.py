"""PCB component placement engine.

Hybrid force-directed + simulated annealing algorithm for computing
optimal component positions on a PCB, given a netlist, footprint geometries,
and board constraints.

Output is EDA-agnostic JSON: {ref: {x, y, rotation, layer}}.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .board_constraints import (
    BoardConstraints,
    FixedPosition,
    KeepoutZone,
    MountingHole,
    infer_board_size,
)
from .footprint_parser import FootprintGeometry


# =========================================================================
# Data model
# =========================================================================

@dataclass
class PlacementComponent:
    """A component being placed on the board."""
    ref: str
    geometry: FootprintGeometry
    x: float = 0.0  # mm, center of component
    y: float = 0.0
    rotation: float = 0.0  # degrees (0, 90, 180, 270)
    layer: str = "F.Cu"
    fixed: bool = False
    group: int = -1  # functional group ID


@dataclass
class PlacementMetrics:
    """Quality metrics for a placement result."""
    total_wire_length_mm: float  # half-perimeter wire length (HPWL)
    overlap_count: int  # number of overlapping component pairs
    overlap_area_mm2: float  # total overlap area
    out_of_bounds_count: int  # components outside board
    keepout_violations: int
    component_clearance_violations: int


@dataclass
class DecouplingIssue:
    """A decoupling capacitor placed too far from its target IC."""
    cap_ref: str
    ic_ref: str
    distance_mm: float
    recommended_max_mm: float = 2.0


@dataclass
class PlacementResult:
    """Complete output of the placement engine."""
    positions: dict[str, dict]  # {ref: {"x": float, "y": float, "rotation": float, "layer": str}}
    metrics: PlacementMetrics
    board_width_mm: float
    board_height_mm: float
    decoupling_issues: list[DecouplingIssue] = field(default_factory=list)
    groups: dict[str, int] = field(default_factory=dict)  # {ref: group_id}


# =========================================================================
# Net connectivity helpers
# =========================================================================

def _build_component_connectivity(
    nets: dict[str, list[tuple[str, str]]],
    refs: set[str],
) -> dict[tuple[str, str], int]:
    """Build edge weights between component pairs based on shared nets.

    Returns {(ref_a, ref_b): weight} where weight = number of shared net connections.
    """
    connectivity: dict[tuple[str, str], int] = {}
    for net_name, nodes in nets.items():
        net_refs = [ref for ref, pin in nodes if ref in refs]
        # Count pairwise connections
        for i in range(len(net_refs)):
            for j in range(i + 1, len(net_refs)):
                a, b = sorted((net_refs[i], net_refs[j]))
                connectivity[(a, b)] = connectivity.get((a, b), 0) + 1
    return connectivity


def _compute_hpwl(
    components: dict[str, PlacementComponent],
    nets: dict[str, list[tuple[str, str]]],
) -> float:
    """Compute half-perimeter wire length (HPWL) for all nets.

    HPWL = sum over all nets of (max_x - min_x + max_y - min_y).
    """
    total = 0.0
    for net_name, nodes in nets.items():
        xs = []
        ys = []
        for ref, pin in nodes:
            comp = components.get(ref)
            if comp:
                xs.append(comp.x)
                ys.append(comp.y)
        if len(xs) >= 2:
            total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


# =========================================================================
# Overlap / DRC detection
# =========================================================================

def _get_bbox(comp: PlacementComponent) -> tuple[float, float, float, float]:
    """Get axis-aligned bounding box of a component considering rotation."""
    w = comp.geometry.width_mm
    h = comp.geometry.height_mm
    if comp.rotation in (90, 270):
        w, h = h, w
    return (
        comp.x - w / 2,
        comp.y - h / 2,
        comp.x + w / 2,
        comp.y + h / 2,
    )


def _bboxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    clearance: float = 0.0,
) -> bool:
    """Check if two axis-aligned bounding boxes overlap (with clearance)."""
    return not (
        a[2] + clearance <= b[0] or
        b[2] + clearance <= a[0] or
        a[3] + clearance <= b[1] or
        b[3] + clearance <= a[1]
    )


def _overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute overlap area between two bounding boxes."""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if dx > 0 and dy > 0:
        return dx * dy
    return 0.0


def _count_overlaps(
    components: dict[str, PlacementComponent],
    clearance: float = 0.0,
) -> tuple[int, float]:
    """Count overlapping pairs and total overlap area."""
    refs = list(components.keys())
    count = 0
    total_area = 0.0
    for i in range(len(refs)):
        bbox_a = _get_bbox(components[refs[i]])
        for j in range(i + 1, len(refs)):
            bbox_b = _get_bbox(components[refs[j]])
            if _bboxes_overlap(bbox_a, bbox_b, clearance):
                count += 1
                total_area += _overlap_area(bbox_a, bbox_b)
    return count, total_area


def _count_out_of_bounds(
    components: dict[str, PlacementComponent],
    board_w: float,
    board_h: float,
    edge_clearance: float,
) -> int:
    """Count components that extend beyond board boundaries."""
    count = 0
    for comp in components.values():
        bbox = _get_bbox(comp)
        if (bbox[0] < edge_clearance or bbox[1] < edge_clearance or
                bbox[2] > board_w - edge_clearance or bbox[3] > board_h - edge_clearance):
            count += 1
    return count


def _count_keepout_violations(
    components: dict[str, PlacementComponent],
    keepouts: list[KeepoutZone],
) -> int:
    """Count components that overlap with keepout zones."""
    count = 0
    for comp in components.values():
        if comp.fixed:
            continue
        bbox = _get_bbox(comp)
        for kz in keepouts:
            kz_bbox = (kz.x, kz.y, kz.x + kz.width, kz.y + kz.height)
            if _bboxes_overlap(bbox, kz_bbox):
                count += 1
                break
    return count


# =========================================================================
# Functional grouping
# =========================================================================

def _cluster_components(
    connectivity: dict[tuple[str, str], int],
    all_refs: set[str],
    threshold: int = 2,
) -> dict[str, int]:
    """Simple greedy clustering: components with >= threshold shared nets are grouped.

    Returns {ref: group_id}.
    """
    groups: dict[str, int] = {}
    group_id = 0

    # Sort edges by weight descending
    edges = sorted(connectivity.items(), key=lambda x: x[1], reverse=True)

    for (a, b), weight in edges:
        if weight < threshold:
            break
        ga = groups.get(a)
        gb = groups.get(b)
        if ga is None and gb is None:
            groups[a] = group_id
            groups[b] = group_id
            group_id += 1
        elif ga is not None and gb is None:
            groups[b] = ga
        elif gb is not None and ga is None:
            groups[a] = gb
        # If both already grouped, leave them (don't merge groups in simple version)

    # Assign ungrouped components to their own group
    for ref in all_refs:
        if ref not in groups:
            groups[ref] = group_id
            group_id += 1

    return groups


# =========================================================================
# Force-directed placement
# =========================================================================

def _force_directed_placement(
    components: dict[str, PlacementComponent],
    connectivity: dict[tuple[str, str], int],
    board_w: float,
    board_h: float,
    edge_clearance: float,
    max_iterations: int = 200,
    convergence_threshold: float = 0.01,
) -> None:
    """Apply force-directed placement to position components.

    Modifies component positions in-place.
    """
    # Initial random placement for non-fixed components
    rng = random.Random(42)
    for comp in components.values():
        if not comp.fixed:
            comp.x = rng.uniform(edge_clearance + comp.geometry.width_mm / 2,
                                 board_w - edge_clearance - comp.geometry.width_mm / 2)
            comp.y = rng.uniform(edge_clearance + comp.geometry.height_mm / 2,
                                 board_h - edge_clearance - comp.geometry.height_mm / 2)

    refs = [r for r, c in components.items() if not c.fixed]
    if not refs:
        return

    # Adaptive step size
    step = min(board_w, board_h) * 0.1

    for iteration in range(max_iterations):
        max_displacement = 0.0

        for ref in refs:
            comp = components[ref]
            fx, fy = 0.0, 0.0

            # Attractive forces (net connections)
            for (a, b), weight in connectivity.items():
                other_ref = None
                if a == ref:
                    other_ref = b
                elif b == ref:
                    other_ref = a
                if other_ref and other_ref in components:
                    other = components[other_ref]
                    dx = other.x - comp.x
                    dy = other.y - comp.y
                    dist = math.sqrt(dx * dx + dy * dy) + 0.001
                    # Attractive force proportional to distance and weight
                    force = weight * dist * 0.01
                    fx += force * dx / dist
                    fy += force * dy / dist

            # Repulsive forces (other components)
            for other_ref, other in components.items():
                if other_ref == ref:
                    continue
                dx = comp.x - other.x
                dy = comp.y - other.y
                dist = math.sqrt(dx * dx + dy * dy) + 0.001
                # Repulsive force inversely proportional to distance squared
                # Scaled by combined component area
                combined_area = (comp.geometry.width_mm * comp.geometry.height_mm +
                                 other.geometry.width_mm * other.geometry.height_mm)
                force = combined_area * 2.0 / (dist * dist)
                fx += force * dx / dist
                fy += force * dy / dist

            # Board boundary forces (push away from edges)
            margin = edge_clearance + max(comp.geometry.width_mm, comp.geometry.height_mm) / 2
            if comp.x < margin:
                fx += (margin - comp.x) * 0.5
            if comp.x > board_w - margin:
                fx -= (comp.x - (board_w - margin)) * 0.5
            if comp.y < margin:
                fy += (margin - comp.y) * 0.5
            if comp.y > board_h - margin:
                fy -= (comp.y - (board_h - margin)) * 0.5

            # Apply force with step size limit
            displacement = math.sqrt(fx * fx + fy * fy)
            if displacement > step:
                fx = fx / displacement * step
                fy = fy / displacement * step
                displacement = step

            comp.x += fx
            comp.y += fy
            max_displacement = max(max_displacement, displacement)

            # Clamp to board
            hw = comp.geometry.width_mm / 2
            hh = comp.geometry.height_mm / 2
            comp.x = max(edge_clearance + hw, min(board_w - edge_clearance - hw, comp.x))
            comp.y = max(edge_clearance + hh, min(board_h - edge_clearance - hh, comp.y))

        # Reduce step size (cooling)
        step *= 0.95

        # Check convergence
        if max_displacement < convergence_threshold:
            break


# =========================================================================
# Simulated annealing refinement
# =========================================================================

def _annealing_cost(
    components: dict[str, PlacementComponent],
    nets: dict[str, list[tuple[str, str]]],
    board_w: float,
    board_h: float,
    constraints: BoardConstraints,
    cap_ic_pairs: list[tuple[str, str]],
) -> float:
    """Compute the cost function for simulated annealing.

    Weighted sum of:
    - HPWL wire length
    - Overlap penalty (heavy)
    - Out-of-bounds penalty (heavy)
    - Keepout violations (heavy)
    - Decoupling cap distance (moderate)
    """
    hpwl = _compute_hpwl(components, nets)
    overlaps, overlap_area = _count_overlaps(components, constraints.component_clearance_mm)
    oob = _count_out_of_bounds(components, board_w, board_h, constraints.edge_clearance_mm)
    keepout_v = _count_keepout_violations(components, constraints.keepout_zones)

    # Decoupling distance cost
    decoupling_cost = 0.0
    for cap_ref, ic_ref in cap_ic_pairs:
        cap = components.get(cap_ref)
        ic = components.get(ic_ref)
        if cap and ic:
            dist = math.sqrt((cap.x - ic.x) ** 2 + (cap.y - ic.y) ** 2)
            if dist > 2.0:  # penalize if > 2mm
                decoupling_cost += (dist - 2.0) ** 2

    return (
        hpwl * 1.0 +
        overlap_area * 1000.0 +
        overlaps * 500.0 +
        oob * 2000.0 +
        keepout_v * 2000.0 +
        decoupling_cost * 50.0
    )


def _simulated_annealing(
    components: dict[str, PlacementComponent],
    nets: dict[str, list[tuple[str, str]]],
    board_w: float,
    board_h: float,
    constraints: BoardConstraints,
    cap_ic_pairs: list[tuple[str, str]],
    max_iterations: int = 10000,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    seed: int = 42,
) -> None:
    """Refine placement using simulated annealing.

    Moves: translate, rotate 90°, swap two similarly-sized components.
    """
    rng = random.Random(seed)
    movable_refs = [r for r, c in components.items() if not c.fixed]
    if not movable_refs:
        return

    current_cost = _annealing_cost(components, nets, board_w, board_h, constraints, cap_ic_pairs)
    temp = initial_temp
    best_cost = current_cost

    # Save best positions
    best_positions: dict[str, tuple[float, float, float]] = {
        ref: (c.x, c.y, c.rotation) for ref, c in components.items()
    }

    for iteration in range(max_iterations):
        # Choose a random move
        move_type = rng.random()
        ref = rng.choice(movable_refs)
        comp = components[ref]

        # Save state
        old_x, old_y, old_rot = comp.x, comp.y, comp.rotation

        if move_type < 0.6:
            # Translate
            max_step = min(board_w, board_h) * 0.1 * (temp / initial_temp)
            comp.x += rng.uniform(-max_step, max_step)
            comp.y += rng.uniform(-max_step, max_step)
            # Clamp
            hw = comp.geometry.width_mm / 2
            hh = comp.geometry.height_mm / 2
            comp.x = max(hw, min(board_w - hw, comp.x))
            comp.y = max(hh, min(board_h - hh, comp.y))

        elif move_type < 0.8:
            # Rotate 90°
            comp.rotation = (comp.rotation + 90) % 360

        else:
            # Swap with another component of similar size
            other_ref = rng.choice(movable_refs)
            if other_ref != ref:
                other = components[other_ref]
                comp.x, other.x = other.x, comp.x
                comp.y, other.y = other.y, comp.y

        # Evaluate
        new_cost = _annealing_cost(components, nets, board_w, board_h, constraints, cap_ic_pairs)
        delta = new_cost - current_cost

        # Accept or reject
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 0.001)):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = {
                    ref: (c.x, c.y, c.rotation) for ref, c in components.items()
                }
        else:
            # Revert
            if move_type < 0.8:
                comp.x, comp.y, comp.rotation = old_x, old_y, old_rot
            else:
                # Revert swap
                if other_ref != ref:
                    other = components[other_ref]
                    comp.x, other.x = other.x, comp.x
                    comp.y, other.y = other.y, comp.y

        temp *= cooling_rate

    # Restore best positions
    for ref, (x, y, rot) in best_positions.items():
        if ref in components:
            components[ref].x = x
            components[ref].y = y
            components[ref].rotation = rot


# =========================================================================
# Decoupling cap identification
# =========================================================================

def _identify_cap_ic_pairs(
    nets: dict[str, list[tuple[str, str]]],
    component_info: dict[str, dict],
) -> list[tuple[str, str]]:
    """Identify capacitor-to-IC pairs for decoupling placement.

    Heuristic: a capacitor (ref starts with C) that shares a power net
    with an IC (ref starts with U) is likely a decoupling cap.

    Returns list of (cap_ref, ic_ref) pairs.
    """
    caps = {ref for ref in component_info if ref.startswith("C")}
    ics = {ref for ref in component_info if ref.startswith("U")}

    if not caps or not ics:
        return []

    # Find power-like nets (heuristic: name contains VCC, VDD, 3V3, 5V, etc.)
    power_net_pattern = {"VCC", "VDD", "3V3", "5V", "3.3V", "1.8V", "VBUS", "VBAT"}

    pairs: list[tuple[str, str]] = []
    for net_name, nodes in nets.items():
        # Check if this is a power net
        is_power = any(p in net_name.upper() for p in power_net_pattern)
        if not is_power:
            continue

        net_caps = [ref for ref, pin in nodes if ref in caps]
        net_ics = [ref for ref, pin in nodes if ref in ics]

        for cap in net_caps:
            for ic in net_ics:
                if (cap, ic) not in pairs:
                    pairs.append((cap, ic))

    return pairs


def _evaluate_decoupling(
    components: dict[str, PlacementComponent],
    cap_ic_pairs: list[tuple[str, str]],
    max_distance_mm: float = 2.0,
) -> list[DecouplingIssue]:
    """Check decoupling cap placement quality."""
    issues = []
    for cap_ref, ic_ref in cap_ic_pairs:
        cap = components.get(cap_ref)
        ic = components.get(ic_ref)
        if cap and ic:
            dist = math.sqrt((cap.x - ic.x) ** 2 + (cap.y - ic.y) ** 2)
            if dist > max_distance_mm:
                issues.append(DecouplingIssue(
                    cap_ref=cap_ref,
                    ic_ref=ic_ref,
                    distance_mm=round(dist, 2),
                    recommended_max_mm=max_distance_mm,
                ))
    return issues


# =========================================================================
# Main placement function
# =========================================================================

def place_components(
    nets: dict[str, list[tuple[str, str]]],
    component_info: dict[str, dict],
    geometries: dict[str, FootprintGeometry],
    constraints: BoardConstraints,
    annealing_iterations: int = 10000,
    seed: int = 42,
) -> PlacementResult:
    """Compute optimal component placement.

    Args:
        nets: Parsed netlist connectivity {net_name: [(ref, pin), ...]}.
        component_info: Component metadata {ref: {"value": str, "part": str, "footprint": str}}.
        geometries: Footprint geometry per component {ref: FootprintGeometry}.
        constraints: Board physical constraints.
        annealing_iterations: Number of simulated annealing iterations.
        seed: Random seed for reproducibility.

    Returns:
        PlacementResult with positions and quality metrics.
    """
    # Determine board size
    board_w, board_h = infer_board_size(geometries, constraints)

    # Build component objects
    all_refs = set(geometries.keys())
    connectivity = _build_component_connectivity(nets, all_refs)
    groups = _cluster_components(connectivity, all_refs)

    # Fixed position lookup
    fixed_lookup = {fp.ref: fp for fp in constraints.fixed_positions}

    components: dict[str, PlacementComponent] = {}
    for ref, geom in geometries.items():
        fixed_pos = fixed_lookup.get(ref)
        comp = PlacementComponent(
            ref=ref,
            geometry=geom,
            group=groups.get(ref, -1),
        )
        if fixed_pos:
            comp.x = fixed_pos.x
            comp.y = fixed_pos.y
            comp.rotation = fixed_pos.rotation
            comp.layer = fixed_pos.layer
            comp.fixed = True
        components[ref] = comp

    # Identify decoupling pairs
    cap_ic_pairs = _identify_cap_ic_pairs(nets, component_info)

    # Phase 1: Force-directed initial placement
    _force_directed_placement(
        components, connectivity, board_w, board_h,
        constraints.edge_clearance_mm,
    )

    # Phase 2: Simulated annealing refinement
    _simulated_annealing(
        components, nets, board_w, board_h, constraints,
        cap_ic_pairs,
        max_iterations=annealing_iterations,
        seed=seed,
    )

    # Evaluate final metrics
    hpwl = _compute_hpwl(components, nets)
    overlaps, overlap_area = _count_overlaps(components, constraints.component_clearance_mm)
    oob = _count_out_of_bounds(components, board_w, board_h, constraints.edge_clearance_mm)
    keepout_v = _count_keepout_violations(components, constraints.keepout_zones)
    clearance_v, _ = _count_overlaps(components, constraints.component_clearance_mm)
    decoupling_issues = _evaluate_decoupling(components, cap_ic_pairs)

    metrics = PlacementMetrics(
        total_wire_length_mm=round(hpwl, 2),
        overlap_count=overlaps,
        overlap_area_mm2=round(overlap_area, 3),
        out_of_bounds_count=oob,
        keepout_violations=keepout_v,
        component_clearance_violations=clearance_v,
    )

    # Build output positions
    positions = {}
    for ref, comp in components.items():
        positions[ref] = {
            "x": round(comp.x, 3),
            "y": round(comp.y, 3),
            "rotation": comp.rotation,
            "layer": comp.layer,
        }

    return PlacementResult(
        positions=positions,
        metrics=metrics,
        board_width_mm=round(board_w, 1),
        board_height_mm=round(board_h, 1),
        decoupling_issues=decoupling_issues,
        groups={ref: comp.group for ref, comp in components.items()},
    )


# =========================================================================
# Serialization
# =========================================================================

def placement_to_dict(result: PlacementResult) -> dict:
    """Convert PlacementResult to a JSON-serializable dict."""
    return {
        "board": {
            "width_mm": result.board_width_mm,
            "height_mm": result.board_height_mm,
        },
        "positions": result.positions,
        "metrics": {
            "total_wire_length_mm": result.metrics.total_wire_length_mm,
            "overlap_count": result.metrics.overlap_count,
            "overlap_area_mm2": result.metrics.overlap_area_mm2,
            "out_of_bounds_count": result.metrics.out_of_bounds_count,
            "keepout_violations": result.metrics.keepout_violations,
            "component_clearance_violations": result.metrics.component_clearance_violations,
        },
        "decoupling_issues": [
            {
                "cap_ref": d.cap_ref,
                "ic_ref": d.ic_ref,
                "distance_mm": d.distance_mm,
                "recommended_max_mm": d.recommended_max_mm,
            }
            for d in result.decoupling_issues
        ],
        "groups": result.groups,
    }
