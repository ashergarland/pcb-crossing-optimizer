"""PCB Crossing Optimizer — crossing minimization and placement for PCB design.

Core modules:
- crossing_analyzer: Pin ordering optimization (Sugiyama sweep)
- footprint_parser: KiCad .kicad_mod geometry extraction
- board_constraints: Physical board constraint model
- placement_engine: Force-directed + annealing placement
- power_validator: IPC-2152 trace width validation
"""

from ._core import (
    FootprintPlan,
    LayerEdge,
    LayerPairCrossing,
    LayerPairReport,
    MultilayerReport,
    PinAssignment,
    PinColumn,
    count_layer_pair_crossings,
    extract_layer_pair_edges,
    format_footprint_plan,
    format_multilayer_report,
    infer_pin_order,
    layer_global_positions,
    main,
    parse_netlist,
    parse_pin_locks,
    plan_footprint,
    plan_to_dict,
    report_to_dict,
    sweep_optimize,
)
from .footprint_parser import (
    FootprintGeometry,
    PadGeometry,
    detect_footprint_dir,
    get_footprint_geometry,
    parse_footprint_geometry,
    resolve_footprint_geometry,
)
from .board_constraints import (
    BoardConstraints,
    FixedPosition,
    KeepoutZone,
    MountingHole,
    infer_board_size,
    parse_board_constraints,
)
from .placement_engine import (
    DecouplingIssue,
    PlacementMetrics,
    PlacementResult,
    place_components,
    placement_to_dict,
)
from .power_validator import (
    PowerViolation,
    StackupParams,
    power_validation_to_dict,
    trace_width_ipc2152,
    validate_power_traces,
)
