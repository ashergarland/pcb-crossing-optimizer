# PCB Crossing Optimizer

**The algorithms behind AI-native PCB design.** This library powers the optimization pipeline in [skidl-vscode](https://github.com/ashergarland/skidl-vscode) — turning netlists into placement-optimized, crossing-free board layouts that any AI agent can produce.

```
Netlist (.net) → Pin ordering → Component placement → Power validation → JSON output
                 (zero crossings)  (zero overlaps)     (IPC-2152)        (any EDA tool)
```

When an AI agent uses the skidl-vscode MCP server to design a PCB, this library does the heavy math:

- **Crossing analysis** — Sugiyama-style barycenter sweep eliminates trace crossings by computing optimal pin orderings
- **Footprint planning** — Pin-to-net assignment for custom footprints, respecting locks and anchor layers
- **Component placement** — Hybrid force-directed + simulated annealing computes positions that minimize wire length, avoid overlaps, and keep decoupling caps near their ICs
- **Power trace validation** — IPC-2152 calculations flag nets where trace width may be insufficient for the current load
- **EDA-agnostic output** — All results are JSON: `{ref: {x, y, rotation, layer}}` — works with KiCad, Altium, EasyEDA, or any tool

## Installation

```
pip install pcb-crossing-optimizer
```

After installation, the `pcb-crossing-optimizer` command is available on your PATH.

## How to use this

**Most users** should install the [skidl-vscode extension](https://github.com/ashergarland/skidl-vscode) — it provides the full end-to-end experience (schematic validation + crossing optimization + placement + power validation) via MCP tools that AI agents call automatically.

**Library developers** and **CLI users** can use this package directly for crossing analysis, placement, and power validation on KiCad netlists.

## Usage

The tool has three subcommands: `analyze` (default), `plan-footprint`, and `place`.

### analyze (default)

Analyze crossings between component layers and compute optimal pin orderings for single-layer routing.

```
pcb-crossing-optimizer [analyze] <netlist.net> --layers "L0_refs | L1_refs | L2_refs" --reorderable REF [REF ...] [--exclude REF:PIN ...] [--json] [--quiet]
```

Examples:
```bash
# Analyze the microSD breakout with passives in the middle layer
pcb-crossing-optimizer examples/microsd_breakout.net --layers "J1 | R1,R2,C1 | J2" --reorderable J2 --exclude J1:SH

# Simple two-connector case
pcb-crossing-optimizer examples/i2c_breakout.net --layers "J1 | J2" --reorderable J2

# JSON output for tooling integration
pcb-crossing-optimizer examples/i2c_breakout.net --layers "J1 | J2" --reorderable J2 --json

# Quiet mode for CI (exit code 0 = no crossings, 1 = crossings remain)
pcb-crossing-optimizer examples/i2c_breakout.net --layers "J1 | J2" --reorderable J2 --quiet
```

### plan-footprint

Compute an optimal pin map for a custom footprint. Given a target component (whose footprint you are designing), fixed anchor components, and optional pin locks, it runs the crossing sweep and produces a complete pin-to-net assignment proposal.

```
pcb-crossing-optimizer plan-footprint <netlist.net> \
    --target J1 \
    --anchors "J2,U1 | R1,R2,R3,C1" \
    --lock 1=NC 2=NC 3=GND_EARLY_A \
    --unmatched end \
    --exclude-nets GND \
    [--json] [--quiet]
```

Options:
- `--target REF`: Component whose footprint is being designed
- `--anchors "..."`: Fixed components organized in layers separated by `|`
- `--lock PIN=NET`: Lock specific pins to nets (use `PIN=NC` for no-connect)
- `--unmatched start|end|split`: Where to place pins with no anchor connections (default: `end`)
- `--exclude-nets NET [NET ...]`: Nets to exclude from analysis (e.g. ground pours)

### place

Compute optimal component positions on a PCB. Uses force-directed initial placement followed by simulated annealing refinement. Output is EDA-agnostic JSON that can be applied to KiCad, Altium, EasyEDA, or any PCB tool.

```
pcb-crossing-optimizer place <netlist.net> \
    [--board WxH] \
    [--fixed REF:X,Y,ROT ...] \
    [--current NET:AMPS ...] \
    [--iterations N]
```

Examples:
```bash
# Auto-place with automatic board sizing
pcb-crossing-optimizer place examples/i2c_breakout.net

# Specify board dimensions and fix connector positions
pcb-crossing-optimizer place examples/i2c_breakout.net --board 35x25 --fixed "J1:2,12,0" "J2:30,12,0"

# Include power trace validation
pcb-crossing-optimizer place examples/microsd_breakout.net --board 40x30 --current "VCC:0.5" "GND:1.0"
```

Output format:
```json
{
  "board": {"width_mm": 35.0, "height_mm": 25.0},
  "positions": {
    "J1": {"x": 2.0, "y": 12.0, "rotation": 0.0, "layer": "F.Cu"},
    "R1": {"x": 15.3, "y": 8.7, "rotation": 90.0, "layer": "F.Cu"},
    ...
  },
  "metrics": {
    "total_wire_length_mm": 42.5,
    "overlap_count": 0,
    "out_of_bounds_count": 0
  },
  "decoupling_issues": [],
  "power_violations": [
    {"net": "GND", "current_a": 1.0, "required_width_mm": 0.56, "severity": "warning"}
  ]
}
```

## How it works

### Crossing analysis
1. Parses a KiCad .net file (S-expression format generated by SKiDL)
2. Extracts pin-to-net connectivity for specified components
3. Sugiyama-style barycenter sweep across all layers with virtual node insertion for long edges
4. Reports crossing counts before/after, recommended reorderings, and remaining crossings

### Component placement
1. Parses footprint geometries from KiCad `.kicad_mod` files (courtyard bounding boxes)
2. Clusters components by net connectivity (functional grouping)
3. **Force-directed initial placement**: spring model with net-weighted attractive forces and courtyard-scaled repulsive forces
4. **Simulated annealing refinement**: translate, rotate, swap moves optimizing HPWL wire length + overlap penalty + decoupling proximity
5. Outputs EDA-agnostic JSON placement directives

### Power validation
- IPC-2152 simplified model: calculates minimum trace width from current, copper weight, and temperature rise
- Flags nets where required trace width exceeds standard design rules

## Project structure

```
src/
├── __init__.py              Package init with backward-compatible exports
├── _core.py                 Crossing analysis, pin ordering, CLI entry point
├── footprint_parser.py      Parse .kicad_mod for courtyard/pad geometry
├── board_constraints.py     Board outline, fixed positions, keepout zones
├── placement_engine.py      Force-directed + simulated annealing placement
└── power_validator.py       IPC-2152 trace width calculations
tests/
├── test_crossing_analyzer.py   43 crossing/planning tests
└── test_placement.py           14 placement/power tests
examples/                    Sample netlists
.github/workflows/           CI + auto-publish to PyPI
```

## Python import

The PyPI package name is `pcb-crossing-optimizer`, but the importable module is `crossing_analyzer`:

```python
# Crossing analysis
from crossing_analyzer import sweep_optimize, PinColumn, parse_netlist, plan_footprint

# Placement
from crossing_analyzer.placement_engine import place_components, placement_to_dict
from crossing_analyzer.board_constraints import BoardConstraints, FixedPosition
from crossing_analyzer.footprint_parser import resolve_footprint_geometry

# Power validation
from crossing_analyzer.power_validator import trace_width_ipc2152, validate_power_traces
```

## MCP integration (AI agents)

When used via the [skidl-vscode](https://github.com/ashergarland/skidl-vscode) MCP server, AI agents have access to these tools:

| Tool | Purpose |
|------|---------|
| `parse_netlist` | Inspect netlist components and connectivity |
| `suggest_crossing_layers` | Auto-suggest layer arrangement for crossing analysis |
| `analyze_crossings` | Minimize trace crossings, compute optimal pin orderings |
| `plan_footprint` | Optimal pin-to-net assignment for custom footprints |
| `suggest_placement` | Full auto-placement with force-directed + annealing |
| `validate_power_traces` | IPC-2152 power trace width validation |

Typical AI agent workflow:
```
parse_netlist → suggest_crossing_layers → analyze_crossings → suggest_placement → validate_power_traces
```

## Running tests

```
pip install -e .
pip install pytest
pytest -v
```

## Publishing to PyPI

Releases are published automatically when changes to `src/` or `pyproject.toml` are pushed to `main`. The workflow uses GitHub Actions trusted publishing (OIDC) with `skip-existing: true`, so only version bumps trigger actual publishes.

### Releasing a new version

1. Update the version in `pyproject.toml`
2. Commit and push to `main`
3. The workflow publishes automatically if the version is new

### Manual publishing (fallback)

```bash
pip install build twine
python -m build
twine upload dist/*
```

## License

MIT
