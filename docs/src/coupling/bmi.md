# BMI

```@meta
CurrentModule = ModflowGPU
```

ModflowGPU implements the [Basic Model Interface](https://bmi.csdms.io) (BMI) of
[BasicModelInterface.jl](https://github.com/Deltares/BasicModelInterface.jl) (version
0.1) for its [`Simulation`](@ref) type, so that a BMI host (a coupler that drives models
through BMI calls only) can run it like any other BMI model. The methods are built on the
[stepping API](stepping.md), and follow libmf6's BMI where the BMI specification allows:
variable names are the libmf6 addresses of the stepping API. All 41 functions of the
specification have a method; the [conformance table](@ref "Conformance") lists what each one
does and the [deviations](@ref "Deviations from the specification") that remain.

The BMI functions belong to BasicModelInterface, not to ModflowGPU. Add the package to
your environment (`import Pkg; Pkg.add("BasicModelInterface")`) and call them with its
module name:

```julia
import BasicModelInterface as BMI
using ModflowGPU
sim = BMI.initialize(Simulation, "path/to/run.toml")
```

## Configuration file

As the BMI specification asks, `BMI.initialize(Simulation, config_file)` takes no other
arguments: the settings come from a configuration file in [TOML](https://toml.io)
format. Its keys are the keywords of [`initialize`](@ref):

```toml
simulation = "model"          # the simulation folder or its mfsim.nam (required)
backend = "rocm"              # a name for select_backend: "cpu" (default), "auto", "rocm", "cuda", ...
precision = "single"          # "auto" (default), "double", "mixed" or "single"
preconditioner = "auto"       # "auto" (default), "jacobi", "chebyshev", "mc_sgs", "mc_ilu0" or "amg"
outdir = "output"             # folder for the output files (default: the simulation folder)
write_output = true           # false writes no output files

[closures]                    # overrides of the IMS closures (optional)
outer_dvclose = 1.0e-4
inner_maximum = 200

[precond_options]             # options of the preconditioner (optional)
relax = 0.97
```

- Only `simulation` is required; the other keys take the defaults of [`initialize`](@ref).
  Unknown keys, closures that IMS does not have, and values of the wrong type are errors.
- `simulation` and `outdir` are relative to the folder of the configuration file (or
  absolute).
- `closures` takes `outer_dvclose`, `outer_maximum`, `inner_dvclose`, `inner_rclose` and
  `inner_maximum`. Strings in `precond_options` become symbols (for example the AMG
  option `smoother = "chebyshev"`).
- A GPU backend needs its package loaded before `BMI.initialize` is called (`using
  AMDGPU`, `using CUDA`, ...; see [GPUs and precision](../gpu.md)).
- `config_file` can also be the simulation folder or its `mfsim.nam`, which runs with
  all defaults (on the CPU, with output files in the simulation folder).

To pass settings as keywords instead, call [`initialize`](@ref) of the stepping API: it
returns the same `Simulation`, which the BMI functions then accept.
`BMI.initialize(Simulation, config_file; precision = :double)` is an error.

## Methods

### Control and time

| Method | ModflowGPU |
| :--- | :--- |
| `BMI.initialize(Simulation, config_file)` | [`initialize`](@ref) with the settings of the [configuration file](@ref "Configuration file") |
| `BMI.update(sim)` | one time step ([`ModflowGPU.update`](@ref)); throws a [`ConvergenceError`](@ref) when the step does not converge and the simulation has no CONTINUE option |
| `BMI.update_until(sim, time)` | time steps until the current time is `time`, which must be the end of a time step; see below |
| `BMI.finalize(sim)` | [`finalize_simulation`](@ref): closes the output files and frees the GPU memory |
| `BMI.get_component_name(sim)` | `"ModflowGPU (MODFLOW 6 GWF)"` |
| `BMI.get_start_time(sim)`, `get_end_time`, `get_current_time`, `get_time_step` | as [`get_start_time`](@ref), [`get_end_time`](@ref), [`get_current_time`](@ref), [`get_time_step`](@ref) |
| `BMI.get_time_units(sim)` | the TDIS `TIME_UNITS`: `"seconds"`, `"minutes"`, `"hours"`, `"days"` or `"years"`; `"none"` when the input gives none |

Times start at 0 and are in the time units of the model. `get_current_time` is the time
at the end of the last completed time step (0 before the first `update`).

ModflowGPU does not split time steps, so `update_until(sim, time)` accepts only a `time`
at the end of a time step (within a relative tolerance of 10⁻¹⁰ of the end time) and
throws an `ArgumentError` for any other time, for a time before the current time, and for
a time after the end of the simulation. `time` equal to the current time does nothing.
To couple at times that are not step ends, give the model matching time steps in TDIS.

### Variables

Variable names are the addresses of the [stepping API](@ref "Addresses"), e.g.
`"GWF/X"` or `"GWF/RIV-1/BOUND"`.

| Method | ModflowGPU |
| :--- | :--- |
| `BMI.get_input_var_names(sim)`, `get_input_item_count` | [`input_addresses`](@ref): `<model>/X` and the `BOUND` of every stress package |
| `BMI.get_output_var_names(sim)`, `get_output_item_count` | [`var_addresses`](@ref): every address (the input ones too) |
| `BMI.get_var_type(sim, name)` | `"Float64"` or `"Int32"` |
| `BMI.get_var_units(sim, name)` | UDUNITS units, see [Units](@ref "Units") |
| `BMI.get_var_itemsize(sim, name)`, `get_var_nbytes` | bytes of one value (8 or 4), of the whole variable (`itemsize · length`) |
| `BMI.get_var_grid(sim, name)` | the grid of the variable, see [Grids](@ref "Grids") |
| `BMI.get_var_location(sim, name)` | `"face"` for a variable on an unstructured grid (one value per cell), else `"node"` |
| `BMI.get_value(sim, name, dest)` | copies the values, flattened, into `dest` and returns `dest` ([`get_value`](@ref)); `dest` must have the length of the variable |
| `BMI.get_value_ptr(sim, name)` | the host mirror ([`get_value_ptr`](@ref)) as a flat `Vector` (for `BOUND` a vector that shares the memory of the `(ncol, maxbound)` mirror) |
| `BMI.get_value_at_indices(sim, name, dest, inds)` | the values at the indices `inds` (1-based) of the flattened variable, into `dest`; returns `dest` |
| `BMI.set_value(sim, name, values)` | [`set_value`](@ref): sent to the device at once; returns `nothing` |
| `BMI.set_value_at_indices(sim, name, inds, values)` | sets the values at the indices `inds` (1-based), keeps the others |

Values are in the forms of the stepping API: `Float64` with absolute heads (also in
`precision = :single` runs), `BOUND` before the MULTIPLIER auxiliary variable, `Int32`
for counts and node numbers. Arrays are flattened in Julia's (column-major) order, so a
`BOUND` of `(ncol, maxbound)` lists the values of record 1, then of record 2, and so on.
The host mirror returned by `get_value_ptr` is a copy of the engine data, synchronised at
the stepping calls; see [Host mirrors and device arrays](@ref). The device arrays are not
reachable through BMI.

!!! note "Setting boundary values through BMI"
    `BMI.update` runs a whole time step, so a BMI host sets values *before* the step
    starts. At the first step of a stress period, the period data of a package (a PERIOD
    block in its input) replace the values set before, as in MF6. Values set for a package
    without a PERIOD block for the coming period are kept. To set boundary values in
    every period, call the stepping API directly and set them after
    [`prepare_time_step`](@ref).

### Units

`BMI.get_var_units` gives the units in [UDUNITS](https://docs.unidata.ucar.edu/udunits/current/)
syntax, from the DIS `LENGTH_UNITS` and the TDIS `TIME_UNITS` of the model, as the
Units column of the [address table](@ref "Addresses"):

| Address table | `LENGTH_UNITS` meters, `TIME_UNITS` days | Not given in the input |
| :--- | :--- | :--- |
| L (heads, elevations) | `"m"` | `"L"` |
| L² (areas) | `"m2"` | `"L2"` |
| L³/T (flows: `SIMVALS`) | `"m3 d-1"` | `"L3 T-1"` |
| T (`TDIS/DELT`, `TOTIM`, `PERTIM`) | `"d"` | `"T"` |
| - (counts, node numbers, `IBOUND`) | `"1"` | `"1"` |

The length units are `"m"`, `"ft"` or `"cm"`, the time units `"s"`, `"min"`, `"h"`, `"d"`
or `"yr"`. MODFLOW itself does not use units, so a model may leave them out; the BMI then
gives the placeholders `"L"` and `"T"`, which are not UDUNITS units. `BOUND` has the
units of its column for the packages with one column (CHD: L, WEL: L³/T, RCH and RCHA:
L/T), and `"none"` for the others, whose columns have different units (see the
[BOUND columns](@ref "BOUND columns")).

## Grids

Every variable is on one of these grids:

| Grid | Variables | Type |
| :--- | :--- | :--- |
| 0 | the variables with one value (`TDIS/...`, `SLN_1/MXITER`, `DIS/NODES`, `DIS/NODESUSER`, `NBOUND`, `MAXBOUND`) | `"scalar"` |
| 1 | the model grid, all cells (user nodes): `DIS/NODEREDUCED`, and the model-node variables below when every cell is active | DIS: `"rectilinear"` (one layer, `ANGROT` 0) or `"structured_quadrilateral"`; DISV and DISU: `"unstructured"` (`"vector"` for DISU without vertices) |
| 2 | the model nodes (cells with IDOMAIN > 0): `X`, `XOLD`, `HREF`, `IBOUND`, `DIS/NODEUSER`, `DIS/TOP`, `DIS/BOT`, `DIS/AREA` when some cells have IDOMAIN ≤ 0 | `"points"` at the cell centres (`"vector"` for DISU without vertices) |
| 1 + 2i | the records of the i-th stress package (`NODELIST`, `SIMVALS`), `MAXBOUND` values | `"vector"` |
| 2 + 2i | the `BOUND` of the i-th stress package, `ncol · MAXBOUND` values | `"vector"` |

The i-th stress package is the i-th package with `BOUND` in
`BMI.get_output_var_names`. With this layout the length of every variable is the size of
its grid (`get_grid_size`), or for a variable at `"face"` the number of faces
(`get_grid_face_count`). Without IDOMAIN (or with IDOMAIN > 0 everywhere) the heads are
on grid 1, otherwise on grid 2; `BMI.get_var_grid(sim, "<model>/X")` tells which.

| Method | DIS (grid 1) | DISV, DISU (grid 1) | Grid 2 (`"points"`) | `"scalar"`, `"vector"` |
| :--- | :--- | :--- | :--- | :--- |
| `get_grid_rank` | 2 with one layer, else 3 | 2 | 3 | 0, 1 |
| `get_grid_size`, `get_grid_node_count` | `nlay·nrow·ncol` | number of vertices (`NVERT`) | number of model nodes | 1, length |
| `get_grid_shape(sim, g, shape)` | `[nrow, ncol]`, or `[nlay, nrow, ncol]` | not supported | not supported | `[]`, `[length]` |
| `get_grid_x`, `get_grid_y` | rectilinear: the `ncol` column and `nrow` row centres (rows from north to south); structured quadrilateral: the centre of every cell | the vertices | the cell centres | not supported |
| `get_grid_z` | structured quadrilateral of rank 3: the cell-centre elevations; else not supported | not supported | the cell-centre elevations | not supported |
| `get_grid_face_count`, `get_grid_nodes_per_face`, `get_grid_face_nodes` | not supported | one face per cell (`nlay·ncpl` for DISV), its vertices counter-clockwise | not supported | not supported |
| `get_grid_edge_count`, `get_grid_edge_nodes`, `get_grid_face_edges` | not supported | the vertex pairs that bound the cells | not supported | not supported |
| `get_grid_spacing`, `get_grid_origin` | not supported | not supported | not supported | not supported |

- **Coordinates** are real-world coordinates: the model coordinates rotated by `ANGROT`
  about the lower-left corner of the grid and moved to `XORIGIN`, `YORIGIN`, as MF6 does.
  A DIS grid with `ANGROT` ≠ 0 or more than one layer is `"structured_quadrilateral"`,
  with the coordinates of every cell (`get_grid_size` values, in user-node order: layer
  by layer, row by row from the north).
- **Grid nodes** of a DIS grid are the cell centres, so that the heads are at the nodes.
  For DISV and DISU, BMI (which follows the [UGRID conventions](https://ugrid-conventions.github.io/ugrid-conventions/))
  calls the vertices nodes and the cells faces; the faces of all layers share the
  vertices of the plan view. MF6 lists cell vertices clockwise; BMI gets them
  counter-clockwise.
- **Indices** in `get_grid_face_nodes`, `get_grid_edge_nodes` and `get_grid_face_edges`
  are 1-based, as the indices of `get_value_at_indices`.
- **Not supported** means that the function throws an `ArgumentError` whose message
  starts with `BMI.<function> is not supported:` and says why.

## Conformance

Every function of the specification, as ModflowGPU implements it (BasicModelInterface.jl
0.1 for the signatures, the [CSDMS documentation](https://bmi.csdms.io) for the meaning):

| Function | Conforms | Notes |
| :--- | :--- | :--- |
| `initialize` | yes | TOML configuration file; `initialize(Simulation)` without a file is not supported |
| `update`, `finalize` | yes | `update` throws on a step that does not converge |
| `update_until` | partly | `time` must be the end of a time step (steps are not split) |
| `get_component_name`, `get_input_item_count`, `get_output_item_count`, `get_input_var_names`, `get_output_var_names` | yes | |
| `get_var_grid`, `get_var_type`, `get_var_itemsize`, `get_var_nbytes`, `get_var_location` | yes | type names are Julia's (`"Float64"`, `"Int32"`) |
| `get_var_units` | partly | UDUNITS when the model gives `LENGTH_UNITS` and `TIME_UNITS`, else `"L"`, `"T"`; `"none"` for multi-column `BOUND` |
| `get_current_time`, `get_start_time`, `get_end_time`, `get_time_step` | yes | |
| `get_time_units` | yes | `"none"` when TDIS has no `TIME_UNITS` |
| `get_value`, `get_value_at_indices`, `set_value`, `set_value_at_indices` | yes | return `dest` (getters) or `nothing` (setters); 1-based indices |
| `get_value_ptr` | partly | a host mirror synchronised at every time step, not the model's own memory |
| `get_grid_type`, `get_grid_rank`, `get_grid_size`, `get_grid_shape`, `get_grid_node_count` | yes | |
| `get_grid_x`, `get_grid_y` | yes | not for `"scalar"`, `"vector"` grids, which have no coordinates |
| `get_grid_z` | partly | not for rectilinear grids of rank 2 and unstructured grids (vertices have no elevation) |
| `get_grid_spacing`, `get_grid_origin` | n/a | no grid is `"uniform_rectilinear"`; not supported |
| `get_grid_edge_count`, `get_grid_face_count`, `get_grid_edge_nodes`, `get_grid_face_edges`, `get_grid_face_nodes`, `get_grid_nodes_per_face` | yes | unstructured grids (DISV, DISU with vertices) only |

### Deviations from the specification

- **Time steps are not split**: `update_until` accepts only times at the end of a time
  step, and never runs backwards.
- **`get_value_ptr` is a mirror**: the model's data are on the device, so the returned
  vector is a host copy that the time-step calls keep up to date; writes into it reach
  the model at the next time step. See [Host mirrors and device arrays](@ref).
- **Units placeholders**: without `LENGTH_UNITS` or `TIME_UNITS` in the input, the units
  contain `L` or `T`, which UDUNITS does not know. `BOUND` of RIV, GHB, DRN and EVT has
  `"none"`.
- **1-based indices**: indices passed to `*_at_indices` and returned by the unstructured
  grid functions start at 1 (Julia), where the CSDMS examples (in C and Python) start
  at 0.
- **Grids of layered unstructured models** are given in plan view (rank 2, no
  `get_grid_z`); the cell elevations are the variables `<model>/DIS/TOP` and `BOT`.
- **Values in Float64**: a `precision = :single` run still exchanges Float64 values (the
  host mirrors), so `get_var_type` does not show the precision of the run.

!!! note "Changes from earlier versions"
    Before the BMI was checked against the specification, `BMI.initialize` took the
    settings as keywords; put them in a configuration file now. `get_grid_x` and
    `get_grid_y` of a DIS grid gave the `ncol + 1` and `nrow + 1` cell edges without
    `ANGROT`; they now give the cell centres in real-world coordinates. Grids 0 and 2 and
    the package grids are new (the scalar and package variables had no grid), DISV and
    DISU grids give their vertices and faces, the units are UDUNITS strings (`"m3 d-1"`
    for `"L3/T"`), `get_time_units` gives `"none"` instead of `"undefined"`,
    `get_value_ptr` returns a flat `Vector` for `BOUND`, and `update_until` throws for
    times that are not step ends instead of running past them.

## A BMI loop

A steady-state model of 10 × 5 cells with constant heads on the west and east edges and a
well in the middle, written as MF6 input and run through BMI calls only. The well pumps
the rate of its input in the first period, and a rate set through BMI in the next two
(its input has no data for periods 2 and 3, so the set values are kept):

```jldoctest bmi
julia> import BasicModelInterface as BMI

julia> using ModflowGPU

julia> dis = build_dis(; delr = 100.0, delc = 100.0, top = 0.0, botm = fill(-20.0, 10, 5, 1), length_units = :meters);

julia> chd = build_boundary(:CHD, dis;
           periods = [1 => (; cells = [(1, i, j) for i in 1:5 for j in (1, 10)], values = zeros(10))]);

julia> wel = build_boundary(:WEL, dis; periods = [1 => (; cells = [(1, 3, 5)], values = [-100.0])]);

julia> gwf = build_model("flow"; dis, npf = build_npf(dis; k = 10.0), ic = build_ic(dis; strt = 0.0),
           boundaries = [chd, wel]);

julia> spec = build_simulation(gwf; tdis = build_tdis(; perlen = fill(10.0, 3), time_units = :days));

julia> simdir = write_simulation(joinpath(mktempdir(), "flow"), spec);

julia> config = joinpath(dirname(simdir), "flow.toml");

julia> write(config, """
           simulation = "flow"
           precision = "double"
           """);

julia> sim = BMI.initialize(Simulation, config);
┌ Info: IMS preconditioner ILU0 is replaced by mc_ilu0 (multicolour ILU0: red–black on DIS grids, greedy colouring on DISV and DISU; the default, see the `preconditioner` option)
└   ims.file = "flow.ims"

julia> BMI.get_time_units(sim), BMI.get_end_time(sim), BMI.get_var_units(sim, "FLOW/WEL-1/SIMVALS")
("days", 30.0, "m3 d-1")

julia> grid = BMI.get_var_grid(sim, "FLOW/X")
1

julia> BMI.get_grid_type(sim, grid), BMI.get_grid_shape(sim, grid, zeros(Int, 2))
("rectilinear", [5, 10])

julia> BMI.get_grid_x(sim, grid, zeros(10))
10-element Vector{Float64}:
  50.0
 150.0
 250.0
 350.0
 450.0
 550.0
 650.0
 750.0
 850.0
 950.0

julia> heads = BMI.get_value_ptr(sim, "FLOW/X");   # host mirror, updated by every step

julia> for rate in (nothing, -200.0, -300.0)
           isnothing(rate) || BMI.set_value(sim, "FLOW/WEL-1/BOUND", [rate])
           BMI.update(sim)
           q = only(BMI.get_value(sim, "FLOW/WEL-1/SIMVALS", zeros(1)))
           println("t = $(BMI.get_current_time(sim)) d: well $q m³/d, lowest head $(round(minimum(heads); digits = 3)) m")
       end
t = 10.0 d: well -100.0 m³/d, lowest head -0.334 m
t = 20.0 d: well -200.0 m³/d, lowest head -0.667 m
t = 30.0 d: well -300.0 m³/d, lowest head -1.001 m

julia> BMI.finalize(sim)
```
