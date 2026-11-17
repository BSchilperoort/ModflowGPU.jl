# BMI

```@meta
CurrentModule = ModflowGPU
```

ModflowGPU implements the [Basic Model Interface](https://bmi.csdms.io) (BMI) of
[BasicModelInterface.jl](https://github.com/Deltares/BasicModelInterface.jl) (version
0.1) for its [`Simulation`](@ref) type, so that a BMI host (a coupler that drives models
through BMI calls only) can run it like any other BMI model. The methods are built on the
[stepping API](stepping.md), and follow libmf6's BMI: variable names are the libmf6
addresses of the stepping API, and the model has one grid.

The BMI functions belong to BasicModelInterface, not to ModflowGPU. Add the package to
your environment (`import Pkg; Pkg.add("BasicModelInterface")`) and call them with its
module name:

```julia
import BasicModelInterface as BMI
using ModflowGPU
sim = BMI.initialize(Simulation, "path/to/simulation"; backend = select_backend("auto"))
```

## Methods

### Control and time

| Method | ModflowGPU |
| :--- | :--- |
| `BMI.initialize(Simulation, config_file; kwargs...)` | [`initialize`](@ref): `config_file` is the simulation folder or its `mfsim.nam`; the keywords are those of `initialize` (`backend`, `precision`, `outdir`, `write_output`, `closures`, `preconditioner`, `precond_options`) |
| `BMI.update(sim)` | one time step ([`ModflowGPU.update`](@ref)); throws a [`ConvergenceError`](@ref) when the step does not converge and the simulation has no CONTINUE option; returns `nothing` |
| `BMI.update_until(sim, time)` | time steps until the current time reaches `time` or the simulation ends; the last step may end after `time` (time steps are never split) |
| `BMI.finalize(sim)` | [`finalize_simulation`](@ref): closes the output files and frees the GPU memory |
| `BMI.get_component_name(sim)` | `"ModflowGPU (MODFLOW 6 GWF)"` |
| `BMI.get_start_time(sim)`, `get_end_time`, `get_current_time`, `get_time_step` | as [`get_start_time`](@ref), [`get_end_time`](@ref), [`get_current_time`](@ref), [`get_time_step`](@ref) |
| `BMI.get_time_units(sim)` | the TDIS `TIME_UNITS`: `"seconds"`, `"minutes"`, `"hours"`, `"days"`, `"years"`, or `"undefined"` when the input gives none |

Times start at 0 and are in the time units of the model. `get_current_time` is the time
at the end of the last completed time step (0 before the first `update`).

### Variables

Variable names are the addresses of the [stepping API](@ref "Addresses"), e.g.
`"GWF/X"` or `"GWF/RIV-1/BOUND"`.

| Method | ModflowGPU |
| :--- | :--- |
| `BMI.get_input_var_names(sim)`, `get_input_item_count` | [`input_addresses`](@ref): `<model>/X` and the `BOUND` of every stress package |
| `BMI.get_output_var_names(sim)`, `get_output_item_count` | [`var_addresses`](@ref): every address (the input ones too) |
| `BMI.get_var_type(sim, name)` | `"Float64"` or `"Int32"` |
| `BMI.get_var_units(sim, name)` | `"L"`, `"L2"`, `"L3/T"`, `"T"` or `"-"`, in the model's length and time units (see the [address table](@ref "Addresses")) |
| `BMI.get_var_itemsize(sim, name)`, `get_var_nbytes` | bytes of one value, of the whole array |
| `BMI.get_var_grid(sim, name)` | 1 for a model variable (`<model>/...`); an `ArgumentError` for `SLN_1/...` and `TDIS/...` |
| `BMI.get_var_location(sim, name)` | `"node"` |
| `BMI.get_value(sim, name, dest)` | copies the values, flattened, into `dest` and returns it ([`get_value`](@ref)) |
| `BMI.get_value_ptr(sim, name)` | the host mirror ([`get_value_ptr`](@ref)), a `Matrix` for `BOUND` |
| `BMI.get_value_at_indices(sim, name, dest, inds)` | the values at the linear indices `inds` of the flattened array |
| `BMI.set_value(sim, name, values)` | [`set_value`](@ref): sent to the device at once |
| `BMI.set_value_at_indices(sim, name, inds, values)` | sets the values at the linear indices `inds`, keeps the others |

Values are in the forms of the stepping API: `Float64` with absolute heads, `BOUND`
before the MULTIPLIER auxiliary variable, `Int32` for counts and node numbers. The host
mirror returned by `get_value_ptr` is a copy of the engine data, synchronised at the
stepping calls; see [Host mirrors and device arrays](@ref). The device arrays are not
reachable through BMI.

!!! note "Setting boundary values through BMI"
    `BMI.update` runs a whole time step, so a BMI host sets values *before* the step
    starts. At the first step of a stress period, the period data of a package (a PERIOD
    block in its input) replace the values set before, as in MF6. Values set for a package
    without a PERIOD block for the coming period are kept. To set boundary values in
    every period, call the stepping API directly and set them after
    [`prepare_time_step`](@ref).

## Grid

Every model variable is on grid 1, the model grid. Its size is the number of cells of the
grid, user nodes, including cells with IDOMAIN ≤ 0. Node variables such as `X` have one
value per *model* node (cells with IDOMAIN > 0) instead; `<model>/DIS/NODEUSER` gives
the user node of each model node. Without IDOMAIN the two are the same.

| Method | DIS | DISV, DISU |
| :--- | :--- | :--- |
| `BMI.get_grid_type(sim, 1)` | `"rectilinear"` | `"unstructured"` |
| `BMI.get_grid_rank(sim, 1)` | 2 with one layer, else 3 | 3 |
| `BMI.get_grid_size(sim, 1)`, `get_grid_node_count` | `nlay·nrow·ncol` | number of cells (DISV: `nlay·ncpl`; DISU: `nodes`) |
| `BMI.get_grid_shape(sim, 1, shape)` | `[nrow, ncol]` with one layer, else `[nlay, nrow, ncol]` | `ArgumentError` |
| `BMI.get_grid_x(sim, 1, x)` | the `ncol + 1` column edges, `xorigin .+ [0; cumsum(delr)]` | `ArgumentError` |
| `BMI.get_grid_y(sim, 1, y)` | the `nrow + 1` row edges from north to south, `yorigin + sum(delc) .- [0; cumsum(delc)]` | `ArgumentError` |

The coordinates are those of the model (`XORIGIN`, `YORIGIN`); the rotation `ANGROT` is
not applied. Cell elevations are the `<model>/DIS/TOP` and `<model>/DIS/BOT` variables,
and the cell areas `<model>/DIS/AREA`.

### Not implemented

These BMI functions have no method for `Simulation` (calling them throws a
`MethodError`): `get_grid_origin`, `get_grid_spacing`, `get_grid_z`,
`get_grid_edge_count`, `get_grid_face_count`, `get_grid_edge_nodes`,
`get_grid_face_edges`, `get_grid_face_nodes` and `get_grid_nodes_per_face`. Use
`get_grid_x` and `get_grid_y` for a DIS grid, and the DIS variables for the elevations.

## A BMI loop

A steady-state model of 10 × 5 cells with constant heads on the west and east edges and a
well in the middle, written as MF6 input and run through BMI. The well pumps the rate of
its input in the first period, and a rate set through BMI in the next two (its input has
no data for periods 2 and 3, so the set values are kept):

```jldoctest bmi
julia> import BasicModelInterface as BMI

julia> using ModflowGPU

julia> dis = build_dis(; delr = 100.0, delc = 100.0, top = 0.0, botm = fill(-20.0, 10, 5, 1));

julia> chd = build_boundary(:CHD, dis;
           periods = [1 => (; cells = [(1, i, j) for i in 1:5 for j in (1, 10)], values = zeros(10))]);

julia> wel = build_boundary(:WEL, dis; periods = [1 => (; cells = [(1, 3, 5)], values = [-100.0])]);

julia> gwf = build_model("flow"; dis, npf = build_npf(dis; k = 10.0), ic = build_ic(dis; strt = 0.0),
           boundaries = [chd, wel]);

julia> spec = build_simulation(gwf; tdis = build_tdis(; perlen = fill(10.0, 3), time_units = :days));

julia> simdir = write_simulation(joinpath(mktempdir(), "flow"), spec);

julia> sim = BMI.initialize(Simulation, simdir; precision = :double);
┌ Info: IMS preconditioner ILU0 is replaced by mc_ilu0 (multicolour ILU0: red–black on DIS grids, greedy colouring on DISV and DISU; the default, see the `preconditioner` option)
└   ims.file = "flow.ims"

julia> BMI.get_time_units(sim), BMI.get_end_time(sim)
("days", 30.0)

julia> BMI.get_grid_type(sim, 1), BMI.get_grid_shape(sim, 1, zeros(Int, 2))
("rectilinear", [5, 10])

julia> BMI.get_grid_x(sim, 1, zeros(11))
11-element Vector{Float64}:
    0.0
  100.0
  200.0
  300.0
  400.0
  500.0
  600.0
  700.0
  800.0
  900.0
 1000.0

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
