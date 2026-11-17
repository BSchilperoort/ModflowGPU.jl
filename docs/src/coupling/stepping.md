# Stepping API

```@meta
CurrentModule = ModflowGPU
```

[`run_simulation`](@ref) runs a simulation from start to end. The stepping API runs it one
time step at a time, so that another model, or your own Julia code, can read the heads and
flows and change boundary values between steps. It follows the API of libmf6 (the
MODFLOW 6 shared library, called the XMI in its documentation, and used from Python
through xmipy and modflowapi): the same calls in the same order, and the same variable
addresses. Code written for libmf6 can be ported call by call.

The functions are listed in the [Stepping API reference](../api/stepping.md). The
[BMI](bmi.md) page describes the Basic Model Interface methods built on them.

## The time-step loop

[`initialize`](@ref) loads a simulation (a folder with `mfsim.nam`, or a
[`SimulationSpec`](@ref) made with the [builder functions](../api/builder.md)) and builds
the model on the CPU or the GPU. It takes the same keywords as `run_simulation`
(`backend`, `precision`, `outdir`, `write_output`, `closures`, `preconditioner`,
`precond_options`). Each time step is then a sequence of calls:

```julia
using ModflowGPU
sim = initialize("path/to/simulation"; backend = select_backend("auto"))
while get_current_time(sim) < get_end_time(sim)
    prepare_time_step(sim)        # next time step; period data at the start of a period
    # ... set boundary values for this step ...
    prepare_solve(sim)            # start the step
    for _ in 1:only(get_value(sim, "SLN_1/MXITER"))
        ModflowGPU.solve(sim) && break   # one outer iteration; true when converged
    end
    finalize_solve(sim)           # flows (SIMVALS) and budget of the step
    finalize_time_step(sim)       # output selected by OC
    # ... read heads and flows ...
end
finalize_simulation(sim)          # close the output files, free the GPU memory
```

[`do_time_step`](@ref) does `prepare_solve`, the outer iterations and `finalize_solve`
in one call, and [`ModflowGPU.update`](@ref) a whole time step. Most coupling code only
needs `prepare_time_step`, `do_time_step` and `finalize_time_step`: set inputs after
`prepare_time_step`, read outputs after `finalize_time_step` (or after `do_time_step`).

The calls must come in this order; a call out of order throws an `ArgumentError` that
says which calls may come before it. After `prepare_time_step`,
[`get_current_time`](@ref) is the time at the *end* of the new step, as in libmf6, and
[`get_time_step`](@ref) its length.
All times are in the time units of the TDIS input (`TIME_UNITS`), from 0 at the start of
the simulation.

Two functions are public but not exported, so write them with the module name:
`ModflowGPU.update` (CUDA.jl exports an `update` too) and `ModflowGPU.solve` (as do the
SciML packages, through CommonSolve). libmf6's `finalize` is
[`finalize_simulation`](@ref), so that it does not clash with Julia's `finalize`.

### From libmf6, xmipy and modflowapi

| libmf6 (XMI) / xmipy | ModflowGPU | Notes |
| :--- | :--- | :--- |
| `mf6 = XmiWrapper(lib, ws)`; `mf6.initialize()` | `sim = initialize(simdir; backend, precision, ...)` | also takes a `SimulationSpec` |
| `mf6.update()` | `ModflowGPU.update(sim)` | returns `false` when the step failed to converge without CONTINUE |
| `mf6.finalize()` | `finalize_simulation(sim)` | frees the GPU memory at once |
| `mf6.prepare_time_step(dt)` | `prepare_time_step(sim)` | `dt` may be passed and is ignored, as in libmf6 |
| `mf6.do_time_step()` | `do_time_step(sim)` | returns whether the step converged |
| `mf6.finalize_time_step()` | `finalize_time_step(sim)` | |
| `mf6.prepare_solve(1)` | `prepare_solve(sim)` | one solution only (`isol = 1`) |
| `mf6.solve(1)` | `ModflowGPU.solve(sim)` | one outer iteration; returns `true` when converged |
| `mf6.finalize_solve(1)` | `finalize_solve(sim)` | returns whether the step converged |
| `mf6.get_subcomponent_count()` | none | always one solution, `SLN_1` |
| `mf6.get_current_time()`, `get_start_time()`, `get_end_time()`, `get_time_step()` | [`get_current_time`](@ref), [`get_start_time`](@ref), [`get_end_time`](@ref), [`get_time_step`](@ref) | |
| `mf6.get_var_address("X", "GWF")` | `get_var_address(sim, "X", "GWF")` | the subcomponent is a keyword: `subcomponent_name = "RIV-1"` |
| `mf6.get_value_ptr(address)` | `get_value_ptr(sim, address)` | a host *copy*, synchronised at the calls: see [below](@ref "Host mirrors and device arrays") |
| `mf6.get_value(address)` | `get_value(sim, address)` | a new array, read now |
| `mf6.set_value(address, values)` | `set_value(sim, address, values)` | sent to the device at once |
| `mf6.get_input_var_names()`, `get_output_var_names()` | [`input_addresses`](@ref), [`var_addresses`](@ref) | |

modflowapi's `run_simulation(dll, sim_ws, callback)` calls one callback at fixed points
of this loop. With the stepping API you write the loop yourself and put that code in its
place: `Callbacks.initialize` after `initialize`; `stress_period_start` and
`timestep_start` after `prepare_time_step` (a period starts when
`only(get_value(sim, "TDIS/KSTP")) == 1`); `iteration_start` and `iteration_end` around
`ModflowGPU.solve`; `timestep_end` after `finalize_time_step`, and `stress_period_end`
there too at the last step of a period; `finalize` before `finalize_simulation`. modflowapi's package objects (`model.riv.stress_period_data`, ...)
have no counterpart: read and write the `BOUND` arrays directly.

What is different from libmf6:

- `get_value_ptr` returns a host copy of the engine data, not the engine memory itself;
  see [Host mirrors and device arrays](@ref).
- Only one GWF model with one IMS solution, and the addresses below. There are no
  exchanges and no other model types.
- `prepare_time_step` cannot change the step length (no adaptive time stepping).

## Addresses

Variables are named by libmf6's addresses: the component, an optional subcomponent and
the variable name, in upper case and joined by `/`. `<model>` is the name of the GWF model
(e.g. `GWF`), `<package>` the name of a stress package (e.g. `RIV-1`, or the name given in
the name file or to [`build_boundary`](@ref)). [`get_var_address`](@ref) builds them:

```julia
get_var_address(sim, "x", "gwf")                                     # "GWF/X"
get_var_address(sim, "bound", "gwf"; subcomponent_name = "riv-1")   # "GWF/RIV-1/BOUND"
```

[`var_addresses`](@ref) lists every address of a simulation (they are all readable), and
[`input_addresses`](@ref) the ones that can be written. Every address is in this table:

| Address | Holds | Shape | Units | Writable |
| :--- | :--- | :--- | :--- | :--- |
| `<model>/X` | heads | `(nodes,)` | L | yes |
| `<model>/XOLD` | heads at the end of the previous time step | `(nodes,)` | L | no |
| `<model>/HREF` | the reference head of each cell (see [below](@ref "Device arrays")) | `(nodes,)` | L | no |
| `<model>/IBOUND` | cell status: 1 active, −1 constant head, 0 inactive or dry | `(nodes,)` | - | no |
| `<model>/DIS/NODEUSER` | user node number of each model node | `(nodes,)` | - | no |
| `<model>/DIS/NODEREDUCED` | model node of each user node; 0 for IDOMAIN = 0, −1 for IDOMAIN < 0 | `(nodesuser,)` | - | no |
| `<model>/DIS/TOP` | cell top | `(nodes,)` | L | no |
| `<model>/DIS/BOT` | cell bottom | `(nodes,)` | L | no |
| `<model>/DIS/AREA` | cell area in plan | `(nodes,)` | L² | no |
| `<model>/DIS/NODES` | number of model nodes | `(1,)` | - | no |
| `<model>/DIS/NODESUSER` | number of user nodes (cells of the grid) | `(1,)` | - | no |
| `<model>/<package>/BOUND` | the values of each boundary record (columns below) | `(ncol, maxbound)` | per column | yes |
| `<model>/<package>/NODELIST` | the model node of each record | `(maxbound,)` | - | no |
| `<model>/<package>/SIMVALS` | the simulated flow of each record, positive into the aquifer | `(maxbound,)` | L³/T | no |
| `<model>/<package>/NBOUND` | the number of records in force | `(1,)` | - | no |
| `<model>/<package>/MAXBOUND` | the number of record slots | `(1,)` | - | no |
| `SLN_1/MXITER` | maximum number of outer iterations (IMS `OUTER_MAXIMUM`) | `(1,)` | - | no |
| `TDIS/DELT` | length of the current time step | `(1,)` | T | no |
| `TDIS/TOTIM` | time at the end of the current time step | `(1,)` | T | no |
| `TDIS/PERTIM` | time since the start of the stress period, at the end of the step | `(1,)` | T | no |
| `TDIS/KPER` | stress period number (0 before the first step) | `(1,)` | - | no |
| `TDIS/KSTP` | time step number in the period (0 before the first step) | `(1,)` | - | no |
| `TDIS/NPER` | number of stress periods | `(1,)` | - | no |

Notes on the table:

- **Nodes.** Model nodes are the cells with IDOMAIN > 0, numbered in user-node order;
  `nodes` is their number. User nodes are all cells of the grid, numbered layer by layer,
  row by row (`nodesuser = nlay·nrow·ncol` for DIS). `X(n)` belongs to user node
  `NODEUSER(n)`. Without IDOMAIN the two numberings are the same.
- **Units.** L and T are the length and time units of the model (DIS `LENGTH_UNITS`, TDIS
  `TIME_UNITS`). `BMI.get_var_units` returns the strings of the Units column.
- **Types.** Values are `Float64` and counts and indices `Int32`; scalars are one-element
  vectors (`only(get_value(sim, "TDIS/KPER"))`).
- **DIS.** The grid variables are named `<model>/DIS/...` for DIS, DISV and DISU grids
  (the name of the DISV or DISU package is accepted too).
- **Dry cells.** In a model with convertible cells (ICELLTYPE ≠ 0), the host `X` holds
  HDRY (−1.0e30) in dry cells, as MF6 stores it, and `IBOUND` is 0 there.
- **Records.** A stress package has `MAXBOUND` record slots, of which the first `NBOUND`
  are in force in the current period. `BOUND` and `NODELIST` change when the period data
  change (at `prepare_time_step` of the first step of a period), and the `BOUND` values
  that come from time series at every `prepare_time_step`. Array input (RCHA, EVTA) has
  one record per cell of a layer (`ncpl`).
- **SIMVALS** are computed in `finalize_solve`: after it, they are the flows of the step
  just solved. For CHD they are the flows through the constant-head cells.

### BOUND columns

The rows of `BOUND` are the values of the PERIOD block, in the order of the MF6 input,
without the auxiliary variables (except DRN's drainage depth). A host read gives them as in the input: head-like columns
(marked h) as absolute heads or elevations, and the values *before* the MULTIPLIER
auxiliary variable (`AUXMULTNAME`) is applied.

| Package | Rows of `BOUND` |
| :--- | :--- |
| CHD | 1 head (h) |
| WEL | 1 rate q (L³/T, negative for pumping) |
| RCH, RCHA | 1 recharge rate (L/T) |
| GHB | 1 boundary head (h), 2 conductance (L²/T) |
| RIV | 1 stage (h), 2 conductance (L²/T), 3 river bottom (h) |
| DRN | 1 drain elevation (h), 2 conductance (L²/T), 3 drainage depth DDRN (from `AUXDEPTHNAME`, 0 without it) |
| EVT, EVTA | 1 ET surface (h), 2 maximum ET rate (L/T), 3 extinction depth (L), 4 PETM0 (1 without `SURF_RATE_SPECIFIED`), then the `NSEG − 1` PXDP and the `NSEG − 1` PETM values |

The order of the DRN and EVT rows is ModflowGPU's; check it if you port code that indexes
these rows in libmf6's arrays.

Values set through `BOUND` are used from the next solve until new period data for the
package replace them, as in MF6. A value that comes from a time series is replaced by
the series at the next `prepare_time_step`. They are not checked as the input is (for example that
a river stage is above its bottom), except when written on the device through
[`ModflowGPU.device_bound!`](@ref).

## Host mirrors and device arrays

The engine keeps its data on the device (the GPU, or the CPU for the CPU backend) in the
precision `T` of the simulation (`Float32` for `precision = :single`, `Float64` for
`:mixed` and `:double`), and stores heads as offsets from a reference head per cell.
There are three ways to reach the data.

### `get_value` and `set_value`

[`get_value`](@ref) returns a new host array, read from the engine when you call it:
`Float64`, absolute heads, `BOUND` before the multiplier. `get_value(sim, address, dest)`
fills `dest` instead. [`set_value`](@ref) writes an address of [`input_addresses`](@ref)
(`X` and every `BOUND`) and sends the values to the device at once, through the same
conversion as period data (multiplier, offsets, precision). `BOUND` may be given as a
`(ncol, maxbound)` matrix or flattened.

Both can be called at any point of the loop. Each call copies the whole array, from or
to the device for the engine variables (`X`, `XOLD`, `BOUND`, `SIMVALS`). This is the
simplest and safest choice.

### `get_value_ptr`: host mirrors

[`get_value_ptr`](@ref) returns a host array that *mirrors* the variable, in the same
form as `get_value`. The same array is returned on every call and stays valid until
`finalize_simulation`, so you can keep it, as with libmf6. Unlike libmf6 it is a copy,
which ModflowGPU synchronises with the engine at the call boundaries:

- **Reads.** A mirror is filled when `get_value_ptr` first creates it, and refreshed
  from the engine at the end of `prepare_time_step`, `prepare_solve` and
  `finalize_solve`. So the `X` mirror holds the heads of the start of the step during
  the outer iterations, and the new heads after `finalize_solve`; use `get_value` to read
  heads between outer iterations.
- **Writes.** Writes into the mirror of `X` or a `BOUND` are sent to the device at the
  start of the next `prepare_time_step` or `prepare_solve`, if the mirror changed. Write
  `BOUND` mirrors after `prepare_time_step` and before `prepare_solve` (or
  `do_time_step`), as libmf6 users do.
- **Lost writes.** A write into a mirror between `prepare_solve` and `finalize_solve` is
  discarded by the refresh in `finalize_solve`, with a warning. Use `set_value` there.
- Mirrors of read-only addresses can be written, but the writes are never sent.

Cost: the mirrors of `BOUND` and `NODELIST` are copied only when the engine values change
(new period data, a write); the other mirrors (`X`, `SIMVALS`, ...) are copied from the
device at each of the three refresh points of a time step. A simulation without mirrors
(reading with `get_value` only) moves no array between host and device in a time step,
apart from the heads that OC saves or prints.

### Device arrays

`get_value_ptr(sim, address; device = true)` returns the engine array itself, for code
that runs on the same device (for example a coupled model written with
KernelAbstractions; see the [GPU guide](../gpu.md#Sharing-the-GPU-with-another-model)).
No copy is made and no mirror is kept. It exists for `X`, `XOLD`, `HREF`, `IBOUND`,
`BOUND`, `NODELIST` and `SIMVALS`; the other addresses throw an `ArgumentError`. The
values differ from the host forms:

- they are in the precision `T` of the simulation (`sim.T`); indices are `Int32`;
- heads (`X`, `XOLD`) and the head-like `BOUND` rows are **offsets** from the reference
  heads: the absolute head of model node `n` is `HREF[n] + X[n]`, and the absolute stage
  of RIV record `i` is `HREF[NODELIST[i]] + BOUND[1, i]`;
- `BOUND` has the MULTIPLIER auxiliary variable applied;
- `HREF` is not a device array: it is a host `Vector{Float64}`, fixed at `initialize`
  (from the starting heads). Copy it to the device once if a kernel needs it.

Keep the offset form in kernels where you can. Offsets are small numbers, which
`Float32` resolves much more finely than absolute heads: near a head of 1000 m, `Float32`
steps by about 0.06 mm, and differences of nearly equal heads lose most of their digits.

On a GPU these arrays live in GPU memory: index them only inside kernels or with
whole-array operations (see
[Scalar indexing is disallowed](../gpu.md#Scalar-indexing-is-disallowed) in the GPU
guide). Do not use them after `finalize_simulation`, which frees them.

Treat the device arrays as read-only, with one exception: to write `BOUND` on the
device, open it with [`ModflowGPU.device_bound!`](@ref) between `prepare_time_step` and
`prepare_solve`. It returns the same array, and `prepare_solve` then checks the records in
force on the device (all values finite; for RIV stage ≥ bottom and conductance ≥ 0) and
throws an `MF6InputError` naming the first bad record. Values written this way are used
until new period data replace them, and are not seen by host reads of `BOUND`
(`get_value`, mirrors), which keep the values last set on the host.

### Which one to use

| Use | When | Cost per call or step |
| :--- | :--- | :--- |
| `get_value`, `set_value` | reading or setting a few arrays now and then; any point of the loop | one copy of the array per call |
| `get_value_ptr` | code ported from libmf6 that keeps pointers; arrays read every step | `X`, `SIMVALS`, ...: three copies per step; `BOUND`, `NODELIST`: only when they change |
| `get_value_ptr(...; device = true)`, `device_bound!` | the other model runs on the same GPU | none; you handle precision `T` and offsets |

## Example: a river stage from a Julia function

A small steady-state model: a strip of 10 × 5 cells between two constant-head columns,
with a river across the middle column. There are three stress periods of 10 days; the
river stage of each period comes from a Julia function, and the river flow is read back
after each step.

```jldoctest stepping
julia> using ModflowGPU

julia> dis = build_dis(; delr = 100.0, delc = 100.0, top = 0.0, botm = fill(-20.0, 10, 5, 1));

julia> chd = build_boundary(:CHD, dis;
           periods = [1 => (; cells = [(1, i, j) for i in 1:5 for j in (1, 10)], values = fill(-1.0, 10))]);

julia> riv = build_boundary(:RIV, dis; name = "river",
           periods = [1 => (; cells = [(1, i, 5) for i in 1:5], values = repeat([-1.0 50.0 -4.0], 5))]);

julia> gwf = build_model("flow"; dis, npf = build_npf(dis; k = 10.0), ic = build_ic(dis; strt = -1.0),
           boundaries = [chd, riv]);

julia> spec = build_simulation(gwf; tdis = build_tdis(; perlen = fill(10.0, 3), time_units = :days));

julia> sim = initialize(spec; precision = :double, write_output = false)
┌ Info: IMS preconditioner ILU0 is replaced by mc_ilu0 (multicolour ILU0: red–black on DIS grids, greedy colouring on DISV and DISU; the default, see the `preconditioner` option)
└   ims.file = "flow.ims"
Simulation("FLOW", T = Float64, step 0 of 3, initialized)

julia> filter(contains("RIVER"), var_addresses(sim))
5-element Vector{String}:
 "FLOW/RIVER/BOUND"
 "FLOW/RIVER/NODELIST"
 "FLOW/RIVER/SIMVALS"
 "FLOW/RIVER/NBOUND"
 "FLOW/RIVER/MAXBOUND"
```

The stage falls 5 cm per day. The `BOUND` mirror is written after `prepare_time_step`,
when the time of the step is known, and sent to the device by `do_time_step`:

```jldoctest stepping
julia> river_stage(t) = -1.0 - t / 20;   # m, t in days

julia> bound = get_value_ptr(sim, "FLOW/RIVER/BOUND");   # (3, 5): stage, conductance, bottom

julia> while get_current_time(sim) < get_end_time(sim)
           prepare_time_step(sim)
           t = get_current_time(sim)
           bound[1, :] .= river_stage(t)
           do_time_step(sim) || error("time step did not converge")
           finalize_time_step(sim)
           q = sum(get_value(sim, "FLOW/RIVER/SIMVALS"))   # m³/d, negative: into the river
           println("t = $t d: stage $(river_stage(t)) m, river flow $(round(q; digits = 1)) m³/d")
       end
t = 10.0 d: stage -1.5 m, river flow -80.4 m³/d
t = 20.0 d: stage -2.0 m, river flow -160.7 m³/d
t = 30.0 d: stage -2.5 m, river flow -241.1 m³/d
```

The device form of the heads is the offset from `HREF`, in the precision of the
simulation:

```jldoctest stepping
julia> x = get_value_ptr(sim, "FLOW/X"; device = true);

julia> eltype(x)
Float64

julia> get_value(sim, "FLOW/X") ≈ get_value(sim, "FLOW/HREF") .+ Array(x)
true

julia> finalize_simulation(sim);
```

## ModflowCoupling

ModflowCoupling couples MODFLOW 6 to the mGV hydrology model. Its patched version (in
`patches/modflow-coupling/` of the ModflowGPU repository) has a GPU engine built on this
API, and is the production example of it:

- `[solver] engine = "gpu"` (`src/gpu_engine.jl`) builds the model in memory with the
  builder functions and runs each period with `prepare_time_step`, `prepare_solve`,
  `ModflowGPU.solve`, `finalize_solve` and `finalize_time_step` (`advance!`), the same
  call sequence as its libmf6 engine;
- with `coupling = "host"`, the RCH, RIV and WEL values of the period are written into
  the `BOUND` host mirrors after `prepare_time_step` (river stages as absolute heads),
  and heads and `SIMVALS` are read from mirrors, as the libmf6 engine does with libmf6's
  arrays;
- with `coupling = "device"` (`src/device_coupling.jl`), the stresses are written into
  the device `BOUND` arrays by KernelAbstractions kernels, opened with
  [`ModflowGPU.device_bound!`](@ref), with river stages computed in offset form; the
  outputs are computed in kernels from the device heads and `SIMVALS`. A period then
  moves no data between host and device except what the caller reads or sets on the
  host.
