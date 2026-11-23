# Differences from MODFLOW 6

ModflowGPU reads MODFLOW 6 input and writes MODFLOW 6 output, and its heads and budgets
agree with MODFLOW 6 within the closure criteria ([Comparing results](results.md)). This
page lists what you will notice that is different: options that are accepted but work
differently, output that is not written, and behaviour of the run and of the stepping
API. Unsupported input is on [Supported packages](packages.md).

## Linear solver

### The preconditioner is replaced

MODFLOW 6's ILU0, MILU0 and ILUT preconditioners work through the matrix one row after
another, which does not run in parallel on a GPU. ModflowGPU replaces them with its own,
chosen with the `preconditioner` keyword of [`run_simulation`](@ref) and
[`initialize`](@ref):

- `:auto` (the default) is `:mc_ilu0`: ILU0 in a multicolour order (red–black on DIS
  grids, a greedy colouring on DISV and DISU grids), so that each colour runs in parallel;
- `:amg` (smoothed-aggregation algebraic multigrid), `:jacobi`, `:chebyshev` and
  `:mc_sgs` (multicolour symmetric Gauss–Seidel).

The IMS options that configure MODFLOW 6's preconditioner (`PRECONDITIONER_LEVELS`,
`PRECONDITIONER_DROP_TOLERANCE`, `RELAXATION_FACTOR`, and the ones that `COMPLEXITY`
sets) are accepted, but their values are not used. The log says so when the run starts:
an `Info` for ILU0 and for settings that come from `COMPLEXITY` alone, a `Warning` when
you set `RELAXATION_FACTOR` or `PRECONDITIONER_LEVELS` yourself. MILU relaxation can be
switched on with `precond_options = (; relax = 0.97)`. The multicolour preconditioners
use their own ordering: `REORDERING_METHOD RCM` and `MD` and `RED_BLACK_ORDERING`
are accepted and have no effect (for `REORDERING_METHOD` an `Info` says so). In
MODFLOW 6 they only change the order of the ILU factorisation, so the number of
iterations, not the converged heads.

The convergence tests of IMS (`INNER_DVCLOSE`, `INNER_RCLOSE` with its `STRICT`,
`L2NORM_RCLOSE` and `RELATIVE_RCLOSE` variants, `OUTER_DVCLOSE`) are the same, so the
converged heads agree, but **the number of inner iterations differs**. The red–black
ILU0 is a weaker preconditioner than MODFLOW 6's ILU0 in natural order and usually needs
more, cheaper, iterations; the [GPU guide](../gpu.md) helps to choose one.

### Other differences in the linear solve

- Each linear solve computes the head change from a zero starting guess (MODFLOW 6 starts
  from the current heads). The residuals, the convergence tests and the updates are the
  same quantities, so only iteration counts are affected.
- The stagnation tests of CG and BiCGSTAB guard their divisions with a relative instead of
  an absolute tolerance. With the absolute guard, solves with tight closures could stall
  and run to `INNER_MAXIMUM` in every outer iteration; MODFLOW 6 can do that on such
  models, ModflowGPU stops earlier.
- When an ILU pivot is too small, ModflowGPU keeps the unfactorised diagonal; MODFLOW 6
  restarts the factorisation with a larger shift.
- With the Newton–Raphson formulation, BiCGSTAB restarts with a new shadow residual when
  its `rho` vanishes at the rounding level (a breakdown). MODFLOW 6 has no restart. It
  happens when the residual is nonzero in a few cells only, for example in the first outer
  iteration of a Newton–Raphson steady state with the red–black ILU0 (`ex-gwf-zaidel`),
  where the solve would otherwise diverge. Other models do not restart, as in MODFLOW 6.

## Outer iterations

- **Iteration counts differ.** Because of the preconditioner, the
  number of outer iterations can differ by a few, and the inner counts differ more. In
  `:single` the outer loop stops at the Float32 closures, often in fewer outer iterations
  than MODFLOW 6.
- **A time step that does not converge** stops the run with a
  [`ConvergenceError`](@ref) after that step's output is written (MODFLOW 6 writes the
  output and stops with an error message). With the `CONTINUE` option in `mfsim.nam`
  ModflowGPU warns and continues, as MODFLOW 6 does.

## Newton–Raphson

The Newton–Raphson formulation, its pseudo-transient continuation, `NEWTON
UNDER_RELAXATION` and backtracking follow MODFLOW 6 (the first matrix of a run equals
libmf6's), but:

- **Heads of dry cells differ.** A cell whose head is below its bottom has no saturation
  and no outflow, and any head between the highest wet neighbour and the bottom solves its
  equation. MODFLOW 6 and ModflowGPU stop at different such heads (for example 1e-8 m and
  1e-6 m below the bottom). Both are below the bottom, and the flows are the same.
- **Iteration counts differ more** than with the standard formulation: the iterations of
  a Newton steady state depend on the preconditioner through the pseudo-transient
  continuation (`ex-gwf-nwt-p03b`: 96 outer iterations, MODFLOW 6 159).
- **`:single` may not converge** on hard Newton problems: `ex-gwf-nwt-p03b` does not
  converge in Float32, while `:mixed` and `:double` do ([Comparing results](results.md)).
  Its water table lies a fraction of a millimetre above the cell bottoms in most cells,
  20 m below the starting heads: in Float32 those saturated thicknesses carry only 2–3
  significant digits.

## XT3D

The XT3D coefficients and the matrix of the first solve equal libmf6's (to 1e-15), but:

- **With the `RHS` option** MODFLOW 6 moves the XT3D terms of the neighbours to the
  right-hand side with the current heads. ModflowGPU keeps them in the residual (it solves
  for head corrections), with the same matrix of the direct terms: the converged heads are
  the same. The outer iteration counts differ, and they depend on the inner solves rather
  than on where the neighbour terms are: ex-gwf-hanix with `XT3D RHS` takes 409 outer
  iterations with the default preconditioner (MC-ILU0), 164 with `preconditioner = :amg`,
  and MODFLOW 6 183 (ILU0).
- **The matrix couples each cell to the neighbours of its neighbours**, so on a DIS grid
  the multicolour ILU0 preconditioner uses more colours than the red–black ordering of
  the standard formulation.
- **Heads defined up to a constant.** A steady-state model whose only boundaries are
  wells and recharge (ex-gwf-whirl) has a singular matrix: adding a constant to every
  head changes no flow. MODFLOW 6 and ModflowGPU end at heads that differ by such a
  constant (ex-gwf-whirl: 8e-6 m); the flows are the same. This holds with or without
  XT3D.

## Several models and exchanges

The models of a solution are solved as one system with MODFLOW 6's matrix (the same
entries to 1e-15 on the examples), with these differences in how it is built and
reported:

- **XT3D on an exchange.** MODFLOW 6 builds an interface model for each side of the
  exchange (the cells near it, in global coordinates) and takes the XT3D coefficients of
  the exchange connections, and of the connections next to them, from it. ModflowGPU
  computes the same coefficients directly on the grid of all models, for the same
  connections: the result is the same, but there is no interface model (and no
  `GWFIM*`/`GWFCON*` memory to read).
- **Preconditioner colouring.** The multicolour preconditioners colour each DIS model
  red–black and keep that when every exchange connection joins two colours; otherwise
  (for example a refined child grid) they colour the whole solution greedily, with more
  colours.
- **Output.** Each model's listing file has the budget table with MODFLOW 6's exchange
  terms and, with the exchange's `PRINT_FLOWS`, its flow table (`EXG GWF-GWF_1 ...`, in
  the steps whose budget OC saves, after the step's summary); the exchange rate summary
  that MODFLOW 6 writes to `mfsim.lst` is not written (there is no `mfsim.lst`). The budget file of a model with an exchange with
  `SAVE_FLOWS` has MODFLOW 6's `FLOW-JA-FACE` list of the exchange.
- **Scope.** Only confined models (`ICELLTYPE` 0, no `ICONVERT`, no `NEWTON`) in one
  solution; `MVR6` on exchanges is not supported (`GNC6` on exchanges is, see
  [Ghost-node correction (GNC6)](@ref)).

## Precision

The `:double` and `:mixed` presets give MODFLOW 6's accuracy. `:single` computes in
Float32, which MODFLOW 6 cannot:

- **Closures are raised** to what Float32 can resolve: `OUTER_DVCLOSE` and
  `INNER_DVCLOSE` to at least 1e-6 m (an `Info` message says when this happens),
  `OUTER_DVCLOSE` also to the resolution of the heads of the model (with heads 1000 m
  from their reference that is about 1e-4 m; at most 1 cm), and an absolute `INNER_RCLOSE` to the
  rounding level of the residual. Once a
  step's head change meets `OUTER_DVCLOSE` but its linear solve did not converge, the
  inner `INNER_DVCLOSE` is raised to the head resolution for the rest of that step.
- **Results differ from MODFLOW 6** by up to about 1e-4 m in heads, and budgets by
  about 1e-6 to 1e-4 of the total inflow; where conductances or storage coefficients are
  very large, single flows differ by more ([Comparing results](@ref "Acceptance limits")).
- **Some models need another preconditioner** in `:single` (`preconditioner = :amg`),
  see the [GPU guide](../gpu.md).

## Output files

| Output | MODFLOW 6 | ModflowGPU |
| :-- | :-- | :-- |
| Head file (`.hds`) | written as OC asks | the same; dry cells have `HDRY` |
| Budget file (`.cbc`) | written as OC asks | the same records |
| `DATA-SPDIS` (specific discharge, NPF `SAVE_SPECIFIC_DISCHARGE`) | written | the same (computed on the device from the step's flows; on the host for models with convertible cells and for several models in one solution) |
| Binary grid file (`.dis.grb`, `.disv.grb`, `.disu.grb`) | written unless `NOGRB` | the same file (version 1; a `CRS` option is not written to it) |
| Model listing file (`.lst`) | full report | a short version, see below |
| Simulation listing file (`mfsim.lst`) | written | not written |
| Observation output (OBS6) | written every time step | the same files, CSV or `BINARY`; values as MODFLOW 6 prints them (16 significant digits) |
| CSV files (`BUDGETCSV`, `CSV_OUTER_OUTPUT`, `CSV_INNER_OUTPUT`, `AUTO_FLOW_REDUCE_CSV`) | written | the same files, see below |

### Listing file

ModflowGPU writes a short model listing file with:

- a header with the backend, the precision and the solver settings in use;
- per time step the number of outer and inner iterations, a message when the step did
  not converge, and the time step length and times;
- the budget tables, in MODFLOW 6's format, when OC prints the budget, at the end of every
  stress period and after a step that did not converge (as MODFLOW 6 does), so they can be
  read with [`read_listing_budgets`](@ref) like MODFLOW 6's;
- the heads, when OC prints them, 10 values per line in `%12.4E`; `HEAD PRINT_FORMAT` is
  not applied;
- the cell conversions of a model with convertible cells (`CELL CONVERSIONS FOR ITER.=...`,
  the cells that became dry or wet), as MODFLOW 6 writes them, before the iteration count
  of the step (those of the starting heads after the header).

It does not contain the echo of the input (`PRINT_INPUT`), the flow tables of
`PRINT_FLOWS`, the solver convergence tables of IMS `PRINT_OPTION`, the memory and timing reports, or the warnings and notes that
MODFLOW 6 writes there. ModflowGPU reports warnings and notes in the Julia log instead.

### Observation files

The observation files have MODFLOW 6's header, times and number format, with these
differences:

- an observation value of zero is written as `0.000000000000000`;
- `RCH` and `EVT` observations by cell, in a model with convertible cells: at the start of
  a stress period that reads new recharge (evapotranspiration) data, MODFLOW 6 matches the
  records in the cells given in the input, ModflowGPU in the highest active cell the record
  was moved to. The values differ only when that cell (usually in layer 1) is dry at the
  start of the period; observations by boundary name are not affected;
- the observation files of GWF–GWF exchanges are written after those of the models.

### CSV files

The CSV files (`BUDGETCSV`, `CSV_OUTER_OUTPUT`, `CSV_INNER_OUTPUT`,
`AUTO_FLOW_REDUCE_CSV`) have MODFLOW 6's headers, columns and number format. Their values
follow from ModflowGPU's solution:

- the budget and well-reduction values match MODFLOW 6's within the budget differences of
  the run (in `:single`, a well whose reduction is at the Float32 rounding level may have a
  row in one file and not in the other);
- the outer-iteration file has the same rows where the outer iterations are the same (in
  `:double`); its inner-iteration counts, and the whole inner-iteration file, are those of
  ModflowGPU's linear solver, whose preconditioner differs from MODFLOW 6's (see
  [The preconditioner is replaced](@ref)): other counts, changes, residuals and step
  lengths. As in MODFLOW 6 the residual is the scaled one for `CG` and the unscaled one
  for `BICGSTAB`; after an iteration that recomputes the residual
  (`NUMBER_ORTHOGONALIZATIONS`) it is the recomputed one;
- where several cells have the same largest change to the rounding level (a symmetric
  model), the cell reported may differ;
- the package column of the outer-iteration file is always empty (no supported package
  has its own convergence test);
- `CSV_INNER_OUTPUT` reads two values per model from the device in every inner iteration,
  which slows down the linear solve, most on a GPU; the other files add no device work.

## Stepping API and BMI

The stepping API works as libmf6 and its XMI/BMI interface do, with these differences:

- [`get_value_ptr`](@ref) returns a host array, a Float64 mirror of the device data that
  is copied at the calls of the time-step loop, not a pointer into the model's memory.
  Heads in it are absolute heads. `get_value_ptr(...; device = true)` returns the device
  array in the precision of the run, with heads as offsets from a reference head per cell.
- Values written through `get_value_ptr` are sent to the device at the next
  `prepare_time_step` or `prepare_solve`. A write between `prepare_solve` and
  `finalize_solve` is discarded with a warning at `finalize_solve`; use
  [`set_value`](@ref) to change a value during the outer iterations.
- `BOUND` is the package data before the `AUXMULTNAME` multiplier is applied. For RIV
  ModflowGPU keeps `BOUND` (stage, conductance, bottom), as earlier MODFLOW 6 versions
  and ModflowCoupling do; MODFLOW 6.8 has separate `STAGE`, `COND` and `RBOT` arrays.
- The checks of boundary records (RIV stage and bottom, DRN elevation and GHB head not
  below the bottom of a convertible cell; conductances not negative) run when period
  data are read and, for a package with time series, at every time step on that step's
  series values. MODFLOW 6 repeats them at the start of every time step (unless
  `NOCHECK`), so it also checks values written through its API; ModflowGPU does not check
  values set with `set_value` or `get_value_ptr`. `BOUND` arrays written on the device
  ([`ModflowGPU.device_bound!`](@ref)) get a reduced check at `prepare_solve`: finite
  values and, for RIV, stage not below bottom and conductance not negative. With the
  simulation option `NOCHECK` these checks are skipped, as in MODFLOW 6 (the EVT PXDP
  checks of `NSEG` > 1 still run, and device `BOUND` values are still checked for being
  finite).
- There are two BMIs. [`MF6Model`](@ref) returns what libmf6's BMI returns (one grid per
  model, DIS x and y as indices, libmf6's shapes and type strings), with the differences
  listed under [Differences from libmf6](@ref); where libmf6 returns a failure status it
  throws. The BMI of [`Simulation`](@ref) follows the BMI specification where libmf6's BMI
  does not: settings come from a configuration file, the grid functions give real-world
  coordinates of the cell centres (DIS) or of the vertices (DISV, DISU), every variable
  has a grid (scalars and package arrays too), and the units are UDUNITS strings. See
  [BMI](../coupling/bmi.md).

## Recharge rates and older MODFLOW 6 versions

RCH rates are per unit area, and ModflowGPU multiplies them by the cell area, as MODFLOW 6
does from version 6.4 on. MODFLOW 6.3 multiplies list recharge rates by the cell area
when it reads them; code that writes rates into `RCH/BOUND` through libmf6 6.3 (as
ModflowCoupling did) must write rates per unit area for ModflowGPU.

## Internal differences you may see in the data

- A dry cell keeps its bottom elevation as head inside the engine; the head file, the
  host heads and [`get_value_ptr`](@ref) give `HDRY` (−1e30) like MODFLOW 6.
- In `:single` the storage flows of convertible cells are computed from the clamped heads
  directly (in exact arithmetic the same as MODFLOW 6's formula), which avoids large
  rounding errors when cells are nearly saturated.
