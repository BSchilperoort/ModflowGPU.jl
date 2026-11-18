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
use their own ordering: `REORDERING_METHOD RCM` and `MD` are rejected (with
`strict = false` they have no effect, with an `Info`), and `RED_BLACK_ORDERING` is
accepted and has no effect.

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
  In Float32 the quadratic smoothing interval of the saturation next to a cell bottom
  (1e-6 of the cell thickness) is only a few rounding units wide.

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
| Budget file (`.cbc`) | written as OC asks | the same records, except `DATA-SPDIS` |
| `DATA-SPDIS` (specific discharge, NPF `SAVE_SPECIFIC_DISCHARGE`) | written | not written (an `Info` at the start of the run) |
| Binary grid file (`.dis.grb`, `.disv.grb`, `.disu.grb`) | written unless `NOGRB` | not written; MODFLOW 6's file can be read with [`read_grb`](@ref) |
| Model listing file (`.lst`) | full report | a short version, see below |
| Simulation listing file (`mfsim.lst`) | written | not written |
| Observation output (OBS6) | written | not written |
| CSV files (`BUDGETCSV`, `CSV_OUTER_OUTPUT`, `CSV_INNER_OUTPUT`, `AUTO_FLOW_REDUCE_CSV`) | written | the options are rejected |

### Listing file

ModflowGPU writes a short model listing file with:

- a header with the backend, the precision and the solver settings in use;
- per time step the number of outer and inner iterations, a message when the step did
  not converge, and the time step length and times;
- the budget tables, in MODFLOW 6's format, when OC prints the budget, at the end of every
  stress period and after a step that did not converge (as MODFLOW 6 does), so they can be
  read with [`read_listing_budgets`](@ref) like MODFLOW 6's;
- the heads, when OC prints them, 10 values per line in `%12.4E`; `HEAD PRINT_FORMAT` is
  not applied.

It does not contain the echo of the input (`PRINT_INPUT`), the flow tables of
`PRINT_FLOWS`, the solver convergence tables of IMS `PRINT_OPTION`, messages about cells
that become dry or wet, the memory and timing reports, or the warnings and notes that
MODFLOW 6 writes there. ModflowGPU reports warnings and notes in the Julia log instead.

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
  below the bottom of a convertible cell; conductances not negative) run only when period
  data are read. MODFLOW 6 repeats them at the start of every time step (unless
  `NOCHECK`), so it also checks values written through its API; ModflowGPU does not check
  values set with `set_value` or `get_value_ptr`. `BOUND` arrays written on the device
  ([`ModflowGPU.device_bound!`](@ref)) get a reduced check at `prepare_solve`: finite
  values and, for RIV, stage not below bottom and conductance not negative.
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
