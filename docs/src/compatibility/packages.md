# Supported packages

ModflowGPU runs MODFLOW 6 groundwater flow (GWF) models: one IMS solution with one GWF
model, or with several GWF models connected by GWF–GWF exchanges (see
[Several models and GWF–GWF exchanges](@ref)). This page lists what it reads and runs.
The tables in the first three sections are generated from the checks in the code when
this manual is built, so they are always current.

Reading is strict: anything outside these tables stops the run with an
[`MF6UnsupportedError`](@ref) that names the file, the package and the option, before
anything runs. No option is silently ignored (see
[Getting started](@ref "Unsupported input")). A few options are accepted but behave
differently from MODFLOW 6; they are listed on
[Differences from MODFLOW 6](differences.md).

## Packages and options

Each row is one block of one input file, with the options ModflowGPU accepts in it. Where
an option selects a method, only the values in brackets are accepted. Grid data, initial
conditions, package data and stress-period data (`GRIDDATA`, `PACKAGEDATA`, `PERIOD` and
similar blocks) are read as MODFLOW 6 reads them, including `OPEN/CLOSE` files, binary
arrays and lists, `LAYERED` arrays, `CONSTANT` and `FACTOR`.

```@eval
Main.CompatTables.options_table()
```

Notes:

- `TS6`, `TAS6` and `OBS6` are `FILEIN` options of the stress packages. Time series are
  supported (see below), and so are observations (see [Observations (OBS6)](@ref)).
- `PRINT_INPUT`, `PRINT_FLOWS`, `MEMORY_PRINT_OPTION` and `PROFILE_OPTION` are accepted
  and have no effect on the output (see the listing file on
  [Differences from MODFLOW 6](@ref "Listing file")).
- `SAVE_SPECIFIC_DISCHARGE` writes the `DATA-SPDIS` record to the budget file, as
  MODFLOW 6 does.
- IMS: the ILU preconditioner options (`PRECONDITIONER_LEVELS`,
  `PRECONDITIONER_DROP_TOLERANCE`, `RELAXATION_FACTOR`) are accepted, but ModflowGPU uses
  its own preconditioner (see [Differences from MODFLOW 6](differences.md)). The
  backtracking options (`BACKTRACKING_NUMBER`, `_TOLERANCE`, `_REDUCTION_FACTOR`,
  `_RESIDUAL_LIMIT`, and the backtracking that `COMPLEXITY COMPLEX` turns on) are applied
  as in MODFLOW 6.
- In OC, `HEAD` and `BUDGET` stand for `HEAD FILEOUT`, `BUDGET FILEOUT` and
  `HEAD PRINT_FORMAT` (accepted, not applied). The `PERIOD` blocks accept `SAVE` and
  `PRINT` of `HEAD` and `BUDGET` with `ALL`, `FIRST`, `LAST`, `FREQUENCY` and `STEPS`.
- A model can have several packages of one type (for example three RCH packages).

## Several models and GWF–GWF exchanges

A simulation can have several GWF models in its IMS solution, connected by `GWF6-GWF6`
exchanges, as in MODFLOW 6: a parent and a refined child grid (local grid refinement),
neighbouring regions, or a model connected to itself (a periodic boundary, such as the
two sides of a global grid along the date line). The models are solved together as one
system, as MODFLOW 6 does; each writes its own head, budget and listing files, and has
its own variables in the stepping API and BMI (`"<model>/X"`, see
[Stepping API](@ref "Several models")).

- Exchange options: `AUXILIARY` (`ANGLDEGX` and `CDIST` as MODFLOW 6 uses them),
  `BOUNDNAMES`, `SAVE_FLOWS` (the exchange flows in each model's budget file),
  `CELL_AVERAGING` (`HARMONIC`, `LOGARITHMIC`, `AMT-LMK`), `XT3D` and `OBS6 FILEIN`
  (observations of the exchange flows), `PRINT_FLOWS` (the flow table of the exchange in
  each model's listing file). `PRINT_INPUT` is accepted and prints nothing; `VARIABLECV`, `DEWATERED` and `NEWTON`
  are accepted and, as in MODFLOW 6, change nothing between confined cells.
- Each model's budget has MODFLOW 6's exchange terms: `FLOW-JA-FACE` (the flow across the
  exchange) and `FLOW-JA-FACE-CHD` (the part into constant-head cells), with the
  exchange's name (`GWF-GWF_1`, ...) as the package name.
- `XT3D` on the exchange gives the same coefficients as MODFLOW 6's interface model,
  also next to models with or without XT3D themselves.
- The models must be confined (`ICELLTYPE` 0, storage without `ICONVERT`, no `NEWTON`) and
  in one solution; models with storage must agree on the steady-state and transient
  periods. Ghost nodes (`GNC6`) and movers (`MVR6`) on exchanges are not supported.
- Exchange observations (`OBS6 FILEIN` in the exchange file) are written as MODFLOW 6 does
  (see [Observations (OBS6)](@ref)).

## Observations (OBS6)

Observation files (`OBS6`) are read for the model (`OBS6` in the name file) and for every
stress package (`OBS6 FILEIN` in its OPTIONS), and their output is written as MODFLOW 6
writes it: one row per time step, CSV (`time` and one column per observation, the names in
upper case, the values with 16 significant digits or `DIGITS`) or `BINARY`. The
observation types are MODFLOW 6's:

| Owner | Types | ID |
| :-- | :-- | :-- |
| Model (`OBS6` in the name file) | `HEAD`, `DRAWDOWN` (`STRT` minus head) | a cellid |
| | `FLOW-JA-FACE` (flow from the second cell into the first) | two cellids |
| `CHD`, `WEL`, `DRN`, `RIV`, `GHB`, `RCH`, `EVT` | the package type (`CHD`, `WEL`, ...): the flow of the package, positive into the model | a cellid, or a boundary name (`BOUNDNAMES`) |
| `WEL` | `WEL-REDUCTION`: the rate taken off by `AUTO_FLOW_REDUCE` | a cellid or a boundary name |
| GWF–GWF exchange (`OBS6 FILEIN` in its OPTIONS) | `FLOW-JA-FACE`: the flow across the exchange, positive into the cell of the first model | an exchange number (its line in `EXCHANGEDATA`), or a boundary name (the sum over its connections) |
| `WEL`, `DRN`, `RIV`, `GHB` | `TO-MVR` | accepted; always `3.0e30` (MVR is not supported) |

A package observation sums the records that match it (all records of a boundary name,
or every record in the cell), as MODFLOW 6 does; when no record matches in a time step,
the value is `3.0e30` ([`ModflowGPU.OBSNODATA`](@ref)). The options `DIGITS` and
`PRINT_INPUT` are accepted (`PRINT_INPUT` writes nothing, see the listing file on
[Differences from MODFLOW 6](@ref "Listing file")). [`read_observations`](@ref) reads the
output files of ModflowGPU and of MODFLOW 6.

## Model features

Besides options, the input is checked for features that change how the model is solved.
These are supported:

```@eval
Main.CompatTables.features_table()
```

Both the standard formulation of convertible cells and wetting and drying (`REWET` with
`WETDRY`, `WETFCT`, `IWETIT`, `IHDWET`) are supported, as are `THICKSTRT`, `VARIABLECV`
(with `DEWATERED`) and `PERCHED`.

The Newton–Raphson formulation (`NEWTON` in the options of the model name file) is
supported as in MODFLOW 6:

- the saturation of convertible cells is the quadratic smoothing of MODFLOW 6 (`SATOMEGA`
  1e-6), horizontal conductances use the saturation of the upstream cell, cells do not dry
  (a cell whose head falls below its bottom keeps that head and has no outflow), and the
  matrix gets the Newton terms of NPF, STO (convertible storage), WEL with
  `AUTO_FLOW_REDUCE` and DRN with `AUXDEPTHNAME` (which then scales the conductance with
  MODFLOW 6's cubic function);
- NPF `HIGHEST_CELL_SATURATION` is supported; `PERCHED`, `VARIABLECV` and `REWET` are input
  errors with `NEWTON`, as in MODFLOW 6;
- `NEWTON UNDER_RELAXATION` limits heads that fall below the bottom of the lowest
  convertible cell;
- in steady-state stress periods the pseudo-transient continuation of IMS is applied
  (turned off with the IMS option `NO_PTC ALL`, or `NO_PTC FIRST` for the first period);
- the matrix is not symmetric: IMS must use `LINEAR_ACCELERATION BICGSTAB` (the default of
  `COMPLEXITY MODERATE` and `COMPLEX`); `CG` is an input error, as in MODFLOW 6.

Backtracking with Newton–Raphson works on the system without the Newton terms, as in
MODFLOW 6. How the results compare with MODFLOW 6 is on
[Differences from MODFLOW 6](@ref "Newton–Raphson") and [Comparing results](results.md).

The XT3D formulation of NPF (the NPF option `XT3D`, with or without `RHS`) is supported for
models whose cells are all confined (`ICELLTYPE` 0), on DIS, DISV and DISU grids (DISU with
vertices and `ANGLDEGX`, as MODFLOW 6 requires):

- the flow across each face uses the full conductivity tensor (`K`, `K22`, `K33` and
  `ANGLE1` to `ANGLE3`; `ANGLE1` is 0 when not given) and the heads of the neighbours of
  both cells, as in MODFLOW 6; the coefficients are computed once, as MODFLOW 6 does for
  permanently confined cells;
- without `RHS` the matrix holds every XT3D term, so a row also has entries for the
  neighbours of the cell's neighbours; with `RHS` the matrix holds only the terms of the
  direct connection and the other terms enter through the residual, which takes more outer
  iterations but gives the same heads;
- the matrix is not symmetric: IMS must use `LINEAR_ACCELERATION BICGSTAB`; `CG` is an input
  error, as in MODFLOW 6;
- XT3D with convertible cells or with `NEWTON` is rejected (see below).

Horizontal flow barriers (HFB6) are supported on DIS, DISV and DISU grids, as in MODFLOW 6:

- each barrier lies on the connection between two cells (`CELLID1`, `CELLID2`; on DISU the
  user node numbers) and has a hydraulic characteristic `HYDCHR` (the barrier's hydraulic
  conductivity divided by its thickness); a zero or negative `HYDCHR` multiplies the
  conductance of the connection by `-HYDCHR`;
- the barriers of a `PERIOD` block apply from that stress period on, until the next block
  replaces them (an empty block removes them);
- between confined cells, and in every connection with `NEWTON`, the barrier changes the
  saturated conductance once per stress period; next to a convertible cell (`ICELLTYPE` 1)
  of the standard formulation the barrier is combined with the head-dependent conductance
  in every outer iteration, with the saturated thickness of the face, as MODFLOW 6 does;
- a barrier with a cell outside the active grid is dropped with a warning; a barrier
  between cells that are not connected, or two barriers on one connection, are input
  errors, as in MODFLOW 6;
- HFB6 has no budget term: the barriers change the flows between cells (`FLOW-JA-FACE`).

HFB6 with `XT3D`, and in a model of a multi-model solution, is rejected (see below).

## Packages and models that are rejected

These file types always give an [`MF6UnsupportedError`](@ref), also when reading with
`strict = false`:

```@eval
Main.CompatTables.rejected_table()
```

## Out of scope, and why

ModflowGPU is built to run large groundwater flow models on a GPU, coupled to land-surface
and hydrology models that compute the surface water themselves. That decides what is in
scope. These parts of MODFLOW 6 will not be supported:

| Feature | Why it is out of scope |
| :-- | :-- |
| SFR, LAK, MAW, UZF (streams, lakes, multi-aquifer wells, unsaturated zone) | Their own nonlinear equations and sequential routing (streams downstream, lake stages, well heads) do not map well onto a GPU. In the coupled models ModflowGPU is made for, the hydrology model handles rivers, lakes and the unsaturated zone, and exchanges water with ModflowGPU through RIV, DRN, RCH and EVT. |
| MVR (water mover) | Moves water between the packages above. |
| CSUB (compaction and subsidence) | A separate physics package, not needed for regional and global flow. |
| BUY, VSC (density and viscosity) | Only meaningful together with a transport model. |
| GWT, GWE, PRT (transport, energy, particle tracking) | ModflowGPU solves groundwater flow only. |
| API package | The stepping API and BMI of ModflowGPU give access to the model from Julia instead. |
| PETSc and parallel input (`HPC6`) | ModflowGPU runs on one GPU with its own linear solver. |

## Not supported yet

These MODFLOW 6 features are in scope but not supported yet. A model that uses them is
rejected with an [`MF6UnsupportedError`](@ref), except where the status says that the
input is accepted.

| Feature | Status |
| :-- | :-- |
| XT3D with convertible cells (`ICELLTYPE` ≠ 0) or with `NEWTON` | not planned at the moment |
| More than one solution; several models with convertible cells or `NEWTON` | not planned at the moment |
| Horizontal flow barriers (HFB6) with `XT3D`, or in a model of a multi-model solution | not planned at the moment |
| Ghost-node correction (GNC6) | not planned at the moment |
| Time-varying hydraulic conductivity and storage (TVK6, TVS6) | not planned at the moment |
| Adaptive time stepping (ATS6) | not planned at the moment |
| CSV output: OC `BUDGETCSV`, IMS `CSV_OUTER_OUTPUT` and `CSV_INNER_OUTPUT`, WEL `AUTO_FLOW_REDUCE_CSV` | not planned at the moment |
| IMS `SCALING_METHOD L2NORM`, `REORDERING_METHOD RCM` and `MD` | not planned at the moment |
| `READARRAYGRID` input of list packages | not planned at the moment |

The tested models are listed on [Comparing results](results.md).
