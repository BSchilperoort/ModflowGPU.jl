# Supported packages

ModflowGPU runs MODFLOW 6 groundwater flow (GWF) models: one GWF model with one IMS
solution. This page lists what it reads and runs. The tables in the first three sections
are generated from the checks in the code when this manual is built, so they are always
current.

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
  supported (see below). An observation file must exist, but it is not read and no
  observation output is written yet.
- `PRINT_INPUT`, `PRINT_FLOWS`, `MEMORY_PRINT_OPTION` and `PROFILE_OPTION` are accepted
  and have no effect on the output (see the listing file on
  [Differences from MODFLOW 6](@ref "Listing file")).
- `SAVE_SPECIFIC_DISCHARGE` is accepted, but the `DATA-SPDIS` record is not written to the
  budget file.
- IMS: the ILU preconditioner options (`PRECONDITIONER_LEVELS`,
  `PRECONDITIONER_DROP_TOLERANCE`, `RELAXATION_FACTOR`) and the backtracking options are
  accepted, but ModflowGPU uses its own preconditioner and does not backtrack (see
  [Differences from MODFLOW 6](differences.md)).
- In OC, `HEAD` and `BUDGET` stand for `HEAD FILEOUT`, `BUDGET FILEOUT` and
  `HEAD PRINT_FORMAT` (accepted, not applied). The `PERIOD` blocks accept `SAVE` and
  `PRINT` of `HEAD` and `BUDGET` with `ALL`, `FIRST`, `LAST`, `FREQUENCY` and `STEPS`.
- A model can have several packages of one type (for example three RCH packages).

## Model features

Besides options, the input is checked for features that change how the model is solved.
These are supported:

```@eval
Main.CompatTables.features_table()
```

Both the standard formulation of convertible cells and wetting and drying (`REWET` with
`WETDRY`, `WETFCT`, `IWETIT`, `IHDWET`) are supported, as are `THICKSTRT`, `VARIABLECV`
(with `DEWATERED`) and `PERCHED`. The Newton–Raphson formulation is not (see below).

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
rejected with an [`MF6UnsupportedError`](@ref).

| Feature | Status |
| :-- | :-- |
| Newton–Raphson formulation (`NEWTON` in the model name file) | planned |
| Backtracking (IMS `BACKTRACKING_NUMBER`) | planned; accepted now but not applied |
| XT3D (NPF `XT3D`) | planned |
| More than one model or solution, GWF–GWF exchanges | planned |
| Observation output (OBS6) | planned; the files are accepted, no output is written |
| Horizontal flow barriers (HFB6) | not planned at the moment |
| Ghost-node correction (GNC6) | not planned at the moment |
| Time-varying hydraulic conductivity and storage (TVK6, TVS6) | not planned at the moment |
| Adaptive time stepping (ATS6) | not planned at the moment |
| CSV output: OC `BUDGETCSV`, IMS `CSV_OUTER_OUTPUT` and `CSV_INNER_OUTPUT`, WEL `AUTO_FLOW_REDUCE_CSV` | not planned at the moment |
| Specific discharge in the budget file (`DATA-SPDIS`) | not planned at the moment; the option is accepted |
| IMS `SCALING_METHOD L2NORM`, `REORDERING_METHOD RCM` and `MD` | not planned at the moment |
| `READARRAYGRID` input of list packages | not planned at the moment |

The tested models are listed on [Comparing results](results.md).
