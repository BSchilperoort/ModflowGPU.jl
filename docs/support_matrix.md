# Support matrix (Milestone A)

What ModflowGPU runs at Milestone A (end of Phase 3), derived from the checks in the code,
not from the plan: the strict-loading tables in `src/input/supported.jl`
(`SUPPORTED_OPTIONS`, `SUPPORTED_VALUES`, `SUPPORTED_FEATURES`, `UNSUPPORTED_PACKAGES`,
`UNSUPPORTED_MODELS`) and every `MF6UnsupportedError` / `unsupported(...)` /
`check_feature(...)` site (`grep -rn "MF6UnsupportedError(\|unsupported(\|check_feature(" src`).
Update this file when a card changes one of them.

`load_simulation(dir; strict = true)` (what `ModflowGPU.run(dir)` uses) throws an
`MF6UnsupportedError` ("unsupported: ...") for anything below marked *rejected*: no option
is silently ignored. With `strict = false` the loader reads all in-scope input (Tiers
0–7) into a spec, but the engine still rejects what it cannot run (the engine checks
are marked *engine*). The builder API (`build_*`, T3.2) produces the same specs and goes
through the same engine checks.

## Simulation

| Item | Supported | Rejected (card that adds it) |
| --- | --- | --- |
| Models | one GWF6 model | more than one GWF6 model (T6.6, `:multi_model`); GWT6, GWE6, PRT6, CHF6, OLF6 (out of scope, always) |
| Solutions | one IMS6 solution in one solution group, MXITER | more than one solution (T6.6); EMS6 (out of scope, always) |
| Exchanges | none | GWF6-GWF6 (T6.6, `:exchange`; also *engine*: `initialize` accepts one model and no exchanges); other exchange types (out of scope); GNC6 or MVR6 on an exchange |
| mfsim.nam OPTIONS | CONTINUE, NOCHECK, MEMORY_PRINT_OPTION, PROFILE_OPTION, MAXERRORS, PRINT_INPUT | HPC6 (parallel input) |
| TDIS6 | TIME_UNITS, START_DATE_TIME; PERIODDATA (PERLEN, NSTP, TSMULT) | ATS6 (adaptive time stepping; also *engine*, `time_steps`) |

## IMS6

| Block | Supported | Rejected |
| --- | --- | --- |
| OPTIONS | PRINT_OPTION, COMPLEXITY (SIMPLE, MODERATE, COMPLEX presets) | CSV_OUTER_OUTPUT, CSV_INNER_OUTPUT, NO_PTC, ATS_OUTER_MAXIMUM_FRACTION |
| NONLINEAR | OUTER_DVCLOSE, OUTER_MAXIMUM, UNDER_RELAXATION NONE / SIMPLE / COOLEY / DBD (+ THETA, KAPPA, GAMMA, MOMENTUM), LINEAR_SOLVER DEFAULT / LINEAR | other LINEAR_SOLVER values |
| LINEAR | INNER_MAXIMUM, INNER_DVCLOSE, INNER_RCLOSE (absolute, STRICT, L2NORM_RCLOSE, RELATIVE_RCLOSE), LINEAR_ACCELERATION CG / BICGSTAB, NUMBER_ORTHOGONALIZATIONS, SCALING_METHOD NONE / DIAGONAL, REORDERING_METHOD NONE | SCALING_METHOD L2NORM, REORDERING_METHOD RCM / MD |

Accepted with a different behaviour (logged warnings, not errors):

- Preconditioner: ILU0, MILU0 (RELAXATION_FACTOR > 0) and ILUT / MILUT
  (PRECONDITIONER_LEVELS > 0, PRECONDITIONER_DROP_TOLERANCE) are accepted and replaced by
  the ModflowGPU option `preconditioner` (`ModflowGPU.run` / `initialize` keyword):
  `:auto` (default: `:mc_ilu0`, T4.4; logged as a message for ILU0, a warning for the
  others), `:jacobi`, `:chebyshev`, `:mc_sgs`, `:mc_ilu0` (multicolour ILU0, T4.2), `:amg` (smoothed-aggregation AMG, T4.3), with `precond_options`
  (docs/mf6_notes.md). RED_BLACK_ORDERING is accepted (it only affects MF6's ILU).
- Backtracking (BACKTRACKING_* and COMPLEXITY COMPLEX's implicit backtracking): accepted,
  not applied, with a warning (T6.2).
- `:single`: closures below the Float32 floors (`FLOAT32_CLOSURES`, docs/precision.md) are
  raised, with an info message.

## GWF model packages

| Package | Supported | Rejected (card) |
| --- | --- | --- |
| Name file | LIST, PRINT_INPUT, PRINT_FLOWS, SAVE_FLOWS | NEWTON (T6.2, `:newton`; also *engine*, `GwfModel`), UNDER_RELAXATION (Newton) |
| DIS6 | LENGTH_UNITS, NOGRB, GRB6 output, XORIGIN, YORIGIN, ANGROT, CRS; IDOMAIN 0, 1 and −1 (pass-through) | DISV6, DISU6 (T6.4, `:disv`/`:disu`; also *engine*, connectivity) |
| NPF6 | ICELLTYPE 0 (confined), K, K22, K33, ANGLE1–3, K22OVERK, K33OVERK, ALTERNATIVE_CELL_AVERAGING (LOGARITHMIC, AMT-LMK, AMT-HMK; harmonic default), SAVE_FLOWS, PRINT_FLOWS, SAVE_SPECIFIC_DISCHARGE | ICELLTYPE ≠ 0 in an active cell (T6.1, `:convertible`), THICKSTRT (T6.1), VARIABLECV, DEWATERED, PERCHED, REWET / WETDRY (T6.1), XT3D (T6.5), TVK6 (T6.x), other options |
| STO6 | SS, SY (read), ICONVERT 0, STORAGECOEFFICIENT, SS_CONFINED_ONLY, SAVE_FLOWS, STEADY-STATE / TRANSIENT periods | ICONVERT ≠ 0 with a transient period (T6.1, `:convertible_storage`; also *engine*, `check_storage_spec`), TVS6 (*engine*) |
| IC6 | STRT | – |
| OC6 | HEAD and BUDGET FILEOUT, HEAD PRINT_FORMAT; SAVE / PRINT HEAD / BUDGET with ALL, FIRST, LAST, FREQUENCY, STEPS | BUDGETCSV (and other options) |
| CHD6, WEL6, GHB6, RIV6, DRN6 (lists) | AUXILIARY, AUXMULTNAME, BOUNDNAMES, PRINT_INPUT, PRINT_FLOWS, SAVE_FLOWS, TS6 / OBS6 FILEIN (see below), MAXBOUND, PERIOD blocks (incl. empty blocks); DRN AUXDEPTHNAME | MOVER (MVR6 out of scope), WEL AUTO_FLOW_REDUCE (*engine*, T6.1), values given as time-series names (T6.3, `:timeseries`), READARRAYGRID input |
| RCH6 (list and READASARRAYS) | as the lists, plus FIXED_CELL, IRCH | TIMEARRAYSERIES / TAS6 values (T6.3) |
| EVT6 (list and READASARRAYS) | as the lists, plus FIXED_CELL, SURF_RATE_SPECIFIED, NSEG ≥ 1 (PXDP, PETM, PETM0), IEVT | TIMEARRAYSERIES / TAS6 values (T6.3) |
| OBS6 | the file must exist; it is read by no one | observations are not evaluated or written until T6.7 (accepted, no output) |
| TS6, TAS6 | FILEIN references are accepted | values that use them (T6.3) |
| Always rejected | – | SFR6, LAK6, MAW6, UZF6, MVR6, CSUB6, BUY6, VSC6, API6 (out of scope), GNC6, HFB6 (not in the plan) |

## Engine features

| Feature | Status |
| --- | --- |
| Formulation | confined (ICELLTYPE 0), Picard outer iterations, flux-form residual, heads as offsets (PLAN.md 3.7) |
| Precision presets | `:double`, `:mixed` (T4.4), `:single`, `:auto` (`:mixed` with Float64, else `:single`) |
| Backends | CPU (KernelAbstractions), ROCm (tested on the RX 7800 XT); CUDA, Metal, oneAPI, OpenCL registered but untested |
| Output | HDS, CBC (FLOW-JA-FACE, STO-SS, package lists), listing budget tables and PRINT HEAD, DIS GRB |
| Stepping | libmf6/XMI-like stepping API and BMI (`get_value_ptr`, `set_value` of X and BOUND), T3.3 |
| Grid | DIS only; no longitude wrap (T7.1), no spherical geometry (T7.2) |

## Validated models (Milestone A gate)

Parity against mf6 6.8.2 in both presets on the CPU backend and on ROCm (tolerances of
PLAN.md section 4; numbers in docs/precision.md and docs/scaling.md): Tier 0
(ex-gwf-toth, ex-gwf-hanic, ex-gwf-hanir), Tier 2 (fhb_const; ex-gwf-hwell-a only in
`:double`, opt-in), and Tier 1, the synthetic coupling models (test/models/coupling_rde
and benchmark/synthetic.jl `tiny` and `indus`). The Tier 3–7 examples load with
`strict = false` only.

T6.0 adds the two confined Tier 3 examples (ICELLTYPE 0, steady state, RIV, RCHA,
K33OVERK, IMS CG with STRICT rclose): ex-gwf-lgrv-gc (9×61×49) and ex-gwf-lgrv-gr
(25×183×147, 672525 cells; `MFGPU_TEST_LGRV_GR=0` skips it), in the universal residual
test and the run parity test, `:single`, `:mixed` and `:double`, CPU and ROCm. Default
preconditioner, except ex-gwf-lgrv-gr in `:single`, which runs with `:amg`: with
`:mc_ilu0` the Float32 solve does not converge (docs/precision.md, T6.0).
