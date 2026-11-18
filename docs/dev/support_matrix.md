# Support matrix (Milestone A, and Phase 6 cards as they merge)

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
| NONLINEAR | OUTER_DVCLOSE, OUTER_MAXIMUM, UNDER_RELAXATION NONE / SIMPLE / COOLEY / DBD (+ THETA, KAPPA, GAMMA, MOMENTUM), BACKTRACKING_NUMBER / _TOLERANCE / _REDUCTION_FACTOR / _RESIDUAL_LIMIT (also COMPLEXITY COMPLEX's; T6.2a, `sln_backtracking`, on DIS, DISV and DISU, confined and convertible cells, every precision preset), LINEAR_SOLVER DEFAULT / LINEAR | other LINEAR_SOLVER values |
| LINEAR | INNER_MAXIMUM, INNER_DVCLOSE, INNER_RCLOSE (absolute, STRICT, L2NORM_RCLOSE, RELATIVE_RCLOSE), LINEAR_ACCELERATION CG / BICGSTAB, NUMBER_ORTHOGONALIZATIONS, SCALING_METHOD NONE / DIAGONAL, REORDERING_METHOD NONE | SCALING_METHOD L2NORM, REORDERING_METHOD RCM / MD |

Accepted with a different behaviour (logged warnings, not errors):

- Preconditioner: ILU0, MILU0 (RELAXATION_FACTOR > 0) and ILUT / MILUT
  (PRECONDITIONER_LEVELS > 0, PRECONDITIONER_DROP_TOLERANCE) are accepted and replaced by
  the ModflowGPU option `preconditioner` (`ModflowGPU.run` / `initialize` keyword):
  `:auto` (default: `:mc_ilu0`, T4.4; logged as a message for ILU0, a warning for the
  others), `:jacobi`, `:chebyshev`, `:mc_sgs`, `:mc_ilu0` (multicolour ILU0, T4.2), `:amg` (smoothed-aggregation AMG, T4.3), with `precond_options`
  (docs/dev/mf6_notes.md). RED_BLACK_ORDERING is accepted (it only affects MF6's ILU).
- `:single`: closures below the Float32 floors (`FLOAT32_CLOSURES`, docs/dev/precision.md) are
  raised, with an info message.

## GWF model packages

| Package | Supported | Rejected (card) |
| --- | --- | --- |
| Name file | LIST, PRINT_INPUT, PRINT_FLOWS, SAVE_FLOWS | NEWTON (T6.2, `:newton`; also *engine*, `GwfModel`), UNDER_RELAXATION (Newton) |
| DIS6 | LENGTH_UNITS, NOGRB, GRB6 FILEOUT (accepted; GRB files are not written), XORIGIN, YORIGIN, ANGROT, CRS; IDOMAIN 0, 1 and −1 (pass-through) | – |
| DISV6 (T6.4) | as DIS6; VERTICES, CELL2D (vertex lists closed as in MF6); IDOMAIN 0, 1 and −1 (pass-through) | – |
| DISU6 (T6.4) | as DIS6, plus VERTICAL_OFFSET_TOLERANCE; CONNECTIONDATA IAC, JA, IHC (0, 1, 2), CL12, HWVA, ANGLDEGX (optional); VERTICES, CELL2D (optional); IDOMAIN 0 and 1 (MF6 rejects −1) | READASARRAYS stress input (MF6 rejects it too) |
| NPF6 | ICELLTYPE 0 (confined) and ≠ 0 (convertible, standard formulation, T6.1), THICKSTRT, VARIABLECV [DEWATERED], PERCHED, REWET (WETFCT, IWETIT, IHDWET) with WETDRY (T6.1), K, K22, K33, ANGLE1–3, K22OVERK, K33OVERK (on DISV/DISU the K ellipse uses the connection angles, `anglex`, T6.4; DISU K22 or SAVE_SPECIFIC_DISCHARGE without ANGLDEGX, or with inconsistent ANGLDEGX, is MF6's error), ALTERNATIVE_CELL_AVERAGING (LOGARITHMIC, AMT-LMK, AMT-HMK; harmonic default), SAVE_FLOWS, PRINT_FLOWS, SAVE_SPECIFIC_DISCHARGE | XT3D (T6.5), HIGHEST_CELL_SATURATION and NEWTON (T6.2), TVK6 (T6.x), other options |
| STO6 | SS, SY, ICONVERT 0 and ≠ 0 (convertible storage, STO-SY, T6.1), STORAGECOEFFICIENT, SS_CONFINED_ONLY, SAVE_FLOWS, STEADY-STATE / TRANSIENT periods | TVS6 (*engine*) |
| IC6 | STRT | – |
| OC6 | HEAD and BUDGET FILEOUT, HEAD PRINT_FORMAT; SAVE / PRINT HEAD / BUDGET with ALL, FIRST, LAST, FREQUENCY, STEPS | BUDGETCSV (and other options) |
| CHD6, WEL6, GHB6, RIV6, DRN6 (lists) | AUXILIARY, AUXMULTNAME, BOUNDNAMES, PRINT_INPUT, PRINT_FLOWS, SAVE_FLOWS, TS6 / OBS6 FILEIN (see below), MAXBOUND, PERIOD blocks (incl. empty blocks); DRN AUXDEPTHNAME; values given as time-series names in any column MF6 allows, auxiliary variables (also the AUXMULTNAME multiplier) included (T6.3) | MOVER (MVR6 out of scope), WEL AUTO_FLOW_REDUCE_CSV, READARRAYGRID input; WEL AUTO_FLOW_REDUCE, FLOW_REDUCTION_LENGTH, AUTO_FLOW_REDUCE_AUXNAME are supported (T6.1) |
| RCH6 (list and READASARRAYS) | as the lists, plus FIXED_CELL, IRCH; READASARRAYS: TAS6 FILEIN and `RECHARGE` / auxiliary `TIMEARRAYSERIES` (T6.3) | – |
| EVT6 (list and READASARRAYS) | as the lists, plus FIXED_CELL, SURF_RATE_SPECIFIED, NSEG ≥ 1 (PXDP, PETM, PETM0), IEVT; READASARRAYS: TAS6 FILEIN and `RATE` / auxiliary `TIMEARRAYSERIES` (T6.3) | – |
| OBS6 | the file must exist; it is read by no one | observations are not evaluated or written until T6.7 (accepted, no output) |
| TS6 (T6.3) | ATTRIBUTES NAME(S), METHOD(S) STEPWISE / LINEAR / LINEAREND, SFAC(S), AUTODEALLOCATE (no effect); several series per file and several files per package; DNODATA values; evaluated per time step on the host in Float64 (time-weighted average over the step, LINEAREND the value at its end) | a series that ends before a step ends (MF6's error; no extension to the end of the simulation) |
| TAS6 (T6.3) | ATTRIBUTES NAME, METHOD STEPWISE / LINEAR, SFAC, AUTODEALLOCATE; one `TIME` block per array (CONSTANT, INTERNAL, OPEN/CLOSE, FACTOR); evaluated per time step on the host in Float64 (time-weighted average) | DISU grids (MF6 does not support them either) |
| Always rejected | – | SFR6, LAK6, MAW6, UZF6, MVR6, CSUB6, BUY6, VSC6, API6 (out of scope), GNC6, HFB6 (not in the plan) |

## Engine features

| Feature | Status |
| --- | --- |
| Formulation | confined (ICELLTYPE 0) and convertible cells with wetting/drying (standard formulation, T6.1; Newton is T6.2), Picard outer iterations, flux-form residual, heads as offsets (PLAN.md 3.7) |
| Precision presets | `:double`, `:mixed` (T4.4), `:single`, `:auto` (`:mixed` with Float64, else `:single`) |
| Backends | CPU (KernelAbstractions), ROCm (tested on the RX 7800 XT); CUDA, Metal, oneAPI, OpenCL registered but untested |
| Output | HDS, CBC (FLOW-JA-FACE, STO-SS, package lists), listing budget tables and PRINT HEAD, in the DIS, DISV (one head record per layer) and DISU (one record) layouts; GRB files are read, not written |
| Stepping | libmf6/XMI-like stepping API and BMI (`get_value_ptr`, `set_value` of X and BOUND), T3.3 |
| Grid | DIS, DISV and DISU (T6.4; host connectivity in MF6's CSR layout; the multicolour preconditioners colour DISV/DISU greedily, DIS red–black); no longitude wrap (not planned: T7.1 was dropped), no spherical geometry (T7.2) |

## Validated models (Milestone A gate)

Parity against mf6 6.8.2 in both presets on the CPU backend and on ROCm (tolerances of
PLAN.md section 4; numbers in docs/dev/precision.md and docs/dev/scaling.md): Tier 0
(ex-gwf-toth, ex-gwf-hanic, ex-gwf-hanir), Tier 2 (ex-gwf-fhb since T6.3, which replaced the
`fhb_const` stand-in; ex-gwf-hwell-a only in `:double`, opt-in), and Tier 1, the synthetic coupling models (test/models/coupling_rde
and benchmark/synthetic.jl `tiny` and `indus`). The Tier 3–7 examples load with
`strict = false` only.

T6.0 adds the two confined Tier 3 examples (ICELLTYPE 0, steady state, RIV, RCHA,
K33OVERK, IMS CG with STRICT rclose): ex-gwf-lgrv-gc (9×61×49) and ex-gwf-lgrv-gr
(25×183×147, 672525 cells; `MFGPU_TEST_LGRV_GR=0` skips it), in the universal residual
test and the run parity test, `:single`, `:mixed` and `:double`, CPU and ROCm. Default
preconditioner, except ex-gwf-lgrv-gr in `:single`, which runs with `:amg`: with
`:mc_ilu0` the Float32 solve does not converge (docs/dev/precision.md, T6.0).

T6.3 adds time series: ex-gwf-fhb (CHD heads and WEL rates from LINEAREND series) in the
universal residual test, the storage residual test and the run parity test against its
tight-relative reference run (its tight run does not converge in MF6), `:single`,
`:mixed` and `:double`, CPU and ROCm; and the synthetic model test/models/ts_tas (TS6
LINEAR, STEPWISE and LINEAREND with SFACS, DNODATA, several series per file, a time-series
AUXMULTNAME multiplier, TS6 GHB conductance, RCHA and EVTA TAS6 arrays and MF6's zeroing
of a TAS-linked array by a later PERIOD block), run with mf6 at test time. ex-gwf-advtidal
also uses time series and is validated since T6.1.

T6.4 adds the Tier 5 DISV examples ex-gwf-u1disv (121 cells), ex-gwf-curve-90 (288) and
ex-gwf-curvilin (864) (CHD, STO steady state, BiCGSTAB), and `rad_disu_confined`
(ex-gwf-rad-disu, DISU with vertices, 550 cells, transient WEL, with STO ICONVERT 0; built
from the reference run at test time, AMG preconditioner) in the universal residual test
and the run parity test, `:single`, `:mixed` and `:double`, CPU and ROCm (docs/dev/precision.md,
T6.4). ex-gwf-u1disv-x and ex-gwf-disvmesh need XT3D (T6.5).

T6.1 adds convertible cells: ex-gwf-twri01, ex-gwf-bcf2ss-p01a, ex-gwf-bump-p01b,
ex-gwf-nwt-p02b (Tier 3), ex-gwf-advtidal, ex-gwf-maw-p03a, ex-gwf-maw-p03c (Tier 2 with
convertible layers), ex-gwf-rad-disu (DISU, convertible storage; AMG) and the synthetic
test/models/wetdry, disu_staggered and afr (mf6 at test time), in the universal residual
test, the libmf6 oracle and the run parity test, `:single`, `:mixed` and `:double`, CPU and
ROCm (docs/dev/precision.md, T6.1, with two documented per-example criteria). Left out of the
convertible examples: the Tier 4 examples (NEWTON, T6.2); OBS6 output of advtidal,
maw-p03c and rad-disu is not written (T6.7).
