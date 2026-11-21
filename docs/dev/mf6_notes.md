# Notes on MF6 behaviour

Findings from reading the MF6 Fortran (`reference/modflow6`, 6.8.2 source) that the port
depends on. Each section names the routines it is based on.

## Constant heads (CHD), MF6 >= 6.4

Routines: `chd_rp`, `chd_ad`, `chd_fc`, `chd_cq`, `calc_chd_rate`, `chd_bd`
(`gwf-chd.f90`); `sln_ls` (`NumericalSolution.f90`); `default_flow_cq`, `hcond`/`vcond`
(`gwf-npf.f90`).

Since 6.4 a CHD cell is no longer an extra equation with a large diagonal. It keeps its
row in the system but is not an unknown:

1. **IBOUND.** `chd_rp` first resets the cells of the previous period's records to
   `ibound = ibcnum` (active), then reads the new period data and sets
   `ibound(node) = -ibcnum` for each record. A cell that is already negative (in two CHD
   records, or in two CHD packages) is an error ("Cell is already a constant head").
2. **Head.** `chd_ad` sets `xnew(node) = xold(node) = head` (with the MULTIPLIER aux
   applied, `head_mult`) at the start of every time step.
3. **Formulate.** `chd_fc` adds nothing. NPF (`npf_fc`) still adds the connection
   conductances to the CHD row and to its active neighbours' rows; other boundary
   packages skip cells with `ibound <= 0` in their `cf` routines.
4. **Row replacement.** In `sln_ls`, for every row with `active <= 0` (CHD and no-flow
   alike): diagonal = 1, off-diagonals zeroed (`zero_row_offdiag`), `rhs = x`. So the CHD
   row reads `h_n = head`. With a symmetric matrix (`isymmetric == 1`, CG) in sequential
   mode, each active row's entries in CHD columns are moved to the right-hand side
   (`rhs -= a_nm * x_m`, `a_nm = 0`) to keep the matrix symmetric. Active neighbours
   therefore see the CHD cell as a fixed head either way.
5. **Flows.** NPF computes `flowja` for every connection from the heads (the conductance
   is zero when either cell is inactive, so CHD-to-active connections carry flow).
   `calc_chd_rate` (called from `chd_cq` and again from `chd_bd`, after exchanges have
   finished `flowja`) then, per record: `rate = -Σ flowja(ipos)` over the cell's
   off-diagonal connections (the net flow from the boundary into the model, stored in
   `simvals` and the CBC); `ratechdin`/`ratechdout` split this over connections to
   **active** neighbours only (`ibound(n2) > 0`), for the budget table; `rhs(i) = -rate`,
   `hcof(i) = 0`; and `flowja(idiag) += rate`.

**Port (T2.3).** `boundary_ibound` sets IBOUND -1 for CHD cells (error on a duplicate);
`apply_chd_heads!` writes the CHD heads (as offsets) into the head vector;
the CHD formulate is a no-op and the generic formulate kernel skips `ibound <= 0`; the
row replacement belongs to the assembly (T2.6); `calc_chd_rate!` ports `calc_chd_rate`
(`simvals`, `ratein`, `rateout`, diagonal of `flowja`). It does not write `rhs = -rate`,
which in MF6 only feeds observations and the MVR. Verified against MF6's CBC CHD records on
ex-gwf-toth and ex-gwf-hanic (`test/parity/boundaries_test.jl`).

**Port (T2.6).** `assemble!` (src/solver/csr.jl) applies `sln_ls`: rows with
`ibound <= 0` become identity rows with `rhs = h'` (the offset CHD head), active rows
with a diagonal below 1e-15 in magnitude get diagonal −1 and `rhs −= h'`, and with
`symmetric = true` (IMS `LINEAR_ACCELERATION CG`, sequential mode) the CHD columns of
active rows are moved to the right-hand side. The off-diagonals of all other rows keep
the coupling to CHD cells (`hcond`/`vcond` are zero only for `ibound == 0`). The
flux-form residual is zero on CHD rows. Verified bitwise (`:double`) against libmf6's
`SLN_1/AMAT` after the first solve on toth (CG), hanic/hanir (BiCGSTAB), fhb_const (CG)
and coupling_rde (`test/parity/assembly_test.jl`). Note: BMI `get_time_step` does not
return the step length; read `TDIS/DELT`.

## Other boundary details used in T2.3

- **MULTIPLIER aux / AUXMULTNAME** (`bnd_rp`, `BndExt`): the aux column named by
  `AUXMULTNAME` multiplies one `bound` column: CHD head, WEL rate, GHB conductance, RCH
  rate. The port applies it on the host in Float64 before head columns become offsets.
- **WEL `AUTO_FLOW_REDUCE`** (`wel_cf`) reduces the rate in partly dry convertible cells;
  rejected (`MF6UnsupportedError`) until convertible cells exist (T6.1).
- **RCH without FIXED_CELL** (`rch_cf`): when a record's cell has `ibound == 0`, the
  recharge goes to the highest active cell below in the same column; if none is active it
  stays on the bottom cell (and is then skipped). With FIXED_CELL nothing moves. RCHA
  (READASARRAYS) has one record per column (NCPL), with IRCH selecting the layer.
- **Area.** RCH rates are per unit area; `rch_cf` multiplies by the cell area
  (`dis%get_area`). GHB `rhs = -C * bhead`, `hcof = -C` (`ghb_cf`); WEL `rhs = -q`
  (`wel_cf`).

## IMS linear solver mapping (T2.7)

Routines: `imslinear_ar`, `imslinear_ap` (`ImsLinear.f90`); `ims_base_cg`,
`ims_base_bcgs`, `ims_base_scale`, `ims_base_testcnvg`, `ims_base_epfact`,
`ims_base_residual` (`ImsLinearBase.f90`); `resolve_ipc`, `preset_config`
(`ImsLinearSettings.f90`). Implementation: `src/solver/krylov.jl`
(`KrylovSettings(::ImsSpec)`, `KrylovSolver`, `solve!`).

| IMS input | MF6 | ModflowGPU |
| --- | --- | --- |
| LINEAR_ACCELERATION CG (SIMPLE default) | PCG on `A` (negative definite active block) | PCG on `S·A`, `S = diag(sign(a_nn))` (`−A` on active rows); same iterates for the correction solve |
| LINEAR_ACCELERATION BICGSTAB (MODERATE, COMPLEX default) | right-preconditioned BiCGSTAB | the same |
| preconditioner: ILU0 (SIMPLE), MILU0 (RELAXATION_FACTOR > 0; MODERATE 0.97), ILUT/MILUT (PRECONDITIONER_LEVELS > 0; COMPLEX 5, drop tolerance 1e-4) | `resolve_ipc(level, relax)`, `ims_base_pcilu0` / `ims_base_ilu0a` in natural order | replaced by the ModflowGPU option `preconditioner` (keyword of `ModflowGPU.run` / `initialize`, stored in `KrylovSettings` with `precond_options`; T4.2, T4.3): `:auto` (default: `DEFAULT_PRECONDITIONER` = `:mc_ilu0`, T4.4; logged as a message for ILU0 and as a warning for MILU0/ILUT/MILUT), `:jacobi`, `:chebyshev`, `:mc_sgs`, `:mc_ilu0`, `:amg` (see below) |
| SCALING_METHOD NONE / DIAGONAL / L2NORM | `ims_base_scale` ISCL 0/1/2 (matrix scaled in place) | the same factors applied in the operator (the matrix is not modified); a warning for L2NORM with CG (the scaled matrix is nonsymmetric, as in MF6) |
| REORDERING_METHOD RCM / MD | permutes the system before ILU | ignored, logged (`:mc_*` use their own colour ordering; no effect with Jacobi or AMG) |
| INNER_MAXIMUM, INNER_DVCLOSE, INNER_RCLOSE (+ STRICT, L2NORM_RCLOSE, RELATIVE_RCLOSE, L2NORM_RELATIVE_RCLOSE) | `ims_base_testcnvg` options 0–4, `ims_base_epfact` | the same, on the host in Float64 |
| NUMBER_ORTHOGONALIZATIONS (COMPLEX default 2) | residual recomputed from `x` every `north` iterations | the same |
| initial guess | the current heads `x` | zero (correction solve `A·δ = res`, PLAN.md 3.7 rule 3); the residuals, `L2NORM0` and updates are the same quantities |

Preconditioners (T4.2, `src/solver/precond/`; all on the scaled, for CG sign-adjusted
operator `Â`, set up at every `solve!` as `imslinear_ap` does):

- `:mc_ilu0`: ILU(0) of `Â` (IKJ, `ims_base_pcilu0` without DELTA restarts) in
  **multicolour order** instead of MF6's natural order: DIS grids red–black by
  `(i + j + k)` parity, other grids (and DIS with pass-through cells) greedy colours. Each
  colour is one parallel kernel in the factorisation and in the forward/backward solves
  (`ims_base_ilu0a`). Red–black ILU0 is a weaker preconditioner than natural-order ILU0
  (about 2.5× MF6's inner iterations on the synthetic models), but runs in parallel.
  Pivots that change sign or vanish are replaced by the unfactorised diagonal (MF6
  restarts with a larger DELTA). MILU relaxation (`precond_options = (; relax)`) is
  implemented but off: IMS RELAXATION_FACTOR is not used, and on red–black orderings it
  did not help (indus: 1055 → 1274 inner iterations at 0.97).
- `:mc_sgs`: symmetric Gauss–Seidel in the same colour order (`(D+L)·D⁻¹·(D+U)`).
- `:chebyshev`: Jacobi-preconditioned Chebyshev polynomial, degree 4 on
  `[λmax/30, λmax]`, `λmax` from 10 Lanczos (Jacobi-PCG) steps × 1.1, capped by Gershgorin
  (`precond_options` `degree`, `ratio`, `lanczos`, `safety`).
- All are SPD for SPD `Â` (CG). The iterates differ from MF6's (other preconditioner);
  the IMS convergence tests are unchanged.

Semantics kept from MF6:

- The convergence flag is 1 only when `ims_base_testcnvg` returns 1. A return of −1
  (STRICT after the first inner iteration; the relative L2 criteria) stops the inner loop
  without convergence (`KrylovResult.status = :criterion`).
- CG tests the scaled `δ` updates and residual; BiCGSTAB unscales both (`tv·DSCALE`,
  `r/DSCALE`, using the row factor DSCALE also for L2NORM, where `x` is scaled by the
  column factor DSCALE2). `L2NORM0` is the scaled initial norm in both, so with L2NORM
  scaling BiCGSTAB compares an unscaled norm with a scaled `L2NORM0`.
- Stagnation exits: CG stops when `is_close(rho, rho0)` or `rho == 0`; BiCGSTAB when rho,
  alpha or omega repeat (`is_close`, rtol 100·eps) or `rho·omega == 0`. Deviation: MF6
  adds the absolute `SIGN(DPREC, d)` to denominators; we add `SIGN(eps·|numerator|, d)`
  (`_safe_div`). The absolute guard stalls the correction solves once the dot products
  reach ~1e-16 (|r| ~ 1e-8), so the inner solve hit INNER_MAXIMUM every outer iteration
  (ex-gwf-hanir `:double`: 16 outer / 14402 inner with it, 4 / 222 without; coupling_rde 226 / 215792
  → 103 / 5722).
- `L2NORM0 == 0` gives zero iterations and convergence.

**Preconditioner options (T4.3).** `precond_options` (a NamedTuple, next to
`preconditioner` in `ModflowGPU.run`, `initialize`, `PicardSettings`, `KrylovSettings`) are
the keywords of the preconditioner's constructor (`build_preconditioner`): for `:amg` those
of `AMGOptions`, for `:chebyshev` `degree`, `ratio`, `lanczos`, `safety`, for `:mc_ilu0`
`relax`; Jacobi has none.

`:amg` is smoothed-aggregation AMG (`src/solver/precond/amg.jl`; MF6 has no AMG): host
setup in Float64 on the scaled (for CG sign-adjusted) operator, V-cycle on the device with
a Chebyshev smoother (symmetric, so CG stays valid; BiCGSTAB uses it as MF6 uses ILU, as
a right preconditioner), dense coarsest solve. Aggregates are kept between outer
iterations and time steps; each linear solve whose matrix changed recomputes the smoothed
prolongators and the Galerkin coarse operators on the device. The hierarchy is rebuilt
when a solve degrades (`rebuild_factor`) and before a failed solve is repeated (the
inner iterations of both attempts count). Differences from MF6 are in the inner
iteration counts only; the convergence tests are unchanged.

## Outer iterations, time steps, budgets and output (T2.8)

Routines: `sln_ca`, `solve`, `sln_get_dxmax`, `sln_has_converged`, `sln_underrelax`,
`finalizeSolve` (`NumericalSolution.f90`); `imslinear_ap` (`ImsLinear.f90`);
`tdis_set_timestep`, `tdis_set_delt` (`tdis.f90`); `gwf_cq`, `gwf_bd`, `gwf_ot`,
`gwf_ot_flow` (`gwf.f90`); `bnd_cq_simrate`, `save_print_model_flows`
(`BoundaryPackage.f90`); `csr_diagsum` (`Sparse.f90`); `budget_ot`, `value_to_string`
(`Budget.f90`); `set_print_flag` (`OutputControl.f90`), `is_selected`
(`TimeStepSelect.f90`). Implementation: src/solver/picard.jl, src/solver/underrelax.jl,
src/timing.jl, src/budget.jl, src/output/oc.jl, src/run.jl.

- **Outer convergence.** An outer iteration converges only when the inner solve
  converged (`imslinear_ap` returns ICNVG 1; STRICT and the relative criteria return −1,
  reset to 0) **and** `|hncg| <= OUTER_DVCLOSE`, where `hncg` is the signed largest
  `x − xtemp` over active cells. With INNER_RCLOSE STRICT (hanic, hanir) an inner solve
  converges only when its first inner iteration does, so the outer loop runs until the
  residual is tiny. The port: `converged = result.converged && |max δ| <= dvclose`.
- **Under-relaxation** is applied only to a non-converged outer iteration: SIMPLE
  `x = xtemp + gamma·δ`; COOLEY one factor from `bigch = hncg` and its weighted history
  (`relax < 1` only changes `x`); DBD a factor per cell (`wsave`, `hchold`, `deold`
  initialised at `kiter == 1`, momentum from `kiter > 4`). A converged iteration keeps the
  full solution. COMPLEXITY MODERATE/COMPLEX select DBD; COMPLEX also backtracking
  (BACKTRACKING_NUMBER 20, see below).
- **Backtracking** (`sln_backtracking`, T6.2a; src/solver/backtracking.jl) runs at the
  start of every outer iteration when BACKTRACKING_NUMBER > 0, whatever UNDER_RELAXATION
  is: MF6 sets `ibflag = 0` without under-relaxation, but `solve` tests only `numtrack`.
  It needs its own `sln_buildsystem` before the one of the solve, and so runs `npf_cf`, and
  with it the wetting and drying, twice per outer iteration (plus once per reduction);
  the port does the same (`_backtracking!`), and reuses the last formulation for the solve
  when the second `wetdry!` changes nothing. The norm is `||A·x − b||₂` over active cells
  right after `sln_buildsystem`, so without `sln_ls`'s zero-diagonal fix; the port takes
  the norm of the flux-form residual, the same quantity. The reduced change is
  `x − xtemp`, `xtemp` being `x` before the previous solve (after its backtracking): the
  under-relaxed update. A dry cell has `x = HDRY` (−1e30) in MF6, so a cell rewetted in
  the next iteration has a change of ~1e30: it always passes the OUTER_DVCLOSE test, and
  its reduced head (≈ (1 − breduc)·HDRY) dries it again in the next `wetdry`. ModflowGPU keeps dry
  heads at the bottom; `save_hprev!` stores HDRY for them so the same happens (ex: the
  wetdry test model, 18 backtracking outer iterations, all in rewetting steps). OUTER_DVCLOSE
  in the reduction test is the one in use (the Float32 floors raise it in `:single`, so
  fewer reductions are done before flag 4). On coupling_rde with COMPLEXITY COMPLEX, MF6
  and the port backtrack in the same 3 of 125 outer iterations with the same norms
  (test/parity/run_test.jl).
- **Time steps.** The first step of a period is `perlen/float(nstp)` or
  `perlen·(1 − tsmult)/(1 − tsmult^nstp)`, later ones `tsmult·delt`, but the **last** step
  of a period is `topertim + perlen − totim`, and `pertim = perlen` at the end of the
  period.
- **Flows** (`gwf_cq`): `flowja` zeroed; NPF off-diagonals (flow into `n`); STO-SS rates on
  the diagonal; each package re-formulated with the final heads (`bnd_cf`) and its
  `simvals` (`hcof·x − rhs`, 0 for `ibound <= 0`) added to the diagonal; CHD
  `calc_chd_rate`; then `csr_diagsum` adds the off-diagonals to the diagonal (`gwf_bd`),
  so the CBC FLOW-JA-FACE diagonal is each cell's imbalance.
- **Budget** (`rate_accumulator`): positive rates are IN, negated negative rates OUT;
  CHD uses `ratechdin`/`ratechdout` (the split over connections to active cells, see
  the CHD section); cumulative volumes `+= rate·delt`. Order: STO-SS (STO-SY if any
  ICONVERT ≠ 0), then the packages. The listing's package column says `STORAGE` for STO.
  The budget table is printed with PRINT BUDGET, at the end of every period, and after a
  failed step (`set_print_flag`).
- **CBC records** (`gwf_ot_flow`, written when OC saves BUDGET): STO-SS (and STO-SY) as
  IMETH 1 arrays over the user nodes (dims `(ncol, nrow, −nlay)`, 0 outside the reduced
  grid) when STO or the model saves flows, also in steady periods; FLOW-JA-FACE as IMETH 1
  with dims `(nja, 1, −1)`; DATA-SPDIS with SAVE_SPECIFIC_DISCHARGE (not written yet);
  then per package an IMETH 6 list with `id1` = user node, `id2` = **record number**,
  `q` = `simvals` (CHD: the net cell rate), the package's aux values, and only records
  with a cell (`node > 0`). Texts: `RCHA` (`"            RCHA"`) and `EVTA` for
  READASARRAYS; MF6 6.8 writes EVTA's text as `"           EVTA "` (left in 16 characters).
  `paknam2` is the package name.
- **Output control.** A PERIOD block stays in force until the next one; an empty block
  turns output off. ALL, FIRST (`kstp == 1`), LAST (end of period), FREQUENCY n
  (`mod(kstp, n) == 0`), STEPS list; several records of one block combine.
- **Non-convergence.** Without the mfsim.nam CONTINUE option MF6 writes the step's output
  and stops; `ModflowGPU.run` throws a `ConvergenceError` after the step's output, and
  warns and continues with CONTINUE.

## Stepping API and BMI (T3.3)

- **Call semantics** (srcbmi/mf6xmi.F90, mf6core.f90): `prepare_time_step` does
  `tdis_set_timestep` and `model_rp` (period data; `dt` is ignored by libmf6), `prepare_solve`
  does `model_ad` (`hold ← x`, CHD heads, storage `rho`), each `solve` is one outer iteration
  (`has_converged` = `icnvg`), `finalize_solve` does `model_cq`/`model_bd` (SIMVALS, budgets),
  `finalize_time_step` writes output. Values set between `prepare_time_step` and
  `prepare_solve` are used for the step and stay until a PERIOD block of that package
  replaces them.
- **Port** (src/stepping.jl): the same calls on `Simulation`; `ModflowGPU.run` is built on
  them. libmf6 hands out pointers into Fortran memory; here `get_value_ptr` is a host Float64
  mirror (absolute heads, BOUND before the MULTIPLIER aux) synchronised at the call
  boundaries, and BOUND writes go through the period-data conversion (`set_bound!`). In
  MF6 6.8 (reference/), RIV keeps STAGE/COND/RBOT as separate arrays with the multiplier applied in
  `cf`; ModflowCoupling (libmf6 via ModflowInterface) uses `BOUND`, which the port keeps.
- `get_time_step` returns `delt` (libmf6); ModflowInterface's `BMI.get_time_step` calls
  `get_end_time` instead (a wrapper bug), see above.

## Time series (T6.3)

- When: `idm_ts_advance` runs at the end of `Mf6PrepareTimestep` (after `tdis_set_timestep`),
  so every linked value is the series over `[totimc, totimc + delt]` of the step about to be
  solved; ModflowGPU evaluates them in `prepare_time_step` (`advance_time_series!`). A
  stepping API write after `prepare_time_step` overrides them, as in libmf6.
- STEPWISE and LINEAR give the time-weighted average over the step, LINEAREND the
  interpolated value at the end. At a step of zero length the value at `t0` is used, and
  STEPWISE at a record time returns the *previous* record (`get_surrounding_records`
  stops before a record that `is_close` to the time).
- A series that starts after `t0` integrates to 0 without an error
  (`get_latest_preceding_node` finds no record); one that ends before `t1` is an error
  (the GWF loaders never set `extendToEndOfSimulation`). Exception: `idm_rp` reads the
  first PERIOD block before `tdis_set_timestep`, and `read_value_or_time_series` assigns
  `GetValue(totimsav, totim)` = the value at time 0, so a list series without a value at
  time 0 stops MF6 ("Error getting value at time 0.00"); `apply_period!` checks the same
  (T6.3 review, checked against mf6). TAS links are not evaluated when made.
- The multiplier (AUXMULTNAME) is applied on use in MF6 and at staging here; a series in
  the multiplier column is evaluated like any other auxiliary series.
- READASARRAYS (IDM `LayerArrayLoadType%rp` → `reset` → TimeArraySeriesManager `Reset`):
  every new PERIOD block first sets the arrays linked to a time-array series to zero, so a
  later block that does not give the variable has zero there (not the last series value).
  test/models/ts_tas checks this against mf6.
- Duplicate series names over the TS6 files of a package: the last one wins (the hash
  table replaces the index).

## Convertible cells, wetting and drying (T6.1)

Standard formulation only (no NEWTON; T6.2). Ported from gwf-npf.f90, gwf-sto.f90,
GwfConductanceUtils.f90, GwfStorageUtils.f90 and gwf-wel.f90 (6.8.2).

- **Saturation** (`sgwf_npf_thksat`): `(h − bot)/(top − bot)`, 1 at or above the top, and
  *not* clamped below 0: a cell with `h <= bot` is dried in the same `npf_cf` call before the
  saturation is used. It is computed per outer iteration in `npf_cf!` (`cf_default_flow`)
  and kept for the flows (`npf_cq` uses the `sat` of the last formulation with the final
  heads), as MF6 does.
- **Conductances** (`hcond`, `vcond`): horizontal `condmean` of `K·sat·thk` of the two cells
  (staggered DISU connections: `staggered_thkfrac` with the neighbour's elevations moved
  into the cell's frame by `Δr`); `condsat` when both cells are confined. Vertical
  `condsat`, or with VARIABLECV (and a convertible cell, and a head below a top) the series
  conductance of the saturated half-thicknesses, the lower cell full (or, with DEWATERED,
  a partly saturated lower cell counted as 0). The per-connection inputs (effective K
  towards the neighbour, `cl1`, `cl2`, `hwva`, whether the lower-numbered cell's bottom is
  higher) are precomputed on the host; the conductance is computed in the row kernels from
  them, so the CSR rows `n` and `m` of a connection compute the same value.
- **PERCHED** (`fc_default_flow`, `sgwf_npf_qcalc`): when the higher-numbered cell of a
  vertical connection is convertible with its head below its top, the flow is
  `C·(h_lo − bot_lo)` from the lower-numbered (upper) cell: the matrix entry of row `lo` is
  0 and its diagonal gets `−C`, so the matrix is not symmetric (MF6 runs CG on it anyway).
- **THICKSTRT**: negative ICELLTYPE cells become confined with the saturation of STRT
  (`calc_initial_sat`), which `calc_condsat` uses; STRT below the bottom is MF6's error.
- **Wetting and drying** (`sgwf_npf_wetdry`) runs at the start of every outer iteration
  (and once with `kiter = 0` when the model is built, `preprocess_input`, with IBOUND
  before the constant heads). First rewetting (REWET, `mod(kiter, IWETIT) == 0`): MF6 loops
  over the nodes in order, and a cell rewetted earlier in the loop (IBOUND 30000) counts as
  wet for the later ones; the first qualifying neighbour in CSR order sets the new head
  (IHDWET 0: `bot + WETFCT·(h_m − bot)`, else `bot + WETFCT·|WETDRY|`; horizontal
  neighbours only for WETDRY > 0). The kernel reproduces this with Jacobi passes (a cell
  depends only on lower-numbered cells, so the passes converge to the sequential result;
  usually 1–2 passes, one host synchronisation each). Then every convertible cell with
  `min(h, top) − bot <= 0` dries (IBOUND 0; a constant-head cell going dry is MF6's error
  "CONSTANT-HEAD CELL WENT DRY"). MF6 sets the head of a dry cell to HDRY = −1e30 and never
  reads it; the engine keeps it at the cell bottom (an offset of bounded size, so the
  Float32 reductions over `|h'|` are unaffected) and writes HDRY to the head file and the
  host heads. `npf_ad`: at the start of a step the old head of a dry cell that can rewet is
  its bottom. Cell conversion messages are not written to the listing file.
- **IBOUND changes**: after a wetting/drying change IBOUND is copied to the host and the RCH
  and EVT records without FIXED_CELL are re-placed from the cells given (`nodesontop`) on the
  highest active cell (`rch_cf`, `evt_cf`, `highest_active`); MF6 does this in every
  `rch_cf`, which gives the same nodes because IBOUND only changes in `sgwf_npf_wetdry`.
  `chd_rp` at a new period keeps dry cells dry. The off-diagonals of a convertible model
  are rewritten every outer iteration (a new `offdiag_id`: the MC-ILU0 preconditioner takes
  the full setup path); confined models keep the constant-`condsat` path.
- **Storage** (`fc_default_sto`, `SsTerms`, `SyTerms`, `cq_default_sto`): `sn` of the old
  and new heads with `sQuadraticSaturation(..., SATOMEGA = 0)` (clamped linear), SS terms
  with the mid-saturation elevations `z`, SS_CONFINED_ONLY. Deviation (same values in exact
  arithmetic): the SY right-hand sides and rate use `thk·sn = clamp(h, bot, top) − bot`,
  e.g. the rate `rho2·(clamp(hold) − clamp(h))` instead of `rho2·thk·snold − rho2·thk·snnew`,
  which cancels catastrophically in Float32 when `sn` is near 1 (ex-gwf-rad-disu: rates
  quantised to powers of 2 of 4–512 ft³/d).
- **Boundary checks** (`riv_ck`, `drn_ck`, `ghb_ck`): RIV bottom and stage, the DRN
  (scaled-conductance) bottom and the GHB head must not be below the bottom of a convertible
  cell; the conductance as given and the multiplier are checked separately. All errors of a
  block are reported together. MF6 also repeats `bnd_ck` in every `gwf_ad`; the engine
  checks when period data are applied (and the stepping API's device writes in
  `prepare_solve`).
- **WEL AUTO_FLOW_REDUCE** (`wel_cf`): an extraction in a convertible cell is multiplied by
  `sQSaturation(bot + f·thk, bot, h)` (f = AUTO_FLOW_REDUCE, ≤ 0 → 0.1, > 1 → 1; with
  FLOW_REDUCTION_LENGTH `bot + f`; AUTO_FLOW_REDUCE_AUXNAME gives f per record).
  AUTO_FLOW_REDUCE_CSV is not supported.

## READARRAYGRID input (T6.18)

- **Records** (Mf6FileGridArray.f90 `param_load`): the first array of a PERIOD block
  defines the records, one per user node whose value is not exactly DNODATA (3.0e30, after
  FACTOR: `FACTOR 1.02` on an array of 3.0e30 makes every cell a record), in node order;
  more than MAXBOUND (default: the number of user nodes, `ainit`) is an error. Later
  arrays of the block are taken at those nodes without a DNODATA test. Arrays not given
  keep their input memory by record index (zero until first given), so a block that gives
  only an auxiliary array with more cells than before shifts nothing but gives the new
  records the values of the old indices (test/models/arraygrid, WELG period 3).
- **Inactive cells** (BoundaryPackageExt.f90 `nodeu_to_nlist`): MF6 6.8.2 lowers NBOUND
  by the number of records at inactive cells, but neither moves the later records up nor
  sets their node, so the values shift when an inactive record is not the last one. We
  drop the records at inactive cells (`_drop_inactive`, src/input/simulation.jl); the
  results agree when the inactive records are last (the GHBG of test/models/arraygrid).
- No BOUNDNAMES, TS6 or TAS6 in the READARRAYGRID DFNs; NETCDF arrays and
  EXPORT_ARRAY_NETCDF are rejected.
