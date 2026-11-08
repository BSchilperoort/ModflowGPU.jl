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

| IMS input | MF6 | ModflowGPU (until Phase 4) |
| --- | --- | --- |
| LINEAR_ACCELERATION CG (SIMPLE default) | PCG on `A` (negative definite active block) | PCG on `S·A`, `S = diag(sign(a_nn))` (`−A` on active rows); same iterates for the correction solve |
| LINEAR_ACCELERATION BICGSTAB (MODERATE, COMPLEX default) | right-preconditioned BiCGSTAB | the same |
| preconditioner: ILU0 (SIMPLE), MILU0 (RELAXATION_FACTOR > 0; MODERATE 0.97), ILUT/MILUT (PRECONDITIONER_LEVELS > 0; COMPLEX 5, drop tolerance 1e-4) | `resolve_ipc(level, relax)` | Jacobi, with a logged warning, unless the ModflowGPU option `preconditioner` is given (T4.3, below); the default for production is decided in T4.4 |
| SCALING_METHOD NONE / DIAGONAL / L2NORM | `ims_base_scale` ISCL 0/1/2 (matrix scaled in place) | the same factors applied in the operator (the matrix is not modified); a warning for L2NORM with CG (the scaled matrix is nonsymmetric, as in MF6) |
| REORDERING_METHOD RCM / MD | permutes the system before ILU | ignored (no effect with Jacobi or AMG), logged |
| INNER_MAXIMUM, INNER_DVCLOSE, INNER_RCLOSE (+ STRICT, L2NORM_RCLOSE, RELATIVE_RCLOSE, L2NORM_RELATIVE_RCLOSE) | `ims_base_testcnvg` options 0–4, `ims_base_epfact` | the same, on the host in Float64 |
| NUMBER_ORTHOGONALIZATIONS (COMPLEX default 2) | residual recomputed from `x` every `north` iterations | the same |
| initial guess | the current heads `x` | zero (correction solve `A·δ = res`, PLAN.md 3.7 rule 3); the residuals, `L2NORM0` and updates are the same quantities |

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

**Preconditioner option (T4.3).** Not an MF6 input: `ModflowGPU.run`, `initialize` (and
`PicardSettings`, `KrylovSettings`) take `preconditioner = :jacobi | :amg` and
`precond_options` (a NamedTuple; for `:amg` the keywords of `AMGOptions`). With
`preconditioner = nothing` (the default) IMS's ILU0/MILU0/ILUT map to Jacobi with a
warning, as before; a given preconditioner replaces IMS's (logged). Interface:
`src/solver/precond/precond.jl` (`precond_setup!` once per linear solve after the scaling,
`precond_apply!` whole-vector `z = M⁻¹·r`); Jacobi stays fused into the CG/BiCGSTAB kernels.

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
  (BACKTRACKING_NUMBER 20; not ported until T6.2: `PicardSettings` warns). On coupling_rde
  with COMPLEXITY COMPLEX, MF6 backtracked in 3 of 125 outer iterations; the converged
  heads agree regardless (test/parity/run_test.jl).
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
