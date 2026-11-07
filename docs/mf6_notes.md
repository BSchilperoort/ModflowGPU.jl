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
| preconditioner: ILU0 (SIMPLE), MILU0 (RELAXATION_FACTOR > 0; MODERATE 0.97), ILUT/MILUT (PRECONDITIONER_LEVELS > 0; COMPLEX 5, drop tolerance 1e-4) | `resolve_ipc(level, relax)` | Jacobi, with a logged warning; ILU/AMG arrive in Phase 4 |
| SCALING_METHOD NONE / DIAGONAL / L2NORM | `ims_base_scale` ISCL 0/1/2 (matrix scaled in place) | the same factors applied in the operator (the matrix is not modified); a warning for L2NORM with CG (the scaled matrix is nonsymmetric, as in MF6) |
| REORDERING_METHOD RCM / MD | permutes the system before ILU | ignored (no effect with Jacobi), logged |
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
  alpha or omega repeat (`is_close`, rtol 100·eps) or `rho·omega == 0`. Denominators get
  `+ SIGN(DPREC, d)`.
- `L2NORM0 == 0` gives zero iterations and convergence.
