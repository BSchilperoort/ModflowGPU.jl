# Phase 2: CPU engine for confined models (KA kernels on the CPU backend)

Task cards for this phase of [PLAN.md](../../PLAN.md). Read PLAN.md sections 1–4 and 6 for the goals, conventions, architecture and validation rules these cards refer to.

Size: S ≈ < 1 day of agent work, M ≈ 1–3 days, L ≈ > 3 days.
`‖` marks cards that can run in parallel with their siblings.

All kernels are written once in KA. Phase 2 tests use `CPU()`. Phase 3 runs
the same code on the GPU.

**T2.1 Discretization and connectivity (M)**
- Depends: T1.4.
- Read: `src/Model/Discretization/Dis.f90` (`connection_init`,
  `nodeu_from_cellid`, `get_cell_volume`), `src/Model/ModelUtilities/Connections.f90`
  (`dis3dconnections`, `iajausr`). Paths are approximate; grep for them.
- Deliver: `src/discretization/dis.jl`, `connectivity.jl`: reduced nodes from
  IDOMAIN (including `IDOMAIN < 0` vertical pass-through cells), CSR with
  diagonal first, `jas`, `ihc`, `cl1`, `cl2`, `hwva`, cell `area`, `top`, `bot`,
  `nodeuser`/`nodereduced`.
- Done when: `ia/ja/ihc/cl1/cl2/hwva` equal libmf6's for toth, hanic, twri01 and
  a synthetic coupling model with inactive cells (tests from T0.2 addresses).

**T2.2 NPF, confined (M)**
- Depends: T2.1.
- Read: `src/Model/GroundwaterFlow/gwf-npf.f90` (`npf_ar`, `prepcheck`,
  `sgwf_npf_preprocess`, `hcond`, `vcond`, `condmean`, `hy_eff`).
- Deliver: the per-cell reference `r` and per-connection `Δr` (section 3.7,
  rule 1; computed on the host in Float64), and `src/gwf/npf.jl`: `condsat` per
  connection (computed in Float64 on the host, stored in `T`) for `ICELLTYPE 0`:
  harmonic default, `LOGARITHMIC`, `AMT-LMK`, `AMT-HMK`; `K22`/`K33` and
  `ANGLE1/2/3` effective K in the connection direction; `THICKSTRT` is parsed
  but rejected until T6.1; `K22OVERK`/`K33OVERK`.
- Convention (from the T2.1 review): per-connection arrays (`condsat`, `Δr`)
  are stored per symmetric pair (`njas`) and indexed through `jas`, like
  `cl1/cl2`. `Δr[jas]` is `r[m] − r[n]` with `n < m`; a row-wise gather for
  row `n` uses `+Δr[jas[ii]]` when `ja[ii] > n` and `−Δr[jas[ii]]` otherwise
  (as `connection_lengths` does).
- Done when: `condsat` equals libmf6 `NPF/CONDSAT` (if exposed) or passes the
  universal residual test on toth, hanic and hanir.

**T2.3 Boundary framework + CHD, WEL, RCH, GHB (M)**
- Depends: T2.1.
- Read: `src/Model/ModelUtilities/BoundaryPackage.f90` (`bnd_rp`, `bnd_fc`,
  `bnd_cq`), `gwf-chd.f90`, `gwf-wel.f90`, `gwf-rch.f90`, `gwf-ghb.f90`.
- Deliver: `src/boundaries/framework.jl` (section 3.4: records, node→record
  map, formulate and add kernels, `simvals`, `MULTIPLIER` aux, `AUXMULTNAME`;
  each package declares which `bound` columns are head-like, and the framework
  converts them to offsets on the host when period data is applied),
  then CHD, WEL (without `AUTO_FLOW_REDUCE` for now: reject it). First add a
  shared `to_offsets(r, values)` helper (Float64, cast to `T` after) in
  `src/gwf/offsets.jl` (from the T2.2 review), used for boundary head columns
  here and for top/bot (T2.4/T2.5) and the initial head (T2.6). Then RCH (list and
  array input, `FIXED_CELL`/highest-active cell), and GHB.
- **CHD:** find out exactly how MF6 ≥ 6.4 treats constant-head nodes in the
  system (ibound < 0, row replacement, how their flows are computed in `cq`) and
  replicate it. Write it down in `docs/mf6_notes.md`.
- Done when: unit tests per package pass, and the toth/hanic residual test
  passes with CHD/WEL.

**T2.4 RIV, DRN, EVT (M) ‖ with T2.5**
- Depends: T2.3.
- Read: `gwf-riv.f90`, `gwf-drn.f90` (incl. `DRNDEPTH` / `AUXDEPTHNAME` smoothing
  via `SmoothingFunctions.f90`), `gwf-evt.f90` (`NSEG`, `PETM`, `PXDP`,
  `SURF_RATE_SPECIFIED`, `FIXED_CELL`).
- Deliver: the three packages, with head-dependent `hcof`/`rhs` recomputed in the
  formulate kernel every outer iteration.
- Done when: unit tests check each branch (stage above/below RBOT, drain on/off,
  EVT above surface / within extinction depth / below it, multi-segment), and,
  once T2.5 (STO) is in, the residual test passes on fhb (TS replaced by
  constants in a test copy) and on a small synthetic coupling model with RIV,
  DRN and EVT (confined, steady and transient; MF6 files in `test/`, reference
  heads from mf6 6.8.2; T3.4 later generalises it). advtidal needs convertible
  cells and moves to T6.1.

**T2.5 STO and IC (S) ‖ with T2.4**
- Depends: T2.3.
- Read: `gwf-sto.f90` (`sto_fc`, `sto_cq`; `STORAGECOEFFICIENT`,
  `SS_CONFINED_ONLY`, steady/transient per period), `gwf-ic.f90`.
- Deliver: confined storage terms (convertible storage is T6.1, rejected for
  now), storage budget terms, and IC.
- Done when: unit tests pass and the fhb residual test passes (with TS in T6.3,
  or with TS replaced by constants in a test copy).

**T2.6 Assembly and matrix oracle (M)**
- Depends: T2.2–T2.5.
- Deliver: a small parametric device view of the CSR structure (`ia`, `ja`,
  `jas`, `isym` as needed; `Adapt.@adapt_structure`) next to the host
  `Connectivity`, then `src/solver/csr.jl` assembly kernel (off-diagonals from `condsat`,
  diagonal = −Σ, plus boundary and storage terms), the flux-form residual
  kernel (section 3.7, rule 2), and `test/parity/assembly_test.jl` with:
  1. the universal residual test (section 4) on every Tier 0–2 example, with
     MF6 heads converted to offsets in Float64 before the cast to `T`;
  2. `amat`/`b'` vs libmf6 after the first `solve` where the address exists.
  3. A cancellation test: a column of cells at 5000 m with 1 cm head
     differences, where the `:single` flux-form residual must match the
     `:double` one within 1e-4 relative. Show in the PR that `b − A·h` in
     Float32 fails this test.
- Done when: all three pass in `:single` and `:double` on all loadable Tier 0–2
  examples and the synthetic coupling model.

**T2.7 Linear solvers (M)**
- Depends: T2.6.
- Read: `src/Solution/LinearMethods/ImsLinearBase.f90` (`ims_base_cg`,
  `ims_base_bcgs`, `ims_base_testcnvg`, scaling), `ImsLinear.f90`, IMS section
  of the IO guide.
- Deliver: `src/solver/krylov.jl`: PCG (on −A, which is SPD for symmetric
  problems) and BiCGSTAB, Jacobi preconditioner, `SCALING_METHOD DIAGONAL`,
  IMS inner convergence tests, iteration counts reported per call. Map IMS
  `LINEAR_ACCELERATION` and `COMPLEXITY` defaults to our choices. ILU0, MILU0
  and ILUT map to Jacobi plus a logged warning until Phase 4.
- Generic in `Ts`. Dot products and norms follow section 3.7, rule 4.
- Done when: on random SPD and nonsymmetric M-matrices the solution agrees with
  `\` within 1e-10 (Float64) or 1e-4 (Float32) relative, and stopping
  criteria agree with MF6 semantics in unit tests.

**T2.8 Picard loop, time stepping, budgets, output (L)**
- Depends: T2.7.
- Read: `src/Solution/NumericalSolution.f90` (`sln_ca`, `solve`,
  `sln_underrelax`, `sln_backtracking`, convergence tables), `src/Timing/tdis.f90`
  (`nstp`, `tsmult`), `gwf.f90` (`gwf_cq`, `gwf_bd`), `gwf-oc.f90`, `Budget.f90`.
- Deliver: `src/solver/picard.jl` with the correction solve (section 3.7,
  rule 3), `underrelax.jl` (SIMPLE, COOLEY, DBD). **Backtracking is not
  implemented here; it comes with T6.2.** IMS `COMPLEXITY MODERATE`/`COMPLEX`
  (used by the coupling model) enable backtracking by default
  (`backtracking_number > 0`), so tests on such models may differ from MF6
  until T6.2. Log a warning when an input would backtrack, and record any such
  difference (and whether MF6's run actually backtracked) in the test and
  `docs/precision.md` rather than failing on it. Also conversion of offsets to
  absolute Float64 heads on the host for all output,
  `src/budget.jl` (FLOW-JA-FACE, per-package in/out, storage, percent
  discrepancy; CHD as in `calc_chd_rate`: the listing's CHD in/out is the
  per-connection split over connections to active cells, the CBC CHD `q` is the
  per-cell net, and the rate is added to `flowja(idiag)`), OC (SAVE/PRINT HEAD/BUDGET ALL/LAST/FREQUENCY/STEPS), HDS/CBC
  writers, a short listing file with budget tables, and `ModflowGPU.run(simdir)`.
- Also deliver `docs/precision.md`: the Float32 closures (section 3.7), and the
  measured `:single` vs `:double` and `:single` vs MF6 differences on every
  parity model. These confirm or revise the `:single` thresholds of section 4.
- Done when: Tier 0 examples and the small synthetic coupling model reproduce MF6
  tight-closure heads and budgets within section 4 tolerances in both
  presets (`test/parity/`), and `ModflowGPU.run` produces files readable by flopy.
