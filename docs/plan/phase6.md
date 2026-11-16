# Phase 6: generic MF6 GWF (Milestone C)

Task cards for this phase of [PLAN.md](../../PLAN.md). Read PLAN.md sections 1–4 and 6 for the goals, conventions, architecture and validation rules these cards refer to.

Size: S ≈ < 1 day of agent work, M ≈ 1–3 days, L ≈ > 3 days.
`‖` marks cards that can run in parallel with their siblings.

All cards need Phase 3. Dependencies inside the phase are on each card; they follow the
`requires` field of each example in `validation/examples.toml` (checked 2026-11-11).
T6.0, T6.1, T6.2a, T6.3 and T6.4 depend on no other Phase 6 card.

**T6.0 Parity on the confined lgrv examples (S) ‖ (done 2026-11-14)**
- Depends: T4.5 (no Phase 6 card).
- `ex-gwf-lgrv-gc` and `ex-gwf-lgrv-gr` have `requires = []`: ICELLTYPE 0 everywhere,
  RIV, RCHA, K33OVERK, and `-gr` is 25×183×147. The strict loader accepts them, but no
  parity test runs them. They are the only real MF6 decks with RIV and RCHA at size
  that the Milestone A engine supports (PLAN.md section 4.1).
- Deliver: both examples in the universal residual test (`test/parity/assembly_test.jl`)
  and in the run parity test (`test/parity/run_test.jl`) against
  `reference/runs/<example>/tight/`, in `:single`, `:mixed` and `:double`, with the
  default preconditioner, on CPU and rocm. If `-gr` is too slow for the default CPU
  suite, make it opt-out like hwell-a (`MFGPU_TEST_HWELL`). Add the measured numbers to
  `docs/precision.md` and both examples to "Validated models" in
  `docs/support_matrix.md`.
- From the T5.3 report: T5.3 (merged first) and this card both changed the Float32
  convergence test in `src/solver/picard.jl` for the same stall (a converged state whose
  first inner correction is above the fixed 1e-6 m INNER_DVCLOSE floor). T5.3 raises
  INNER_DVCLOSE to OUTER_DVCLOSE once the outer test is met and only the inner test
  fails, which also loosens the inner test when OUTER_DVCLOSE is loose. This card
  floors INNER_DVCLOSE at `eps32·max|h'|` under STRICT. Reconcile them into one rule
  that passes both the Indus coupling parity (patch 0002) and the lgrv tests.
- Done when: the section 4 heads and budget criteria pass in all three presets on CPU
  and rocm. A failure is fixed in this card if it is a bug in a supported feature, and
  otherwise reported with the missing feature named.

**T6.1 Convertible cells and rewetting (L) ‖ (done 2026-11-16)**
- Depends: no Phase 6 card.
- Read: `gwf-npf.f90` (`sat`, `THICKSTRT`, `VARIABLECV [DEWATERED]`,
  `PERCHED`, `wd` rewetting: `WETFCT`, `IWETIT`, `IHDWET`), `gwf-sto.f90` (Sy,
  convertible storage).
- Deliver: saturation-dependent conductance recomputed per outer iteration in a
  kernel (`condsat` stays constant; a per-connection saturation factor is
  applied), convertible storage, and rewetting as a per-cell kernel with a
  neighbour check, run between outer iterations as MF6 does.
- From the T2.3 review: RCH places records on the highest active cell once per period
  (`apply_period!`); MF6 redoes it in every `rch_cf` from `nodesontop`, which matters once
  IBOUND changes (dry cells). Also port the GHB input checks (`ghb_ck`: negative cond or
  multiplier, bhead below the bottom of a convertible cell). WEL `AUTO_FLOW_REDUCE` too.
- From the T2.4 review: RIV/DRN check negative conductance after the multiplier; MF6
  checks raw cond and the multiplier separately. Port the cell-bottom `*_ck` checks for
  convertible cells, and move EVT to the highest active cell when IBOUND changes.
- From the T4.5 review: the off-diagonals are rewritten only when `m.offdiag` says so
  (`src/gwf/model.jl`). Head-dependent conductance must set `m.offdiag = nothing` every
  outer iteration, or pass `offdiag = true`.
- From the T4.6 report: the period-data fast path keeps moved RCH/EVT nodes when a
  block applied with an `ibound` is followed by the same cells without one. The model
  path always passes `ibound = nothing` today; this becomes reachable once IBOUND
  changes.
- From the T6.3 merge (2026-11-16): this card owns `ex-gwf-advtidal` now. Also:
  (1) record checks (`riv_ck` etc.) on time-series values use the average over the
  period's first step, while MF6 uses the value at time 0 in period 1 and the previous
  step's average later: matters for the cell-bottom checks of this card;
  (2) `reference_rundir` (test helper) can fall back to the `relaxed` reference run:
  make that explicit per example before advtidal and maw-p03c use it;
  (3) a file named twice in TS6 FILEIN is not rejected (MF6: "Found duplicate
  time-series file name");
  (4) a package with series is restaged from the host every step, which overwrites
  device-only BOUND writes (`device_bound!`) to its other columns.
- From the T6.4 merge (2026-11-16): this card owns `ex-gwf-rad-disu` (convertible
  storage; `rad_disu_confined` is the ICONVERT 0 stand-in). (1) Its `:single` CBC
  tolerance of 5e-4 × total inflow has little headroom (a reviewer variant with 9
  excluded cells measured 7.5e-4): replace it with a bound from Float32 rounding of
  the cell flows. (2) Default MC-ILU0 + BiCGSTAB stalls on this radial grid (no fixed
  heads, nearly singular): use AMG or look at the ordering. (3) No run parity test
  covers DISU staggered horizontal connections (IHC=2): add a small one (the reviewer
  checked a 4-cell model by hand: 0 in `:double`).
- Done when: the Tier 3 examples pass parity. `ex-gwf-advtidal` also needs T6.3: it is
  owned by whichever of T6.1 and T6.3 merges last, and the first of the two leaves it
  out.

**T6.2 Newton–Raphson (L)**
- Depends: T6.1 for the Newton part: all eight Tier 4 examples require `convertible`
  cells, and the Newton terms are the derivatives of the saturation functions that T6.1
  adds. The backtracking part depends on no Phase 6 card: deliver it first on its own
  branch as **T6.2a** (S, ‖). It removes the "backtracking is not applied" warning for
  `COMPLEXITY MODERATE`/`COMPLEX` models, among them the coupling model.
- From the T6.1 merge (2026-11-16): PERCHED + VARIABLECV DEWATERED gives NaN when a
  cell sits exactly at its bottom (saturation ~1e-305, so `hwva/(sat·thk/2/K)`
  overflows); MF6 stays finite on the same input. Add a guard (floor on the
  saturation or the denominator) and a test. Reproducer (local, not committed):
  `reference/issues/T6.1-perched-dewatered-nan`. Also: `run(simdir)` writes output
  next to the inputs by default, which overwrote a reference run once; consider a
  guard against writing into `REFERENCE_DIR`.
- Read: `gwf-npf.f90` (`npf_fn`), `gwf-sto.f90` (`sto_fn`), boundary `*_fn`,
  `SmoothingFunctions.f90`, `NumericalSolution.f90` (Newton under-relaxation,
  backtracking, `UNDER_RELAXATION DBD`).
- Deliver: Newton terms (nonsymmetric → BiCGSTAB, AMG/ILU preconditioning of the
  nonsymmetric matrix), backtracking (also for non-Newton models with
  `COMPLEXITY MODERATE`/`COMPLEX`, deferred from T2.8), and `NEWTON
  UNDER_RELAXATION`.
- Done when: T6.2a: the coupling_rde COMPLEX run backtracks where MF6 does (MF6: in 3
  of 125 outer iterations, `docs/precision.md`), the warning is gone, and the run
  parity tests still pass. T6.2: the Tier 4 examples pass parity.

**T6.3 Time series and time-array series (M) ‖ (done 2026-11-16)**
- Depends: no Phase 6 card.
- Read: `src/Utilities/TimeSeries/*.f90`.
- Deliver: TS6 files (`STEPWISE`, `LINEAR`, `LINEAREND`, `SFAC`) and TAS6,
  evaluated per time step on the host and copied to device `bound` arrays.
- Done when: fhb passes parity (replacing the `fhb_const` stand-in of Tier 2), and
  advtidal too if T6.1 is merged (see T6.1).

**T6.4 DISV and DISU (M) ‖ (done 2026-11-16)**
- Depends: no Phase 6 card.
- Read: `Disv.f90`, `Disu.f90`, `Connections.f90` (`disvconnections`,
  `disuconnections`, `anglex`).
- Deliver: connectivity for vertex and unstructured grids, `anglex` for the K
  ellipse, and HDS/CBC layouts for these grids.
- Done when: Tier 5 examples pass parity.

**T6.5 XT3D (L)**
- Depends: T6.4 for the DISV examples only (`ex-gwf-u1disv-x`, `ex-gwf-disvmesh`).
  `ex-gwf-hanix` and `ex-gwf-whirl` are DIS models and need no other Phase 6 card, so
  the card can start on DIS before T6.4 is merged.
- Read: `src/Model/ModelUtilities/Xt3dInterface.f90`, `Xt3dAlgorithm.f90`.
- Deliver: the extended connection stencil (CSR with neighbours of neighbours,
  `XT3D` and `RHS` options), formulated per cell in a kernel.
- Done when: Tier 6 examples pass parity.

**T6.6 Multi-model solutions and GWF–GWF exchanges (L)**
- Depends: T6.5 for `ex-gwf-u1gwfgwf-s2`, `-s3` and `-s4` (XT3D in NPF or on the
  exchange). `-s1`, `ex-gwf-lgrv-lgr` and `ex-gwf-spbc` need no other Phase 6 card: all
  Tier 7 models are DIS, so T6.4 is not a dependency.
- Read: `src/Exchange/exg-gwfgwf.f90`, `src/Solution/NumericalSolution.f90`
  (global matrix across models), `GwfGwfConnection.f90` (interface model).
- Deliver: several GWF models in one solution, exchanges with `CDIST`, `ANGLDEGX`,
  `IHC`, `XT3D` on the exchange, and a self-exchange (spbc).
- Done when: Tier 7 examples pass parity.

**T6.7 OBS for GWF heads and flows (S)**
- Depends: T6.3 for fhb; T6.1 and T6.3 for advtidal; T6.1 and T6.4 for rad-disu
  (DISU with convertible storage). The OBS code itself can be written and unit-tested
  on the Milestone A models before those cards.
- Deliver: `HEAD`, `DRAWDOWN`, `FLOW-JA-FACE` and package observations, written
  as CSV like MF6.
- From the T6.4 merge: `write_simulation` and the BMI grid x/y support DIS only; GRB
  files are read but not written; DATA-SPDIS is not written for any grid.
- From the T6.1 merge: (1) record checks (`riv_ck` etc.) on time-series values use the
  average over the period's first step; MF6 uses the value at time 0 in period 1 and
  the previous step's average later. (2) A package with time series is restaged from
  the host every step, and `move_to_highest_active!` restages RCH/EVT when IBOUND
  changes: both overwrite device-only BOUND writes (`device_bound!`); needs a partial
  restage. (3) The `:single` cell-by-cell budgets of advtidal, maw-p03c and rad-disu
  are not verified (`docs/precision.md`). (4) Cell-conversion messages are not written
  to the listing.
- Done when: observation files of advtidal, fhb and rad-disu match within
  tolerance.
