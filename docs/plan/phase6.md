# Phase 6: generic MF6 GWF (Milestone C); cards are independent once Phase 3 is done

Task cards for this phase of [PLAN.md](../../PLAN.md). Read PLAN.md sections 1–4 and 6 for the goals, conventions, architecture and validation rules these cards refer to.

Size: S ≈ < 1 day of agent work, M ≈ 1–3 days, L ≈ > 3 days.
`‖` marks cards that can run in parallel with their siblings.

**T6.1 Convertible cells and rewetting (L) ‖**
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
- Done when: Tier 3 examples pass parity.

**T6.2 Newton–Raphson (L) ‖**
- Read: `gwf-npf.f90` (`npf_fn`), `gwf-sto.f90` (`sto_fn`), boundary `*_fn`,
  `SmoothingFunctions.f90`, `NumericalSolution.f90` (Newton under-relaxation,
  backtracking, `UNDER_RELAXATION DBD`).
- Deliver: Newton terms (nonsymmetric → BiCGSTAB, AMG/ILU preconditioning of the
  nonsymmetric matrix), backtracking (also for non-Newton models with
  `COMPLEXITY MODERATE`/`COMPLEX`, deferred from T2.8), and `NEWTON
  UNDER_RELAXATION`.
- Done when: Tier 4 examples pass parity.

**T6.3 Time series and time-array series (M) ‖**
- Read: `src/Utilities/TimeSeries/*.f90`.
- Deliver: TS6 files (`STEPWISE`, `LINEAR`, `LINEAREND`, `SFAC`) and TAS6,
  evaluated per time step on the host and copied to device `bound` arrays.
- Done when: fhb and advtidal pass parity.

**T6.4 DISV and DISU (M) ‖**
- Read: `Disv.f90`, `Disu.f90`, `Connections.f90` (`disvconnections`,
  `disuconnections`, `anglex`).
- Deliver: connectivity for vertex and unstructured grids, `anglex` for the K
  ellipse, and HDS/CBC layouts for these grids.
- Done when: Tier 5 examples pass parity.

**T6.5 XT3D (L)**
- Depends: T6.4.
- Read: `src/Model/ModelUtilities/Xt3dInterface.f90`, `Xt3dAlgorithm.f90`.
- Deliver: the extended connection stencil (CSR with neighbours of neighbours,
  `XT3D` and `RHS` options), formulated per cell in a kernel.
- Done when: Tier 6 examples pass parity.

**T6.6 Multi-model solutions and GWF–GWF exchanges (L)**
- Read: `src/Exchange/exg-gwfgwf.f90`, `src/Solution/NumericalSolution.f90`
  (global matrix across models), `GwfGwfConnection.f90` (interface model).
- Deliver: several GWF models in one solution, exchanges with `CDIST`, `ANGLDEGX`,
  `IHC`, `XT3D` on the exchange, and a self-exchange (spbc).
- Done when: Tier 7 examples pass parity. This is the base for T7.1.

**T6.7 OBS for GWF heads and flows (S) ‖**
- Deliver: `HEAD`, `DRAWDOWN`, `FLOW-JA-FACE` and package observations, written
  as CSV like MF6.
- Done when: observation files of advtidal, fhb and rad-disu match within
  tolerance.
