# ModflowGPU: plan for a GPU port of MODFLOW 6 groundwater flow

This plan is written to be executed by coding agents, one task card at a time.
The cards are in `docs/plan/phase<N>.md` (section 5). Each card lists its
dependencies, what to read, what to deliver, and how to prove it is done. Do not start a card before its dependencies are merged.

Working package name: **ModflowGPU.jl** (module `ModflowGPU`). It lives in the
root of this repository (`mf-gpu`). Renaming it later is a search-and-replace.

---

## 1. Goal and scope

**Goal.** Run the groundwater-flow (GWF) part of MODFLOW 6 on GPUs from Julia,
with KernelAbstractions.jl (KA) kernels and GPU backends loaded through package
extensions, as in [mGV](https://github.com/SWIFT-WUR/mGV). The first production
user is [ModflowCoupling.jl](https://github.com/SWIFT-WUR/Modflow-coupling),
which couples MODFLOW 6 to mGV through BMI for basin to global models at 5 arcmin.

**Correctness standard.** MODFLOW 6 itself is the reference. Every feature is
accepted only when it reproduces MF6 heads and budgets on MF6 input files
(section 4).

**Milestones**

| Milestone | Content | Why |
| --- | --- | --- |
| **A: coupling feature set** | DIS, NPF (confined), STO, IC, CHD, WEL, RCH, GHB, RIV, DRN, EVT, OC, Picard outer loop, CG/BiCGSTAB, budgets | Everything ModflowCoupling uses (see 1.1) |
| **B: coupled production runs** | ModflowCoupling with `engine = "gpu"`, mGV and MODFLOW on one GPU, global 5′ steady state + daily transient | The scientific use case |
| **C: generic MF6 GWF** | Convertible cells, rewetting, Newton, time series, DISV/DISU, XT3D, GWF–GWF exchanges, OBS | Wider use and the MF6 example suite |
| **D: 30″ resolution (stretch)** | Regional 30″ on one GPU, then global 30″ through multi-GPU domain decomposition | Higher-resolution global modelling (Phase 7) |

**Out of scope** (document it; do not build): SFR, LAK, MAW, UZF, MVR, CSUB,
BUY, VSC, API package, GWT, GWE, PRT, and PETSc parallel runs. These are
sequential or very branchy, and the coupling does not need them.

### 1.1 What the coupling model needs (from `ModflowCoupling/src/mf6_writer.jl`)

- `DIS`: 2 layers, uniform `DELR`/`DELC`, `IDOMAIN` 0/1/2 (2 = CHD cells), `NOGRB`.
- `NPF`: `ICELLTYPE 0` (confined everywhere), `K`, `K33`, default CELLAVG.
- `STO`: `STORAGECOEFFICIENT`, `ICONVERT 0`, `SS`; `TRANSIENT` or `STEADY-STATE`.
- `IC`, `OC` (`PRINT BUDGET ALL`, optional `SAVE HEAD/BUDGET`).
- `CHD` in both layers; `RCH` list on all active cells; `RIV` (stage set each
  period through BMI); `DRN` (static); `EVT` with `NSEG 1` and `FIXED_CELL`;
  `WEL` (optional).
- `IMS`: `COMPLEXITY COMPLEX`, `BICGSTAB`, `UNDER_RELAXATION SIMPLE`, and loose
  closures (`outer_dvclose = 5`, `inner_rclose = 10 STRICT`).
- Libmf6 access pattern: `prepare_time_step` → write `RCH/BOUND`, `RIV/BOUND`,
  `WEL/BOUND` → `prepare_solve` → `solve` until converged → `finalize_*` → read
  `X`, `RIV/SIMVALS`, `DRN/SIMVALS`, `EVT/SIMVALS`.

The flow problem is **linear except for the head-dependent boundaries** (RIV
switching at RBOT, DRN on/off, EVT piecewise). For a confined model with
CG-compatible boundaries the matrix is symmetric, so preconditioned CG applies.

### 1.2 Sizes and the hardware

- Development GPU: **AMD Radeon RX 7800 XT**, gfx1101, 16 GB, wavefront 32;
  ROCm 7.2.2 in `/opt/rocm`; Julia 1.12.6 (juliaup); AMDGPU.jl 2.10.0 with
  KernelAbstractions 0.10.3 works (checked 2026-11-06).
- **Float32 and Float64 are both supported from the start** (section 3.7).
  Float32 is required on backends without Float64 (Metal, Intel Arc via
  oneAPI, some OpenCL devices). On every backend it halves memory and cuts
  matrix traffic by about a third. Float64 gives exact MF6 parity. RDNA3 runs
  Float64 at a small fraction of its Float32 rate, but sparse solvers are
  limited by memory bandwidth, so both are viable here. Hardware Float64
  atomics are not guaranteed on this card, so the design avoids atomics
  (section 3.4).
- Global 5′ grid: 4320 × 2160 × 2 ≈ 18.7 M cells, of which roughly a third
  are land. That gives ~5–7 M active nodes with about 6 non-zeros per row. CSR in
  Float64 + Int32 plus ~15 Krylov/AMG work vectors is about 2–3 GB, which fits
  in 16 GB next to mGV. Measure this; mGV's own global footprint is unknown.
- Global 30″ (43200 × 21600 × 2 ≈ 1.9 G cells, ~0.5–0.7 G active nodes) needs
  roughly 50–80 GB and does not fit one 16 GB GPU. It needs multi-GPU
  (Phase 7). A regional 30″ model does fit: the Indus at 30″ is 2040 × 1800 × 2
  ≈ 7.3 M cells, about the size of global 5′.

---

## 2. Conventions for every agent

- **Read before you write:** the MF6 Fortran source for the feature (cloned at
  `reference/modflow6`, see T0.2). Port behaviour, not structure. Cite the
  Fortran routine in a comment where the formula comes from, as
  ModflowCoupling does for the Python reference.
- **Style:** match mGV: Runic formatting (`runic --inplace .`), docstrings on
  public functions, `@kernel` functions named `*_kernel!` with a launcher function
  next to them, `@Const` for read-only kernel arguments, `get_backend(array)` to
  launch, `Adapt.@adapt_structure` to move structs to the device, and `copyto!`
  (not slicing) to update device arrays.
- **Backends:** the core package depends only on KernelAbstractions. GPU
  packages are weak dependencies; `ext/AMDGPUExt.jl`, `CUDAExt.jl`,
  `MetalExt.jl`, `oneAPIExt.jl` and `OpenCLExt.jl` each call
  `register_backend!` in `__init__`. Copy `mGV/src/backend_setup.jl` and adapt
  it, keeping the same names (`select_backend`, `register_backend!`,
  `backend_options`).
- **Precision:** Float32 and Float64 are both first-class from the first
  kernel on (section 3.7). Engine structs are parametric in `T` (state and
  assembly), `Ts` (Krylov), `Tp` (preconditioner) and `Ti` (indices, default
  `Int32`). Every unit and parity test runs for the `:single` and `:double`
  presets, plus `:mixed` once T4.4 adds it. **A card is not done if it passes
  only in Float64.** Follow the formulation rules of section 3.7 in every
  kernel. Host-side parsing, specs and setup stay Float64.
- **No scalar indexing on device arrays.** GPU tests run with
  `AMDGPU.allowscalar(false)` (or the backend's equivalent).
- **Host vs device:** input parsing, connectivity building, graph colouring and
  AMG setup run on the host. Everything that runs once per outer iteration
  (assembly, boundary formulation, SpMV, preconditioner application,
  convergence checks, budgets) runs in KA kernels.
- **Tests:** `julia --project -e 'using Pkg; Pkg.test()'` runs on the CPU
  backend and must pass in CI without a GPU. **GPU tests run only on this
  machine** (RX 7800 XT), never in CI: `MFGPU_TEST_BACKEND=rocm julia
  --project=test test/runtests.jl`. Their results must match the CPU backend
  within 1e-10 relative for `:double`, and within 1e-4 m in heads for `:single`
  (reductions run in a different order). Every PR that touches kernels, solver or assembly
  pastes the local GPU test summary into its description. Parity tests that need
  `reference/` skip with a message when it is missing.
- **Data:** Indus static inputs are in `/home/bart/data/modflow/data`. Tests
  find them through `INDUS_DATA=/home/bart/data/modflow/data` and skip with a
  message when it is unset. Never copy the data into this repository. Global
  inputs are not available yet; use the synthetic models of T3.4 instead.
- **Git:** one branch and one PR per card, titled `T<id>: <name>`. Do not
  commit anything under `reference/` (it is gitignored).
- **ModflowCoupling changes are stored as patch files in this repository**
  (`patches/modflow-coupling/`), never pushed to SWIFT-WUR/Modflow-coupling and
  never made in `/home/bart/data/modflow/ModflowCoupling` (the owner's checkout).
  Workflow:
  1. `validation/fetch_reference.sh` clones
     `https://github.com/SWIFT-WUR/Modflow-coupling` into
     `reference/Modflow-coupling` and checks out the base commit
     `8dd8c010796fd3648b855bed51195c0f2e61dd60` (2026-10-19).
  2. `scripts/apply_coupling_patches.sh` resets that clone to the base commit
     and applies `patches/modflow-coupling/*.patch` in order with `git am`.
  3. Make changes there as commits, then export them with
     `git format-patch <base>..HEAD -o patches/modflow-coupling/`, with one
     numbered patch per card (`0001-T5.1-engine-switch.patch`, …). Commit the
     patch files in this repository.
  4. The patched clone uses ModflowGPU from this repository through a
     `[sources]` entry (`ModflowGPU = {path = "../.."}`), added by the first patch.
  `patches/modflow-coupling/README.md` lists each patch, its card, and the base
  commit. If the base must move to a newer upstream commit, rebase all patches
  and update the README in one PR.
- **Report:** finish each card with a short note in the PR: what was ported
  (Fortran routines), test results with numbers, deviations from MF6, open
  issues.

---

## 3. Architecture

### 3.1 Repository layout

```
mf-gpu/
├── Project.toml            # ModflowGPU; [weakdeps] AMDGPU, CUDA, Metal, oneAPI, OpenCL
├── src/
│   ├── ModflowGPU.jl
│   ├── backend_setup.jl    # from mGV
│   ├── input/              # MF6 file reader (host only): tokenizer, blocks, arrays, lists, dfn, simulation
│   ├── spec/               # host-side model description: plain structs, no device arrays
│   ├── discretization/     # dis.jl, disv.jl, disu.jl → Connectivity
│   ├── gwf/                # npf.jl, sto.jl, ic.jl
│   ├── boundaries/         # framework.jl, chd, wel, rch, ghb, riv, drn, evt
│   ├── solver/             # csr.jl, spmv.jl, krylov.jl, precond/, picard.jl, underrelax.jl, scaling.jl
│   ├── budget.jl
│   ├── output/             # hds.jl, cbc.jl, listing.jl (writers and readers)
│   ├── builder.jl          # in-memory model construction (no files)
│   ├── stepping.jl         # libmf6-like API: prepare_time_step, solve, …, get_value_ptr
│   └── bmi.jl              # BasicModelInterface
├── ext/                    # backend extensions
├── test/                   # Project.toml (workspace), unit/, parity/, gpu/
├── validation/             # fetch_reference.sh, run_reference.jl, compare.jl, examples.toml
├── benchmark/              # synthetic.jl, run.jl, results/*.md
├── docs/                   # design notes, MF6 deviations, support matrix
├── patches/modflow-coupling/  # git format-patch series for ModflowCoupling + README
├── scripts/                # apply_coupling_patches.sh and other helpers
└── reference/              # gitignored: mf6 binaries, example inputs, MF6 source,
                            #   reference runs, Modflow-coupling clone
```

### 3.2 Data flow

```
MF6 input files ──► input/ parser ──┐
                                    ├──► spec::SimulationSpec (host) ──► build ──► GwfModel{backend}
ModflowCoupling StaticParameters ───┘        (builder.jl, no files)                 │
                                                                                     ▼
                     time loop: TDIS → per period: read/apply stresses → Picard outer loop
                       outer: formulate (NPF, STO, boundaries) → assemble CSR → linear solve
                              → under-relax → convergence → budgets → output
```

The **spec** layer separates file formats from the engine. The MF6 reader and
ModflowCoupling's in-memory builder both produce a `SimulationSpec`. At global
scale, writing and parsing ASCII MF6 input (GBs of text) is too slow, so the
builder path is required, not optional.

### 3.3 Matrix: MF6-compatible CSR on reduced nodes

- Use MF6's own system layout: active ("reduced") nodes only, CSR `ia`/`ja`
  with the **diagonal stored first in each row**, plus `jas` (symmetric
  connection index) for per-connection arrays. Keep MF6's sign convention
  (negative diagonal).
  - This lets tests compare our `amat`/`rhs` directly with libmf6 (T2.6).
  - One format covers DIS, DISV, DISU, XT3D and GWF–GWF exchanges.
  - Julia `A[j, i, k]` column-major linear indices equal MF6 user node numbers
    (`(k-1)·nrow·ncol + (i-1)·ncol + j`), so mapping user ↔ reduced nodes is a
    pair of index arrays (`nodeuser`, `nodereduced`).
- Per connection, store saturated conductance `condsat` once (confined
  conductances never change) and the reference difference `Δr` (section 3.7).
  Per outer iteration only the diagonal and RHS change for the coupling model,
  so assembly is a cheap kernel.
- With reference offsets (section 3.7) the matrix `A` is the same as MF6's.
  Only the right-hand side changes, to `b' = b − A·r`, which tests compute in
  Float64 when comparing with libmf6.
- A structured-grid matrix-free stencil or SELL-C-σ format is a **later
  optimisation** (T4.5), adopted only if a benchmark shows it is worth it.

### 3.4 Boundary packages without atomics

Each list package keeps MF6's layout: `nodelist[nbound]`, `bound[ncol, nbound]`,
`hcof[nbound]`, `rhs[nbound]`, `simvals[nbound]`. A formulate kernel runs per
record (computes `hcof`/`rhs` from the current head). An add kernel then runs per
node over a node→record CSR map built at setup, adding the sums to the diagonal
and RHS. Packages are applied one after another. This avoids atomics and is
deterministic, which keeps CPU/GPU comparisons tight. Head-like columns of
`bound` (CHD/GHB head, RIV stage and RBOT, DRN elevation, EVT surface) hold
offsets from the cell's reference (section 3.7), not absolute elevations.

### 3.5 Solver stack

1. **Outer loop (Picard)** as in IMS `NumericalSolution`: formulate,
   assemble, compute the flux-form residual, linear solve for the **head
   correction** (section 3.7; MF6 solves for heads), under-relaxation (`NONE`,
   `SIMPLE`, `COOLEY`, `DBD`), and the `outer_dvclose` check on the correction.
   Add backtracking and Newton in Phase 6.
2. **Krylov:** hand-written PCG and BiCGSTAB in `Ts` with IMS convergence semantics
   (`inner_dvclose`, and `inner_rclose` with its `STRICT` / `L2NORM_RCLOSE` /
   `RELATIVE_RCLOSE` variants). Read `ImsLinearBase.f90` for the exact tests.
   Hand-written, not Krylov.jl, so the stopping rules match MF6 and kernels can be fused.
   Krylov.jl may be used as a test oracle.
3. **Preconditioners:** first Jacobi (T2.7). Then Chebyshev polynomial,
   multicolour symmetric Gauss–Seidel / ILU0, and smoothed-aggregation AMG
   (T4.2–T4.4). AMG setup runs on the host, and the hierarchy is copied to the
   device as CSR. For confined models the off-diagonals are constant, so the
   hierarchy is reused across outer iterations and time steps.
4. **Precision:** all of the above in `:single` and `:double` from the start
   (section 3.7). T4.4 adds the `:mixed` preset and picks the production
   default from measurements.

### 3.6 Public APIs

- `ModflowGPU.run(simdir; backend = "auto")`: run an MF6 simulation directory and
  write `.hds`/`.cbc`/`.lst`, like `mf6`.
- `ModflowGPU.Simulation(spec; backend)`: build a model from a spec in memory.
- **Stepping API (libmf6-like):** `initialize`, `prepare_time_step`,
  `prepare_solve`, `solve`, `finalize_solve`, `finalize_time_step`,
  `get_value_ptr(sim, "GWF/X")`, `"GWF/RIV/BOUND"`, `"GWF/DRN/SIMVALS"`, … Use the
  same addresses as libmf6, so ModflowCoupling's `advance!` can switch engines
  with few changes. `get_value_ptr` returns device arrays in the engine's
  precision; head-like arrays (`X`, head columns of `BOUND`) are offsets, and
  the reference is exposed as `"GWF/HREF"`. `get_value`/`set_value` copy to and
  from the host and convert between offsets and absolute Float64 values.
- `BasicModelInterface` methods for general use, always in absolute values.

### 3.7 Precision: Float32 and Float64 from the start

Every kernel, test and benchmark runs in both precisions from Phase 2 on. The
preset is a keyword (`precision = :auto`) and a config option:

| Preset | State and assembly `T` | Krylov `Ts` | Preconditioner `Tp` | Use |
| --- | --- | --- | --- | --- |
| `:double` | Float64 | Float64 | Float64 | Exact MF6 parity |
| `:single` | Float32 | Float32 | Float32 | Backends without Float64; less memory everywhere |
| `:mixed` (T4.4) | Float64 | Float64 | Float32 | Float64 accuracy with a cheaper preconditioner |
| `:auto` | `:double` if `supports_float64(backend)`, else `:single` | | | Until T4.4 sets the production default |

`supports_float64(::Backend)` is an extension hook like mGV's `pin_memory!`.
CPU, CUDA and ROCm return `true`, Metal returns `false`, and oneAPI and OpenCL
query the device (`fp64` / `cl_khr_fp64`). Requesting `:double` or `:mixed` on a
device without Float64 is an error that names `:single`.

**Formulation rules.** They apply in every preset: they cost little in Float64
and make Float32 accurate.

1. **Reference offsets.** On the host, compute per cell
   `r_n` in Float64: `r_n = STRT_n` for confined cells (ICELLTYPE 0), falling
   back to `clamp(STRT_n, bot_n, top_n)` when STRT is implausibly far from the cell
   (threshold chosen in T2.8); `clamp(STRT_n, bot_n, top_n)` for convertible cells
   (revisit in T6.1). Decided 2026-11-07 after T2.6: the clamp gave offsets up to
   10600 m on toth. `r` is not re-referenced between stress periods (revisit only if
   long transient runs drift; T4.4). The device stores heads as
   offsets `h'_n = h_n − r_n`. Every head-like value (CHD/GHB head, RIV stage
   and RBOT, DRN elevation, EVT surface, cell top and bottom, IC, old heads) is
   converted to its cell's offset on the host in Float64 before it is cast to
   `T`. Per connection, store `Δr_nm = r_m − r_n`. This keeps device values to
   tens of metres instead of thousands: the Float32 step size drops from
   ~0.5 mm at 5000 m to a few µm. Absolute heads exist only on the host, in
   Float64, as `r + h'`. All MF6 formulas used here depend on head
   differences, so they carry over unchanged. Check this for every new package.
2. **Flux-form residual.** Compute each row's residual from the connection
   terms `C_nm·(h'_m − h'_n + Δr_nm)` plus the boundary and storage terms, never
   as `b − A·h`, which cancels large numbers in Float32.
3. **Correction solve.** Each outer iteration solves `A·δ = residual` with a
   zero initial guess and updates `h' ← h' + relaxation·δ`. The outer loop is
   then iterative refinement: a Float32 Krylov solve that reduces the residual
   by only a few orders of magnitude still converges, because the residual
   itself is accurate (rules 1–2). `outer_dvclose` applies to `max |δ|`, which
   equals MF6's head change.
4. **Accumulation.** Budget totals, dot products and norms accumulate in
   Float64 where the backend has it, and otherwise in Float32 with pairwise or
   compensated (Kahan) summation. Simulation time and `delt` stay Float64 on
   the host; kernels receive `T(delt)`.
5. **Kernel hygiene.** No Float64 literals or Float64-returning calls in
   kernels: write `T(0.5)`, `one(T)`, `oftype(x, …)`, not `0.5`. A GPU test
   (T3.1) compiles every kernel for `:single` on rocm and fails if the device
   LLVM IR contains `double`. This is how Metal and oneAPI compatibility is
   checked on this machine, which has neither.

**Closures in Float32.** Float32 runs cannot reach `1e-9` closures. T2.8 sets
the tightest closures `:single` reliably reaches on Tier 0–2. A starting
guess: `OUTER_DVCLOSE 1e-5`, `INNER_RCLOSE 1e-6` relative. The result goes in
`docs/precision.md`, and the parity runs use it as the single-precision "tight"
setting.

---

## 4. Validation strategy

| Level | Test | Pass criterion |
| --- | --- | --- |
| Parsing | All in-scope example decks parse; arrays/lists equal a flopy-free independent read where available | exact |
| Connectivity | `ia`, `ja`, `ihc`, `cl1`, `cl2`, `hwva` vs libmf6 memory (`<model>/DIS/...`, `<model>/CON/...` addresses) | exact / 1e-14 rel |
| **Assembly, universal** | Plug MF6's converged heads (from a *tight-closure* reference run) into our assembly; residual `‖b − A·h_mf6‖∞ / max_row Σ|terms|` | < 1e-8 |
| Assembly, oracle | `amat` and `b'` (section 3.3) vs libmf6 `SLN_1/AMAT`, `SLN_1/RHS` after the first `solve` call (if exposed; see T0.2) | `:double` < 1e-12 rel; `:single` < 1e-6 rel |
| Heads | Our run vs MF6 run, both with tight closures | `:double`: max abs diff < 1e-6 m (or 1e-7 × head range); `:single`: < 1e-3 m |
| Budgets | Per-package in/out rates per stress period vs MF6 listing/CBC | `:double`: < 1e-6 rel, or 1e-6 × total in; `:single`: < 1e-4 × total in, and percent discrepancy < 0.01 % |
| Precision | Same model in `:single` vs `:double` | heads < 1e-3 m; package flows < 1e-4 × total in |
| Backends | rocm vs CPU backend, same model | `:double` < 1e-10 rel; `:single` heads < 1e-4 m |
| Coupling | ModflowCoupling Indus tests with `engine = "gpu"` vs `"mf6"` (tight closures) | `:double`: heads < 1e-4 m; baseflow, capillary rise < 1e-6 rel. `:single`: heads < 1e-3 m; baseflow, capillary rise < 1e-4 × domain maximum |

The `:single` thresholds are proposals. T2.8 checks them on Tier 0–2 and the
synthetic models, and T5.3 on the Indus. A card may tighten them, or loosen them
with a written reason in `docs/precision.md`.

**Tight-closure reference runs.** Production IMS settings (`outer_dvclose =
5 m` in ModflowCoupling) leave answers that depend on the solver path. All parity
runs therefore use a patched IMS: `OUTER_DVCLOSE 1e-9`, `INNER_DVCLOSE 1e-10`,
`INNER_RCLOSE 1e-10`, and enough iterations. Our `:single` runs use the Float32
closures of section 3.7 instead. `validation/run_reference.jl` writes
these runs to `reference/runs/<example>/tight/` and keeps the unpatched run in
`.../raw/`. Also report how far apart the engines are at production settings,
but do not use that for pass/fail.

**MF6 versions.** Use mf6 **6.8.2** (`mf6.8.2_linux.zip`, which includes
`libmf6.so`) for the examples, which flopy 3.10 generated for MF6 ≥ 6.7. Use
`Modflow6_jll` 6.3.0 where ModflowCoupling uses it. Note the 6.3 quirk that RCH
list rates are multiplied by the cell area on read (`calibrate_bmi!`). The
ModflowGPU engine stores rates per unit area and multiplies by the area during
formulation, like MF6 ≥ 6.4. Check the version difference against the RCH
source code.

### 4.1 Example ladder (MF6 examples, `mf6examples.zip` release asset)

Package lists come from a scan of the 2026-10-29 build of the examples,
checked against the inputs in T0.3 (`validation/examples.toml`, whose
`requires` field lists the features each example needs; filter on it rather
than on the tier alone).

| Tier | Feature focus | Examples (directory names in the zip) |
| --- | --- | --- |
| 0 | Confined, steady, CHD/WEL, anisotropy | `ex-gwf-toth` (DIS 200×1×100, CHD), `ex-gwf-hanic`, `ex-gwf-hanir` (K22/ANGLE1, CHD, WEL) |
| 1 | Coupling package set | **synthetic coupling models** (T3.4) at several sizes; Indus via ModflowCoupling's writer (needs data) |
| 2 | Time series, transient | `ex-gwf-fhb` (TS on CHD/WEL, STO) |
| 3 | Convertible / standard formulation / rewetting | `ex-gwf-twri01` (VARIABLECV DEWATERED PERCHED, DRN, RCHA), `ex-gwf-advtidal` (convertible layer 1, VARIABLECV DEWATERED PERCHED; GHB, WEL, RIV, 3× RCH, EVT NSEG 3, TS, steady→transient; also needs T6.3), `ex-gwf-bcf2ss-p01a` (REWET, RIV, CG), `ex-gwf-bump-p01b`, `ex-gwf-nwt-p02b` (14 lay, REWET); size tests that are confined (ICELLTYPE 0) despite their tier: `ex-gwf-lgrv-gc`, `ex-gwf-lgrv-gr` (25×183×147, RIV, RCH) |
| 4 | Newton | `ex-gwf-bcf2ss-p02a`, `ex-gwf-bump-p01a/c`, `ex-gwf-nwt-p02a`, `ex-gwf-nwt-p03a/b`, `ex-gwf-zaidel`, `ex-gwf-capture` |
| 5 | DISV / DISU | `ex-gwf-u1disv`, `ex-gwf-curve-90`, `ex-gwf-curvilin`, `ex-gwf-rad-disu` (DISU, STO, WEL, OBS) |
| 6 | XT3D | `ex-gwf-hanix`, `ex-gwf-u1disv-x`, `ex-gwf-whirl`, `ex-gwf-disvmesh` (DISV + XT3D, GHB, RCH) |
| 7 | Multi-model / exchanges | `ex-gwf-u1gwfgwf-s1` (no XT3D), `-s2` (XT3D in NPF), `-s3` (NPF and exchange), `-s4` (XT3D on the exchange only), `ex-gwf-lgrv-lgr`, **`ex-gwf-spbc`** (GWF–GWF exchange of a model with itself = periodic boundary; the template for global longitude wrap) |
| — | Out of scope | anything with SFR, LAK, MAW, UZF, MVR, CSUB (`sfr-*`, `lak-*`, `maw-*`, `drn-p01*`, `sagehen`, `csub-*`, `lgr`, `hwell-b`), all GWT/GWE/PRT |

`ex-gwf-hwell/ex-gwf-hwell-a` (11×118×118, STO, WEL) is an extra Tier 2 size
test once STO is done. `ex-gwf-maw-p03a/c` (no MAW package in those variants;
steady state without STO, convertible layer 1) are extra Tier 3 size tests.

---

## 5. Task cards

The cards live in one file per phase under `docs/plan/`. An agent working on a
card reads this file (sections 1–4 and 6) and its own phase file. Done cards are
marked; their merge commits on `main` hold the card reports.

- **Phase 0: setup and reference environment**: [docs/plan/phase0.md](docs/plan/phase0.md)
  - T0.1 Package scaffold (done)
  - T0.2 Reference toolchain (done)
  - T0.3 Example inventory and reference runs (done)
  - T0.4 GPU baseline numbers (done)
- **Phase 1: MF6 input and output (host only)**: [docs/plan/phase1.md](docs/plan/phase1.md)
  - T1.1 Tokenizer and block parser (done)
  - T1.2 Array input (done)
  - T1.3 List and period input (done)
  - T1.4 Simulation loader → `SimulationSpec` (done)
  - T1.5 Binary output readers/writers (done)
- **Phase 2: CPU engine for confined models (KA kernels on the CPU backend)**: [docs/plan/phase2.md](docs/plan/phase2.md)
  - T2.1 Discretization and connectivity (done)
  - T2.2 NPF, confined (done)
  - T2.3 Boundary framework + CHD, WEL, RCH, GHB (done)
  - T2.4 RIV, DRN, EVT (done)
  - T2.5 STO and IC (done)
  - T2.6 Assembly and matrix oracle (done)
  - T2.7 Linear solvers (done)
  - T2.8 Picard loop, time stepping, budgets, output
- **Phase 3: GPU execution and Milestone A**: [docs/plan/phase3.md](docs/plan/phase3.md)
  - T3.1 Device execution
  - T3.2 Builder API
  - T3.3 Stepping API
  - T3.4 Synthetic coupling-model generator
- **Phase 4: solver performance**: [docs/plan/phase4.md](docs/plan/phase4.md)
  - T4.1 Benchmark harness
  - T4.2 Chebyshev and multicolour smoothers
  - T4.3 Algebraic multigrid preconditioner
  - T4.4 `:mixed` preset and production default
  - T4.5 Kernel fusion and matrix formats
- **Phase 5: ModflowCoupling integration and Milestone B**: [docs/plan/phase5.md](docs/plan/phase5.md)
  - T5.1 Engine switch in ModflowCoupling
  - T5.2 Device-resident coupling
  - T5.3 Coupling parity on the Indus
  - T5.4 mGV + ModflowGPU on one GPU
  - T5.5a Global-size dress rehearsal
  - T5.5b Global 5′ run with real data
- **Phase 6: generic MF6 GWF (Milestone C); cards are independent once Phase 3 is done**: [docs/plan/phase6.md](docs/plan/phase6.md)
  - T6.1 Convertible cells and rewetting
  - T6.2 Newton–Raphson
  - T6.3 Time series and time-array series
  - T6.4 DISV and DISU
  - T6.5 XT3D
  - T6.6 Multi-model solutions and GWF–GWF exchanges
  - T6.7 OBS for GWF heads and flows
- **Phase 7: beyond MF6 parity and 30″ (design notes first; build after Milestone B)**: [docs/plan/phase7.md](docs/plan/phase7.md)
  - T7.1 Longitude wrap-around for global grids
  - T7.2 Spherical geometry
  - T7.3 Regional 30″ on one GPU
  - T7.4 30″ design note
  - T7.5 Multi-GPU domain decomposition

---

## 6. Dependency graph and parallelism

```
T0.1 ─┬─ T0.4
      ├─ T1.5 ───────────────────────────────┐
T0.2 ─┼─ T0.3 ───────────────────────────────┤ (parity data)
      └─ T1.1 → T1.2 → T1.3 → T1.4 → T2.1 ─┬─ T2.2 ─────────┐
                                            └─ T2.3 ─┬ T2.4 ─┤
                                                     └ T2.5 ─┴→ T2.6 → T2.7 → T2.8 → T3.1
T3.1 ─┬─ T3.2 → T3.4 → T4.1 ─┬─ T4.2 ─┐
      │                      └─ T4.3 ─┴→ T4.4, T4.5
      ├─ T3.3 → T5.1 ─┬─ T5.2 → T5.4 → T5.5a (also needs T4.3) → T5.5b (needs global data)
      │               └─ T5.3
      └─ T6.1 ‖ T6.2 ‖ T6.3 ‖ T6.4 → T6.5 ; T6.4 → T6.6 → T7.1 ; T6.7
T4.3 + T4.4 → T7.3 → T7.4 → T7.5
```

Critical path to Milestone B: T0.1 → T1.1–T1.4 → T2.1 → T2.3 → T2.4/5 → T2.6 →
T2.7 → T2.8 → T3.1 → T3.3 → T5.1 → T5.2 → T5.4, with T4.3 needed for the global
steady state.

---

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| Jacobi-preconditioned CG takes too many iterations on the global steady state (strong K contrasts, ~6 M nodes) | AMG (T4.3) early on the critical path for T5.5a/b; daily transient steps are well conditioned because of the storage term |
| KA/AMDGPU versions differ between mGV and ModflowGPU | Resolve in T0.1; keep compat ranges wide; test one shared environment in T5.4 |
| Matrix layout in newer libmf6 hides `AMAT` from BMI | Universal residual test (section 4) needs only MF6 heads |
| MF6 6.3 (JLL) vs 6.8.2 behaviour differences | Parity examples use 6.8.2; ModflowCoupling parity uses tight settings and reports per-version differences |
| Loose production closures make engines disagree | Pass/fail only with tight closures; report production differences separately |
| Float64 atomics slow/unsupported on RDNA3 | Gather-based assembly (section 3.4); no atomics in hot paths |
| `:single` is not accurate enough for badly conditioned steady states | Offsets, flux-form residual and correction solves (section 3.7); `:mixed` preset (T4.4); measured per model in `docs/precision.md` |
| Float64 sneaks into kernels and breaks Metal/oneAPI, unnoticed because those backends cannot be tested here | IR hygiene test on rocm (T3.1); `T(…)` literals by convention |
| A new package uses absolute elevations, which breaks the offset form | Rule 1 of section 3.7 requires each package card to check this; parity tests in `:single` catch it |
| GPU and mGV together exceed 16 GB at global scale | Measure in T3.4/T5.5a; `:single` / `:mixed` presets (section 3.7, T4.4); multi-GPU (T7.5) |
| Global input data not available | Synthetic `global5m` (T3.4) stands in through T5.5a; only T5.5b waits for the data |
| GPU regressions go unnoticed because CI has no GPU | GPU test summary required in every kernel/solver PR (section 2); run the full GPU suite before each milestone gate |
| ModflowCoupling patches stop applying after upstream changes | Patches are pinned to base `8dd8c01`; rebasing is an explicit PR (section 2) |

---

## 8. Decisions (project owner, 2026-11-06)

1. **Data:** Indus inputs are in `/home/bart/data/modflow/data`
   (`INDUS_DATA`). Global inputs are not available yet: synthetic models stand
   in (T5.5a), and T5.5b waits for the data.
2. **ModflowCoupling changes:** stored as a patch series in
   `patches/modflow-coupling/` in this repository (section 2). No fork, no
   upstream PRs, and no edits to the owner's checkout.
3. **Name:** `ModflowGPU.jl`.
4. **30″:** in scope as a stretch goal (Milestone D): regional 30″ on one GPU
   first (T7.3), then a design note (T7.4) before any multi-GPU work (T7.5).
5. **GPU tests:** only on this machine. CI runs CPU tests and formatting only.

Still open: where 30″ inputs would come from (synthetic models cover T7.3
until then), and which multi-GPU hardware T7.5 could run on.
