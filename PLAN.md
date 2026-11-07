# ModflowGPU: plan for a GPU port of MODFLOW 6 groundwater flow

This plan is written to be executed by coding agents, one task card at a time.
Each card lists its dependencies, what to read, what to deliver, and how to
prove it is done. Do not start a card before its dependencies are merged.

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
   `r_n = clamp(STRT_n, bot_n, top_n)` in Float64. The device stores heads as
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

Size: S ≈ < 1 day of agent work, M ≈ 1–3 days, L ≈ > 3 days.
`‖` marks cards that can run in parallel with their siblings.

### Phase 0: setup and reference environment

**T0.1 Package scaffold (S)**
- Depends: none.
- Read: `mGV/Project.toml`, `mGV/src/backend_setup.jl`, `mGV/ext/*.jl`,
  `mGV/test/backend_test.jl`, `mGV/.github/workflows/formatting.yml`.
- Deliver: `Project.toml` (deps: KernelAbstractions, Adapt, Atomix, BasicModelInterface,
  Dates, Printf, LinearAlgebra, SparseArrays; weakdeps + extensions for AMDGPU, CUDA, Metal,
  oneAPI, OpenCL). Also `src/ModflowGPU.jl`, `src/backend_setup.jl`, `ext/*.jl`,
  `test/Project.toml` as a workspace project (as in mGV), `test/runtests.jl` with
  the backend selection tests ported from mGV, `.gitignore` (`reference/`,
  `benchmark/out/`, `Manifest.toml` policy as in mGV), CI workflow (CPU tests +
  Runic only; no GPU jobs), `CLAUDE.md` summarising section 2 (including the
  data path, the local-only GPU tests and the patch workflow).
- **KA compat:** this machine has KA 0.10.3 (pulled in by AMDGPU 2.10); mGV
  pins `KernelAbstractions = "0.9.41"`, which excludes 0.10. Set compat to
  `"0.9.41, 0.10"` and check that both resolve, because Milestone B loads mGV
  and ModflowGPU in one environment. Record the finding in `docs/compat.md`.
- Also deliver the precision skeleton of section 3.7: `supports_float64`
  (default `true`; methods in the Metal, oneAPI and OpenCL extensions), the
  presets `:double`, `:single`, `:auto` resolving to `(T, Ts, Tp)`, and a test
  helper `for_each_precision(f)` that later cards use to run every test in both
  presets.
- Done when: `Pkg.test()` passes on CPU, `using AMDGPU, ModflowGPU;
  ModflowGPU.select_backend("rocm")` returns a `ROCBackend` on this machine, and
  preset resolution is tested with fake backends that report no Float64.

**T0.2 Reference toolchain (S) ‖**
- Depends: none.
- Deliver: `validation/fetch_reference.sh`, which downloads into `reference/`:
  - `mf6.8.2_linux.zip` (mf6 + libmf6.so),
  - `mf6.8.2_dfns.zip` (input definition files),
  - `mf6examples.zip` (release tag `current` of MODFLOW-ORG/modflow6-examples),
  - a shallow clone of `MODFLOW-ORG/modflow6` at tag `6.8.2` (Fortran source to read),
  - a clone of `SWIFT-WUR/Modflow-coupling` at the patch base commit (section 2).
  Each download goes into its own new directory. Also deliver
  `scripts/apply_coupling_patches.sh` and `patches/modflow-coupling/README.md`
  (no patches yet).
- Check, and write the result to `docs/reference_tooling.md`:
  1. ModflowInterface.jl + Modflow6_jll 6.3.0 initialises and runs one example.
  2. Can ModflowInterface load the 6.8.2 `libmf6.so` (JLL override via
     `~/.julia/artifacts/Overrides.toml` or a preference)? If yes, use 6.8.2 for
     the oracle tests.
  3. Which solution/discretization addresses libmf6 exposes: list
     `get_input_var_names`/`get_output_var_names` and look for `SLN_1/AMAT`,
     `SLN_1/RHS`, `SLN_1/IA`, `SLN_1/JA`, `SLN_1/X`, `<MODEL>/CON/IA`,
     `<MODEL>/CON/JA`, `<MODEL>/CON/IHC`, `<MODEL>/CON/CL1`, `<MODEL>/CON/HWVA`,
     `<MODEL>/NPF/CONDSAT`. If `AMAT` is not exposed in a version, the universal
     residual test in section 4 is the fallback.
- Done when: the script runs on a clean checkout, and the doc lists the working
  addresses per MF6 version.

**T0.3 Example inventory and reference runs (M)**
- Depends: T0.2.
- Deliver:
  - `validation/examples.toml`: one entry per in-scope GWF example directory, with
    tier, grid type, sizes, packages, NPF/IMS options, TS/OBS usage, steady/transient.
    Generate it with a script, then check it by hand against section 4.1, fixing
    the "convertible" flags.
  - `validation/run_reference.jl`: runs mf6 6.8.2 on each example in a copy, raw
    and tight-closure (section 4), into `reference/runs/<ex>/{raw,tight}/`.
- Done when: all Tier 0–7 examples run with mf6 in both modes (list failures with
  reasons), and the TOML is committed.

**T0.4 GPU baseline numbers (S) ‖**
- Depends: T0.1.
- Deliver: `benchmark/hardware.jl` measuring on rocm and CPU: stream triad
  bandwidth (Float32/Float64), a KA CSR SpMV on a 7-point Laplacian (1 M, 10 M rows,
  Float64 and Float32 values with Int32 indices), `dot` via `mapreduce`. Also test
  whether Float64 `Atomix.@atomic` add works and how fast it is. Results go in
  `benchmark/results/hardware.md`.
- Done when: the numbers are recorded. They set the speed-of-light targets for Phase 4.

### Phase 1: MF6 input and output (host only)

**T1.1 Tokenizer and block parser (M)**
- Depends: T0.1, T0.2.
- Read: MF6 IO guide (`mf6io.pdf` in the release, "Input Instructions" chapter),
  `src/Utilities/BlockParser.f90`, DFN files.
- Deliver: `src/input/tokenizer.jl`, `blocks.jl`: case-insensitive keywords;
  comments `#`, `!`, `//`; quoted strings; `BEGIN name [n] … END name`; repeated
  period blocks; `OPEN/CLOSE` for whole-block external files.
- Use the DFN files (`reference/dfns/*.dfn`) to drive a schema: block names,
  variable names, types, shapes, optional flags. Generate Julia definitions
  from the DFNs with a script committed under `src/input/dfn/`, rather than
  hand-writing each package's options.
- Done when: every input file of every Tier 0–7 example tokenises into blocks
  without error (test loops over `reference/examples`, skipped when missing).

**T1.2 Array input (S)**
- Depends: T1.1.
- Deliver: `src/input/arrays.jl`: `CONSTANT`, `INTERNAL [FACTOR f] [IPRN n]`,
  `OPEN/CLOSE file [FACTOR f] [(BINARY)]`, `LAYERED`, shapes from DIS/DISV/DISU
  dimensions, integer and double arrays.
- Done when: unit tests cover each form, and the binary form round-trips through
  a writer.

**T1.3 List and period input (M)**
- Depends: T1.1, T1.2.
- Deliver: `src/input/lists.jl`: cellid forms (`lay row col`, `lay cell2d`,
  `node`), `AUXILIARY`, `BOUNDNAMES`, `OPEN/CLOSE` list files, and period
  semantics: data persists until the next `PERIOD` block, an empty block clears
  it, and `READASARRAYS` (`RCHA`, `EVTA`). Time-series names in place of
  numbers are kept as symbols for T6.3.
- Done when: the period data of advtidal, twri01 and fhb match hand-checked
  values in tests.

**T1.4 Simulation loader → `SimulationSpec` (M)**
- Depends: T1.3.
- Deliver: `src/spec/*.jl` (plain host structs: `SimulationSpec`, `TdisSpec`,
  `GwfSpec`, `DisSpec`/`DisvSpec`/`DisuSpec`, `NpfSpec`, `StoSpec`, `IcSpec`,
  `BoundarySpec{Kind}`, `ImsSpec`, `OcSpec`, `ExchangeSpec`) and
  `src/input/simulation.jl` reading `mfsim.nam`, TDIS, model name files, and
  packages. Unsupported packages raise an error naming the package. Options that
  are parsed but not implemented raise an error, not a silent ignore.
- Done when: all Tier 0–2 examples load. Higher tiers load or fail with
  "unsupported: <thing>".

**T1.5 Binary output readers/writers (S) ‖**
- Depends: T0.1.
- Read: MF6 IO guide, "Description of Binary Output Files"; flopy's
  `binaryfile` reader for reference.
- Deliver: `src/output/hds.jl`, `cbc.jl`: read and write head (DIS, DISV, DISU
  layouts) and budget files (including `FLOW-JA-FACE`, imeth 1/6 records with
  aux), and a listing-file budget table parser (port `listing_rates` from
  ModflowCoupling's tests).
- Done when: reading the reference runs of T0.3 works for all tiers, and our
  writer → reader round-trip is exact.

### Phase 2: CPU engine for confined models (KA kernels on the CPU backend)

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
  then CHD, WEL (without `AUTO_FLOW_REDUCE` for now: reject it), RCH (list and
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
- Deliver: `src/solver/csr.jl` assembly kernel (off-diagonals from `condsat`,
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

### Phase 3: GPU execution and Milestone A

**T3.1 Device execution (M)**
- Depends: T2.8.
- Deliver: `Adapt` rules for all engine structs, a `backend` keyword through the
  build path, removal of all host-only code paths from the per-iteration loop,
  and `test/gpu/` running the full parity suite with `MFGPU_TEST_BACKEND`.
  Implement reductions (`dot`, `norm`, max-abs, argmax for the IMS convergence
  table) with a backend-agnostic approach: `mapreduce` on device arrays first,
  and AcceleratedKernels.jl or a hand-written KA two-pass reduction if it is
  slow (measure, see T0.4).
- Add the kernel-hygiene test (section 3.7, rule 5): compile every kernel for
  `:single` on rocm, capture the device LLVM IR (`AMDGPU.@device_code_llvm` or
  GPUCompiler reflection), and fail on any `double`.
- Done when: every Phase 2 parity test passes on rocm in both presets, rocm vs
  CPU agree within the section 4 backend tolerances, and the hygiene test passes.

**T3.2 Builder API (M) ‖ with T3.3**
- Depends: T3.1.
- Deliver: `src/builder.jl`: build a `SimulationSpec` from Julia arrays (grid
  arrays as `(ncol, nrow, nlay)` like ModflowCoupling; boundary records as
  cell lists + value matrices), plus an MF6 writer for small specs (for parity
  runs). Port ModflowCoupling's `mf6_writer.jl`, or generalise it if it fits.
- Done when: building the coupling model in memory and from its MF6 files gives
  bitwise-identical specs.

**T3.3 Stepping API (M) ‖ with T3.2**
- Depends: T3.1.
- Deliver: `src/stepping.jl` with the libmf6-like functions and addresses of
  section 3.6 (at least `X`, `NODEUSER`, `NODELIST`, `BOUND`, `SIMVALS`,
  `MXITER`), and `src/bmi.jl` (BasicModelInterface: grids, time,
  `get_value(_ptr)`, `set_value`, `update`).
- Done when: a test drives the synthetic coupling model through the same call
  sequence as `ModflowCoupling.advance!` and matches `ModflowGPU.run` exactly.

**T3.4 Synthetic coupling-model generator (S) ‖**
- Depends: T3.2.
- Deliver: `benchmark/synthetic.jl`: a coupling-like model of given `ncol × nrow`
  (2 layers): land mask with ocean CHD rim, log-normal K with 4–6 orders of
  magnitude contrast, a DEM, a river network (RIV on high-accumulation cells),
  DRN on land, EVT, RCH, uniform DELR/DELC. Write it to MF6 files for small sizes
  and build it in memory for all sizes. Presets: `tiny` (50×40), `indus`
  (204×180, same as the Indus grid), `continental` (1000×800), `global5m`
  (4320×2160, ~35 % land, the stand-in for the missing global data),
  `regional30s` (2040×1800, Indus extent at 30″), and `global30s` (43200×21600;
  memory estimate only, never allocated on one GPU).
- Done when: `tiny` and `indus` pass parity vs mf6 6.8.2 (steady and 12 daily
  transient periods with changing RCH/RIV stage) in both presets, `global5m`
  and `regional30s` build and fit in GPU memory (report bytes per preset), and
  `global30s` reports its estimated memory per preset.

**Milestone A gate:** T3.1–T3.4 merged; parity on Tier 0–2 + synthetic models
on CPU and rocm; `docs/support_matrix.md` lists supported packages/options.

### Phase 4: solver performance

**T4.1 Benchmark harness (S)**
- Depends: T3.4.
- Deliver: `benchmark/run.jl` running `indus`, `continental`, `global5m`,
  steady state and 30 daily steps, on rocm and CPU, plus mf6 6.8.2 (serial) up to
  the size where it finishes in reasonable time. Record wall time per time step,
  outer and inner iterations, achieved GB/s in SpMV, peak device memory, and time
  split (assembly / SpMV / preconditioner / reductions / host transfers) with
  TimerOutputs, as mGV does. Write results to `benchmark/results/<date>.md`.
- Done when: the baseline table exists. Phase 4 targets are then set from it,
  for example a daily step of `global5m` in < 1 s and steady state in minutes.

**T4.2 Chebyshev and multicolour smoothers (M) ‖**
- Depends: T4.1.
- Deliver: `src/solver/precond/chebyshev.jl` (eigenvalue bounds from a few
  Lanczos/CG steps), `multicolor.jl` (red–black colouring for DIS by `(i+j+k)`
  parity; greedy colouring for general graphs on the host), and symmetric
  Gauss–Seidel / ILU0 per colour.
- Done when: inner iteration counts and wall time are reported vs Jacobi on all
  benchmark sizes. Keep whichever is fastest per case.

**T4.3 Algebraic multigrid preconditioner (L) ‖**
- Depends: T4.1.
- Deliver: `src/solver/precond/amg.jl`: smoothed-aggregation hierarchy built on
  the host (AlgebraicMultigrid.jl, or our own if its API does not fit), strength
  threshold tuned for strong K contrasts and the strong vertical coupling of
  2-layer models (try aggregating each vertical column first). V-cycle on the
  device with CSR SpMV, a Chebyshev or Jacobi smoother, and a direct or many-sweep
  coarsest level. Reuse the hierarchy across outer iterations and time steps,
  and rebuild only when the off-diagonals change (convertible/Newton) or
  convergence degrades past a threshold.
- Done when: `global5m` steady state converges to tight closure, and the
  iteration count is roughly independent of grid size (compare `indus` vs
  `continental` vs `global5m`).

**T4.4 `:mixed` preset and production default (M)**
- Depends: T4.3.
- Deliver: the `:mixed` preset (Float64 state and Krylov, Float32 AMG
  hierarchy and smoother data), and a benchmark of `:single`, `:mixed` and
  `:double` on all benchmark sizes. Measure wall time, memory, iterations, and
  the difference from `:double`, especially for the `global5m` steady state,
  where conditioning is worst.
- Done when: `:mixed` meets the `:double` parity tolerances, and
  `docs/precision.md` recommends a production default per backend class (with
  and without Float64) based on the numbers. `:auto` then follows that
  recommendation.

**T4.5 Kernel fusion and matrix formats (M)**
- Depends: T4.2 or T4.3.
- Deliver: fused PCG/BiCGSTAB update kernels (axpy + dot in one pass),
  SpMV variants (CSR scalar vs vector, SELL-C-σ, and a matrix-free 7-point
  stencil for DIS), selected per grid type by benchmark.
- Done when: SpMV reaches ≥ 70 % of the T0.4 triad bandwidth on `global5m`,
  or the reason why not is documented.

### Phase 5: ModflowCoupling integration and Milestone B

All ModflowCoupling changes follow the patch workflow of section 2: work in
`reference/Modflow-coupling` (base `8dd8c01`), and deliver numbered patches in
`patches/modflow-coupling/`. Each card's "done" check starts from a fresh
`scripts/apply_coupling_patches.sh` run, so the patch series always applies
cleanly. Run the tests with `INDUS_DATA=/home/bart/data/modflow/data`.

**T5.1 Engine switch in ModflowCoupling (M)**
- Depends: T3.3.
- Deliver (as `0001-…patch` and up): a `[sources]` entry pointing at this
  repository, plus `[solver] engine = "mf6" | "gpu"` and `backend = "auto"` in the TOML
  config. With `gpu`, build the ModflowGPU model from `StaticParameters` through
  the builder (no `mf6_writer`), and drop the 6.3-specific RCH area scaling in
  `calibrate_bmi!` for the GPU engine (rates per unit area). Keep `advance!`
  calling the stepping API, plus `[solver] precision = "auto"`. River stages
  are head-like: write them through `set_value` (absolute, converted to
  offsets), or subtract `"GWF/HREF"` in Float64 before writing into `BOUND`.
- Done when: ModflowCoupling's test suite passes with both engines
  (`INDUS_DATA=/home/bart/data/modflow/data`).

**T5.2 Device-resident coupling (M)**
- Depends: T5.1.
- Deliver: rewrite `apply_stresses!` and `collect_outputs!` in ModflowCoupling
  as KA kernels on device arrays (record→cell index maps on the device), so the
  per-period work stays on the GPU. Compute the river stage in offset form
  (`bottom_bankfull` as an offset plus the Manning depth), so it stays accurate
  in Float32. Output `groundwater_depth` as `top' − h'` from offsets. BMI `get_value_ptr` returns device arrays
  when the engine is on a GPU.
- Done when: results match T5.1, and a daily step has no host↔device transfer
  beyond the BMI calls that ask for host arrays (check with the profiler or
  timers).

**T5.3 Coupling parity on the Indus (S)**
- Depends: T5.1.
- Deliver: `test/coupling_parity.jl` (in the patched ModflowCoupling clone, as a patch) running 1979-01 to 1979-03
  monthly and 31 daily periods with tight solver settings on both engines, with
  the GPU engine in `:single` and `:double`.
  Report max differences in heads, `baseflow_to_routing`, and `capillary_rise`,
  and also the differences at production settings (`outer_dvclose = 5`).
- Done when: the section 4 coupling criteria pass.

**T5.4 mGV + ModflowGPU on one GPU (M)**
- Depends: T5.2.
- Read: `Modflow-coupling/examples/mgv_coupling/*.jl`, mGV `src/bmi/`.
- Deliver: the daily coupling example running both models on the rocm backend,
  exchanging device arrays (in `:single` the exchanged fields are Float32 on
  both sides; in `:double`, convert Float32 mGV ↔ Float64 MODFLOW in a kernel; check that `align_to_grid` is the identity when grids match, and keep
  the host path otherwise). Resolve the KA/AMDGPU compat between mGV and
  ModflowGPU (see T0.1).
- Done when: the Indus 1-year coupled run gives the same results as the
  mf6-engine run within T5.3 tolerances, and timings for both engines are
  reported.

**T5.5a Global-size dress rehearsal (M)**
- Depends: T4.3, T5.4.
- Global inputs are not available yet, so run the full pipeline at global size
  on synthetic data: the `global5m` model of T3.4 driven through
  ModflowCoupling's GPU engine (build from arrays, not NetCDF), steady-state
  spin-up, then 1 year daily. Add mGV's global memory footprint, from its
  `configs/global_config.toml` if mGV can run on synthetic forcing, otherwise
  as an estimate from its array sizes.
- Done when: a report gives device memory of both models together and wall time
  per daily step and for the spin-up, per precision preset, and states whether
  both fit in 16 GB. If
  they do not, it proposes a fix (`:single` or `:mixed` preset, mGV memory reduction,
  Phase 7).

**T5.5b Global 5′ run with real data (L), blocked until global inputs exist**
- Depends: T5.5a and the global input data (not available on 2026-11-06).
- Deliver: a global ModflowCoupling config, a steady-state spin-up
  (`period = "steady"`) on the GPU, then 1 year daily coupled with mGV. Compare
  with the existing global steady-state head (`mf_ss_gwl_5minDRN.nc`; its Indus
  cut is in `/home/bart/data/modflow/data`) as a sanity check.
- Done when: the run completes on the 16 GB GPU.

### Phase 6: generic MF6 GWF (Milestone C); cards are independent once Phase 3 is done

**T6.1 Convertible cells and rewetting (L) ‖**
- Read: `gwf-npf.f90` (`sat`, `THICKSTRT`, `VARIABLECV [DEWATERED]`,
  `PERCHED`, `wd` rewetting: `WETFCT`, `IWETIT`, `IHDWET`), `gwf-sto.f90` (Sy,
  convertible storage).
- Deliver: saturation-dependent conductance recomputed per outer iteration in a
  kernel (`condsat` stays constant; a per-connection saturation factor is
  applied), convertible storage, and rewetting as a per-cell kernel with a
  neighbour check, run between outer iterations as MF6 does.
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

### Phase 7: beyond MF6 parity and 30″ (design notes first; build after Milestone B)

**T7.1 Longitude wrap-around for global grids (M).** Connect column 1 and
column `ncol` with a periodic connection, using the T6.6 self-exchange.
Validate against mf6 with an equivalent `GWF6-GWF6` self-exchange. Also remove
ModflowCoupling's limitation "a domain that crosses the edge of the input maps
cannot be cut".

**T7.2 Spherical geometry (M).** Row-dependent cell width and area (true
geometry instead of RCH/EVT area-factor scaling). This deviates from MF6 DIS;
validate against an mf6 DISU model with the same geometry.

**T7.3 Regional 30″ on one GPU (M).** Depends: T4.3, T4.4. Run the
`regional30s` synthetic model (and, if the owner provides 30″ Indus inputs, the
real Indus at 30″) as steady state and daily transient, in `:single` and
`:mixed`. Report memory, AMG
iteration counts versus the 5′ model (grid-independence check), and time per
daily step. This is the first Milestone D result and needs no extra hardware.

**T7.4 30″ design note (S).** Depends: T7.3. Before building multi-GPU, write
`docs/design_30s.md`. It covers: memory per active node measured in T7.3, scaled
to global 30″; minimum GPU count and memory per GPU; partitioning (latitude
bands vs land-balanced blocks, given that ~70 % of the grid is ocean);
communication volume per iteration; and whether the coupling with mGV can be
partitioned the same way. End with a go/no-go recommendation for T7.5 and the
hardware it needs.

**T7.5 Multi-GPU domain decomposition (L).** Depends: T7.4 (go). MPI.jl,
partitions from T7.4, halo exchange of heads, per-partition AMG inside a
block-Jacobi/additive Schwarz preconditioner, and global reductions. Develop and
test on this machine with several MPI ranks sharing the one GPU (and on the CPU
backend). The decomposed run must match the single-domain run on `regional30s`
within 1e-8 m. Real multi-GPU scaling runs need hardware that this machine does
not have.

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
