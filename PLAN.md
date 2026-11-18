# ModflowGPU: plan for a GPU port of MODFLOW 6 groundwater flow

This plan is written to be executed by coding agents, one task card at a time.
The cards are in `docs/dev/plan/phase<N>.md` (section 5). Each card lists its
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
| **D: beyond MF6 parity (stretch)** | Spherical geometry and multi-GPU domain decomposition at 5′ | True cell geometry on the global grid; more than one GPU (Phase 7). The 30″ goals and the longitude wrap were dropped on 2026-11-16 (section 8, decision 11) |

**Out of scope** (document it; do not build): SFR, LAK, MAW, UZF, MVR, CSUB,
BUY, VSC, API package, GWT, GWE, PRT, and PETSc parallel runs. These are
sequential or very branchy, and the coupling does not need them. A few GWF
features are not yet placed in or out of scope (HFB, GNC, TVK/TVS, ATS and some
output options); they are listed under "Still open" in section 8 and are
rejected at load time until decided (`docs/dev/support_matrix.md`).

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
  ROCm 7.2.2 in `/opt/rocm`; Julia 1.12.6 (juliaup); AMDGPU.jl 2.11.0 with
  KernelAbstractions 0.10.5 (the versions of the Phase 4 benchmarks). KA 0.10
  is the minimum from T5.0 on (section 8, decision 7; `docs/dev/compat.md`).
- **Float32 and Float64 are both supported from the start** (section 3.7).
  Float32 is required on backends without Float64 (Metal, Intel Arc via
  oneAPI, some OpenCL devices). On every backend it halves memory and cuts
  matrix traffic by about a third. Float64 gives exact MF6 parity. RDNA3 runs
  Float64 at a small fraction of its Float32 rate, but sparse solvers are
  limited by memory bandwidth, so both are viable here. Hardware Float64
  atomics are not guaranteed on this card, so the design avoids atomics
  (section 3.4).
- Global 5′ grid: 4320 × 2160 × 2 ≈ 18.7 M cells, of which roughly a third
  are land. That gives ~5–7 M active nodes with about 6 non-zeros per row.
  Measured on the synthetic `global5m` (6.6 M nodes): the model arrays take
  1.9 GiB in `:single` and 3.0 GiB in `:double` (`docs/dev/scaling.md`), and the
  peak with the default preconditioner is about 2.5 GiB (`:single`) to 4 GiB
  (`:double`) (`benchmark/results/`). That fits in 16 GB next to mGV, whose own
  global footprint is still unknown (T5.5a).
- Global 30″ (43200 × 21600 × 2 ≈ 1.9 G cells, ~0.65 G active nodes) needs
  about 190 GiB in `:single` and 300 GiB in `:double` for the model arrays
  alone (`docs/dev/scaling.md`, scaled from `global5m`). That is 12 to 19 cards of
  16 GB. Its 3.9 G CSR entries also exceed `Int32` indices. A regional 30″
  model would fit on one card: the Indus at 30″ is 2040 × 1800 × 2 ≈ 7.3 M
  cells, about the size of global 5′. 30″ is not a goal for now (section 8,
  decision 11); the numbers stay here for reference.

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
  `backend_options`). mGV is now a package with the same layout: GPU packages
  as weak dependencies, one `ext/*Ext.jl` per backend, BMI in `src/bmi/`, on
  KA 0.10 (mGV `main` of 2026-11-13).
- **Precision:** Float32 and Float64 are both first-class from the first
  kernel on (section 3.7). Engine structs are parametric in `T` (state and
  assembly), `Ts` (Krylov), `Tp` (preconditioner) and `Ti` (indices, default
  `Int32`). Every unit and parity test runs for the `:single` and `:double`
  presets; the solver and end-to-end tests also for `:mixed` (T4.4). **A card is not done if it passes
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
  (reductions run in a different order). Every card that touches kernels, solver or assembly
  includes the local GPU test summary in its card report. Parity tests that need
  `reference/` skip with a message when it is missing.
- **Compat:** `KernelAbstractions = "0.10"` since T5.0 (section 8,
  decision 7); KA 0.9 is no longer supported or tested.
- **Data:** Indus static inputs are in `/home/bart/data/modflow/data`. Tests
  find them through `INDUS_DATA=/home/bart/data/modflow/data` and skip with a
  message when it is unset. Never copy the data into this repository.
  MODFLOW global 5′ inputs (`*_oc.nc`, same fields as the Indus files) are in
  `/home/bart/data/modflow/global_data`. mGV global inputs: parameters
  `/home/bart/data/mGV/vic_global_5min_params_lossless.nc`, forcing
  `/media/bart/Data/Work/SWIFT/WFDE5_v3.1_5arcmin.zarr`; T5.5a generates the
  global mGV routing parameters with
  [mGV-input-data](https://github.com/SWIFT-WUR/mGV-input-data). Only T5.5a/T5.5b use
  them; tests keep using the synthetic models of T3.4.
- **User docs:** a card that changes user-visible behaviour (a new package or option, a changed
  keyword, a new address) updates the matching page under `docs/src/` in the same branch; the
  reviewer checks it. `docs/make.jl` must still build without warnings.
- **Git:** one local branch per card, `T<id>-<short-name>`, branched from
  `main`. No GitHub PRs and no pushed card branches. After an independent
  review the branch is merged into `main` with `git merge --no-ff`, with the
  card report as the merge commit message, and `main` is pushed to `origin`.
  An agent does not merge its own branch or push (section 8, decision 6). Do
  not commit anything under `reference/` (it is gitignored).
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
  and update the README in one card.
- **Report:** finish each card with a card report (it becomes the merge commit
  message): what was ported (Fortran routines), test results with numbers
  (including the GPU test summary), deviations from MF6, open issues. Open
  issues that a later card must pick up are also written into that card in
  `docs/dev/plan/` ("From the T<id> review: …") when the branch is merged.

---

## 3. Architecture

### 3.1 Repository layout

```
mf-gpu/
├── Project.toml            # ModflowGPU; [weakdeps] AMDGPU, CUDA, Metal, oneAPI, OpenCL
├── src/                    # file-by-file map with test groups: docs/dev/api_map.md
│   ├── ModflowGPU.jl
│   ├── backend_setup.jl    # from mGV
│   ├── precision.jl        # presets, resolve_precision, supports_float64
│   ├── input/              # MF6 file reader (host only): tokenizer, blocks, arrays, lists, options, dfn/, simulation, supported
│   ├── spec/               # host-side model description: plain structs, no device arrays
│   ├── discretization/     # dis.jl, connectivity.jl (disv.jl, disu.jl: T6.4)
│   ├── gwf/                # model.jl, npf.jl, sto.jl, ic.jl, offsets.jl
│   ├── boundaries/         # framework.jl, chd, wel, rch, ghb, riv, drn, evt
│   ├── solver/             # csr.jl, krylov.jl, picard.jl, underrelax.jl, precond/ (chebyshev, multicolor, amg)
│   ├── budget.jl
│   ├── output/             # hds.jl, cbc.jl, grb.jl, listing.jl, oc.jl (writers and readers)
│   ├── builder.jl          # in-memory model construction (no files)
│   ├── mf6_writer.jl       # MF6 input files from a spec (small models, parity runs)
│   ├── run.jl              # ModflowGPU.run
│   ├── timing.jl           # TDIS on the host
│   ├── instrument.jl       # optional section timers for the benchmarks
│   ├── stepping.jl         # libmf6-like API: prepare_time_step, solve, …, get_value_ptr
│   └── bmi.jl              # BasicModelInterface
├── ext/                    # backend extensions
├── test/                   # Project.toml (workspace), unit/, parity/, gpu/, models/
├── validation/             # fetch_reference.sh, run_reference.jl, examples.toml, libmf6 probes
├── benchmark/              # hardware.jl, synthetic.jl, run.jl, results/*.md
├── docs/                   # Documenter manual (phase 8); dev/: plan/ (task cards), api_map, precision, scaling, compat, MF6 notes, support matrix
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
- CSR is the only matrix format. T4.5 measured the alternatives and adopted
  neither (`benchmark/results/2026-11-08-T4.5.md`): SELL-C-σ makes the SpMV
  4–8 % faster but needs a second copy of the matrix that is refreshed every
  outer iteration, and a matrix-free stencil needs the full grid, which moves
  more bytes than CSR over the reduced nodes.

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
3. **Preconditioners:** Jacobi (T2.7), Chebyshev polynomial, multicolour
   symmetric Gauss–Seidel and ILU0 (T4.2), and smoothed-aggregation AMG (T4.3),
   selected with the `preconditioner` keyword. The default is multicolour ILU0
   (`:mc_ilu0`, T4.4): it is the fastest over a steady state plus daily steps
   on every benchmark model, and AMG wins only some steady states. AMG setup
   runs on the host, and the hierarchy is copied to the device as CSR. For
   confined models the off-diagonals are constant, so the hierarchy is reused
   across outer iterations and time steps.
4. **Precision:** all of the above in `:single` and `:double` from the start
   (section 3.7). T4.4 added the `:mixed` preset and made it the production
   default on backends with Float64.

### 3.6 Public APIs

- `ModflowGPU.run(simdir; backend = CPU(), precision = :auto, preconditioner = :auto)`:
  run an MF6 simulation directory and write `.hds`/`.cbc`/`.lst`, like `mf6`.
  `backend` is a KA backend, for example `select_backend("rocm")`.
- `ModflowGPU.initialize(simdir or spec; backend, precision, preconditioner)`:
  build a `Simulation` from files or from a spec in memory.
- **Stepping API (libmf6-like):** `initialize`, `prepare_time_step`,
  `prepare_solve`, `solve`, `finalize_solve`, `finalize_time_step`,
  `get_value_ptr(sim, "GWF/X")`, `"GWF/RIV/BOUND"`, `"GWF/DRN/SIMVALS"`, … Use the
  same addresses as libmf6, so ModflowCoupling's `advance!` can switch engines
  with few changes. As implemented in T3.3:
  - `get_value_ptr(sim, address)` returns a **host mirror** in Float64 with
    absolute heads, synchronised with the device at the call boundaries (see
    the header of `src/stepping.jl`). This is what libmf6 callers expect.
  - `get_value_ptr(sim, address; device = true)` returns the engine array on
    the device in precision `T`. Head-like arrays (`X`, head columns of
    `BOUND`) are offsets, and the reference is exposed as `"GWF/HREF"`. Writes
    there bypass the conversions. T5.2 uses this form.
  - `get_value`/`set_value` copy to and from the host and convert between
    offsets and absolute Float64 values.
- `BasicModelInterface` methods for general use, always in absolute values.

### 3.7 Precision: Float32 and Float64 from the start

Every kernel, test and benchmark runs in both precisions from Phase 2 on. The
preset is a keyword (`precision = :auto`) and a config option:

| Preset | State and assembly `T` | Krylov `Ts` | Preconditioner `Tp` | Use |
| --- | --- | --- | --- | --- |
| `:double` | Float64 | Float64 | Float64 | Exact MF6 parity |
| `:single` | Float32 | Float32 | Float32 | Backends without Float64; less memory everywhere |
| `:mixed` (T4.4) | Float64 | Float64 | Float32 | Float64 accuracy with a cheaper preconditioner |
| `:auto` | `:mixed` if `supports_float64(backend)`, else `:single` | | | Production default (T4.4, with `preconditioner = :auto` = `:mc_ilu0`; docs/dev/precision.md) |

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

**Closures in Float32.** Float32 runs cannot reach `1e-9` closures. T2.8 set
floors that `:single` applies to any tighter closure (`FLOAT32_CLOSURES`,
`docs/dev/precision.md`), and the parity runs use them as the single-precision
"tight" setting:

- `OUTER_DVCLOSE ≥ max(1e-6 m, eps32 · max |h'|)`, checked every outer
  iteration. A fixed floor does not work: on toth the correction stalls at
  3e-5 m because the offsets reach 1010 m.
- `INNER_DVCLOSE ≥ 1e-6 m`.
- Absolute `INNER_RCLOSE ≥ eps32 · max_n Σ|terms|` of the first outer iteration
  of the step (× √n for the L2 options), the rounding level of the flux-form
  residual.

---

## 4. Validation strategy

| Level | Test | Pass criterion |
| --- | --- | --- |
| Parsing | All in-scope example decks parse; arrays/lists equal a flopy-free independent read where available | exact |
| Connectivity | `ia`, `ja`, `ihc`, `cl1`, `cl2`, `hwva` vs libmf6 memory (`<model>/DIS/...`, `<model>/CON/...` addresses) | exact / 1e-14 rel |
| **Assembly, universal** | Plug MF6's converged heads (from a *tight-closure* reference run) into our assembly; residual `‖b − A·h_mf6‖∞ / max_row Σ|terms|` | `:double`: < 1e-8; `:single`: at most the bound from rounding the offset heads to Float32 (`docs/dev/precision.md`) |
| Assembly, oracle | `amat` and `b'` (section 3.3) vs libmf6 `SLN_1/AMAT`, `SLN_1/RHS` after the first `solve` call (if exposed; see T0.2) | `:double` < 1e-12 rel; `:single` < 1e-6 rel |
| Heads | Our run vs MF6 run, both with tight closures | `:double`: max abs diff < 1e-6 m (or 1e-7 × head range); `:single`: < 1e-3 m |
| Budgets | Per-package in/out rates per stress period vs MF6 listing/CBC | `:double`: < 1e-6 rel, or 1e-6 × total in; `:single`: < 1e-4 × total in, and percent discrepancy < 0.01 % |
| Precision | Same model in `:single` vs `:double` | heads < 1e-3 m; package flows < 1e-4 × total in |
| Backends | rocm vs CPU backend, same model | `:double` < 1e-10 rel; `:single` heads < 1e-4 m |
| Coupling | ModflowCoupling Indus tests with `engine = "gpu"` vs `"mf6"` (tight closures) | `:double`: heads < 1e-4 m; baseflow, capillary rise < 1e-6 rel. `:single`: heads < 1e-3 m; baseflow, capillary rise < 1e-4 × domain maximum |

`:mixed` (the production default, T4.4) has Float64 state and Krylov vectors
and must meet the `:double` criteria in every row.

T2.8 confirmed the `:single` thresholds on Tier 0–2 and the synthetic models
(`docs/dev/precision.md`); T5.3 checks them on the Indus. A card may tighten them,
or loosen them with a written reason in `docs/dev/precision.md`.

**Tight-closure reference runs.** Production IMS settings (`outer_dvclose =
5 m` in ModflowCoupling) leave answers that depend on the solver path. All parity
runs therefore use a patched IMS: `OUTER_DVCLOSE 1e-9`, `INNER_DVCLOSE 1e-10`,
`INNER_RCLOSE 1e-10`, and enough iterations. Our `:single` runs use the Float32
closures of section 3.7 instead. `validation/run_reference.jl` writes
these runs to `reference/runs/<example>/tight/` and keeps the unpatched run in
`.../raw/`. Also report how far apart the engines are at production settings,
but do not use that for pass/fail.

T5.1 measured that gap on the Indus (monthly periods): at `outer_dvclose = 5`
MF6 stops up to 1.9 m from its own tight-closure heads, while the GPU engine
ends within 1e-5 m of them. The engines therefore differ visibly at production
settings, and the difference is MF6's early stop, not a difference in the
equations. T5.3 turns this into a closure recommendation for the owner.

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
| 2 | Time series, transient | `ex-gwf-fhb` (TS on CHD/WEL, STO; needs T6.3, until then the test copy `fhb_const` with constants stands in) |
| 3 | Convertible / standard formulation / rewetting | `ex-gwf-twri01` (VARIABLECV DEWATERED PERCHED, DRN, RCHA), `ex-gwf-advtidal` (convertible layer 1, VARIABLECV DEWATERED PERCHED; GHB, WEL, RIV, 3× RCH, EVT NSEG 3, TS, steady→transient; also needs T6.3), `ex-gwf-bcf2ss-p01a` (REWET, RIV, CG), `ex-gwf-bump-p01b`, `ex-gwf-nwt-p02b` (14 lay, REWET); size tests that are confined (ICELLTYPE 0) despite their tier and run on the Milestone A engine: `ex-gwf-lgrv-gc`, `ex-gwf-lgrv-gr` (25×183×147, RIV, RCH); their parity test is T6.0 |
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

The cards live in one file per phase under `docs/dev/plan/`. An agent working on a
card reads this file (sections 1–4 and 6) and its own phase file. Done cards are
marked; their merge commits on `main` hold the card reports.

- **Phase 0: setup and reference environment**: [docs/dev/plan/phase0.md](docs/dev/plan/phase0.md)
  - T0.1 Package scaffold (done)
  - T0.2 Reference toolchain (done)
  - T0.3 Example inventory and reference runs (done)
  - T0.4 GPU baseline numbers (done)
- **Phase 1: MF6 input and output (host only)**: [docs/dev/plan/phase1.md](docs/dev/plan/phase1.md)
  - T1.1 Tokenizer and block parser (done)
  - T1.2 Array input (done)
  - T1.3 List and period input (done)
  - T1.4 Simulation loader → `SimulationSpec` (done)
  - T1.5 Binary output readers/writers (done)
- **Phase 2: CPU engine for confined models (KA kernels on the CPU backend)**: [docs/dev/plan/phase2.md](docs/dev/plan/phase2.md)
  - T2.1 Discretization and connectivity (done)
  - T2.2 NPF, confined (done)
  - T2.3 Boundary framework + CHD, WEL, RCH, GHB (done)
  - T2.4 RIV, DRN, EVT (done)
  - T2.5 STO and IC (done)
  - T2.6 Assembly and matrix oracle (done)
  - T2.7 Linear solvers (done)
  - T2.8 Picard loop, time stepping, budgets, output (done)
- **Phase 3: GPU execution and Milestone A**: [docs/dev/plan/phase3.md](docs/dev/plan/phase3.md) (Milestone A reached 2026-11-07)
  - T3.1 Device execution (done)
  - T3.2 Builder API (done)
  - T3.3 Stepping API (done)
  - T3.4 Synthetic coupling-model generator (done)
- **Phase 4: solver performance**: [docs/dev/plan/phase4.md](docs/dev/plan/phase4.md) (T4.1–T4.6 complete 2026-11-08; the four missed targets are card T4.7)
  - T4.1 Benchmark harness (done)
  - T4.2 Chebyshev and multicolour smoothers (done)
  - T4.3 Algebraic multigrid preconditioner (done)
  - T4.4 `:mixed` preset and production default (done)
  - T4.5 Kernel fusion and matrix formats (done)
  - T4.6 Period-data path (done)
  - T4.7 Carried-over Phase 4 targets (merged 2026-11-16: 2 of 5 targets met; the real global model is not re-timed yet, see the status block)
- **Phase 5: ModflowCoupling integration and Milestone B**: [docs/dev/plan/phase5.md](docs/dev/plan/phase5.md)
  - T5.0 KernelAbstractions 0.10 as the minimum (done)
  - T5.1 Engine switch in ModflowCoupling (done)
  - T5.2 Device-resident coupling (done)
  - T5.3 Coupling parity on the Indus (done)
  - T5.4 mGV + ModflowGPU on one GPU (merged 2026-11-14; the closed-loop parity criterion is an open owner decision, see the card)
  - T5.5a Global-size dress rehearsal (done 2026-11-15)
  - T5.5b Global 5′ run with real data (merged 2026-11-15: the run completed on the GPU; the 200-year spin-up has not converged in heads and used 1979 forcing only, see the card)
  - T5.6 Global parity and timing against MODFLOW 6 (done 2026-11-15)
- **Phase 6: generic MF6 GWF (Milestone C); dependencies between the cards are in the phase file and section 6**: [docs/dev/plan/phase6.md](docs/dev/plan/phase6.md)
  - T6.0 Parity on the confined lgrv examples (done)
  - T6.1 Convertible cells and rewetting (done 2026-11-16)
  - T6.2 Newton–Raphson (backtracking part T6.2a done 2026-11-18)
  - T6.3 Time series and time-array series (done 2026-11-16)
  - T6.4 DISV and DISU (done 2026-11-16)
  - T6.5 XT3D
  - T6.6 Multi-model solutions and GWF–GWF exchanges
  - T6.7 OBS for GWF heads and flows
  - T6.8 Parallel test runner (tooling; after T6.2)
- **Phase 7: beyond MF6 parity (build after Milestone B)**: [docs/dev/plan/phase7.md](docs/dev/plan/phase7.md)
  - T7.1 Longitude wrap-around for global grids (dropped 2026-11-16)
  - T7.2 Spherical geometry
  - T7.3 Regional 30″ on one GPU (dropped 2026-11-16)
  - T7.4 30″ design note (dropped 2026-11-16)
  - T7.5 Multi-GPU domain decomposition (at 5′; includes its own design section)
- **Phase 8: user documentation (Documenter.jl manual on GitHub Pages)**: [docs/dev/plan/phase8.md](docs/dev/plan/phase8.md) (owner decisions 2026-11-17 at the top of the phase file)
  - T8.0 Move the developer notes to `docs/dev/` (done)
  - T8.1 Documenter scaffold and CI deployment (done)
  - T8.2 Public API and API reference (done)
  - T8.3 Installation, getting started and GPU guide (done)
  - T8.4 MF6 compatibility: packages and results (done)
  - T8.5 Stepping API and BMI guide (done)
  - T8.6 Example gallery: infrastructure and first examples (done)
  - T8.7 Remaining gallery examples (done)
  - T8.8 Performance page (done)
  - T8.9 BMI conformance with the specification (done)
  - T8.10 BMI as in MODFLOW 6 (libmf6 and ModflowInterface.jl) (done)

---

## 6. Dependency graph and parallelism

```
T0.1 ─┬─ T0.4
      ├─ T1.5 ───────────────────────────────┐
T0.2 ─┼─ T0.3 ───────────────────────────────┤ (parity data)
      └─ T1.1 → T1.2 → T1.3 → T1.4 → T2.1 ─┬─ T2.2 ─────────┐
                                            └─ T2.3 ─┬ T2.4 ─┤
                                                     └ T2.5 ─┴→ T2.6 → T2.7 → T2.8 → T3.1
T3.1 ─┬─ T3.2 → T3.4 → T4.1 ─┬─ T4.2 ─┬→ T4.4 → T4.5 → T4.7 (scheduled after T5.4)
      │                      ├─ T4.3 ─┘
      │                      └─ T4.6
      ├─ T3.2 + T3.3 → T5.1 ─┬─ T5.2 → T5.4 → T5.5a → T5.5b
      │                      └─ T5.3
      │   T5.0 (done) → T5.4 (mGV on KA 0.10: done)
      └─ Phase 6, no dependency inside the phase: T6.0, T6.1, T6.2a (backtracking),
         T6.3, T6.4, and the DIS parts of T6.5 (hanix, whirl) and T6.6 (s1, lgrv-lgr, spbc)
           T6.1 → T6.2 (Newton)
           T6.4 → T6.5 (DISV examples) ; T6.5 → T6.6 (s2–s4)
           T6.1 + T6.3 → advtidal (owned by whichever of the two merges last)
           T6.1 + T6.3 + T6.4 → T6.7
T4.3 + T4.4 → T7.5 ; T6.4 (done) → T7.2 (validation against DISU)
```

Critical path to Milestone B: T0.1 → T1.1–T1.4 → T2.1 → T2.3 → T2.4/5 → T2.6 →
T2.7 → T2.8 → T3.1 → T3.2/T3.3 → T5.1 → T5.2 → T5.4 (with T5.0) → T5.5a →
T5.5b. Done up to T5.1; mGV is on KA 0.10 and the global inputs are available.

---

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| The preconditioner takes too many iterations on the real global steady state (strong K contrasts, ~6 M nodes) | Solved on synthetic data: MC-ILU0 (T4.2, default) converges the `global5m` steady state in 2.9 s. The synthetic steady state is better conditioned than real data will be (T3.4), so T5.5b re-checks it, with AMG (T4.3) as the fallback |
| KA/AMDGPU versions differ between mGV and ModflowGPU | KA 0.10 is the minimum for both (decision 7). mGV requires KA 0.10.7 and AMDGPU 2.11 (2026-11-13); T5.0 does the same here. T5.4 tests the shared environment and re-runs the benchmark in it |
| Matrix layout in newer libmf6 hides `AMAT` from BMI | Universal residual test (section 4) needs only MF6 heads |
| MF6 6.3 (JLL) vs 6.8.2 behaviour differences | Parity examples use 6.8.2; ModflowCoupling parity uses tight settings and reports per-version differences |
| Loose production closures make engines disagree | Pass/fail only with tight closures; report production differences separately. Measured in T5.1: up to 1.9 m on the Indus, from MF6 stopping early (section 4). T5.3 proposes production closures for the GPU engine |
| Float64 atomics slow/unsupported on RDNA3 | Gather-based assembly (section 3.4); no atomics in hot paths |
| `:single` is not accurate enough for badly conditioned steady states | Offsets, flux-form residual and correction solves (section 3.7); `:mixed` preset (T4.4); measured per model in `docs/dev/precision.md` |
| Float64 sneaks into kernels and breaks Metal/oneAPI, unnoticed because those backends cannot be tested here | IR hygiene test on rocm (T3.1); `T(…)` literals by convention |
| A new package uses absolute elevations, which breaks the offset form | Rule 1 of section 3.7 requires each package card to check this; parity tests in `:single` catch it |
| GPU and mGV together exceed 16 GB at global scale | ModflowGPU alone peaks at 2.5–4 GiB on `global5m` (section 1.2); mGV's share is measured in T5.5a. Device memory of a finalized simulation is freed only by the GC until T5.4 fixes it. `:single` / `:mixed` presets (section 3.7, T4.4); multi-GPU (T7.5) |
| Global inputs differ from what the coupling expects (field names, `_nogl`/`_v2` variants, mGV forcing variables) | The data is available since 2026-11-14 (decision 1). T5.5a checks the inputs early (read and grid checks only), so T5.5b does not discover mismatches at the end |
| GPU regressions go unnoticed because CI has no GPU | GPU test summary required in the card report of every kernel/solver card (section 2); run the full GPU suite before each milestone gate |
| ModflowCoupling patches stop applying after upstream changes | Patches are pinned to base `8dd8c01`; rebasing is its own card (section 2) |
| An open issue from a card report is forgotten because it lives only in a merge commit message | Open issues are copied into the card that must pick them up when the branch is merged (section 2, Report) |

---

## 8. Decisions (project owner, 2026-11-06 unless dated otherwise)

1. **Data:** Indus inputs are in `/home/bart/data/modflow/data`
   (`INDUS_DATA`). Global inputs are not available yet: synthetic models stand
   in (T5.5a), and T5.5b waits for the data. *Update 2026-11-14:* the global
   inputs are available: MODFLOW in `/home/bart/data/modflow/global_data`, mGV
   parameters in `/home/bart/data/mGV/vic_global_5min_params_lossless.nc` and
   forcing in `/media/bart/Data/Work/SWIFT/WFDE5_v3.1_5arcmin.zarr` (section 2,
   Data).
2. **ModflowCoupling changes:** stored as a patch series in
   `patches/modflow-coupling/` in this repository (section 2). No fork, no
   upstream PRs, and no edits to the owner's checkout.
3. **Name:** `ModflowGPU.jl`.
4. **30″:** in scope as a stretch goal (Milestone D): regional 30″ on one GPU
   first (T7.3), then a design note (T7.4) before any multi-GPU work (T7.5).
   *Superseded 2026-11-16 by decision 11.*
5. **GPU tests:** only on this machine. CI runs CPU tests and formatting only.
6. **Git workflow (2026-11-07):** local card branches, an independent review,
   then a local `git merge --no-ff` into `main` with the card report as the
   merge message, and a push of `main` to `origin`. No GitHub PRs (section 2).
7. **KernelAbstractions 0.10 is the minimum (2026-11-11).** ModflowGPU drops
   KA 0.9 in T5.0. The owner updates mGV to KA 0.10; T5.4 waits for that.
   *Update 2026-11-14:* done. mGV `main` (2026-11-13) is a package with GPU
   weak dependencies and `ext/*Ext.jl` (like ModflowGPU), BMI in `src/bmi/`,
   and compat `KernelAbstractions = "0.10.7"`, `AMDGPU = "2.11"`.
8. **Milestone B includes the real global run (2026-11-11).** T5.5b stays
   blocked on the global inputs. The owner arranges them when the work reaches
   that card; nothing before it needs them. *Update 2026-11-14:* the inputs
   are available (decision 1), so T5.5b is no longer blocked.
9. **Production closures of the GPU engine (2026-11-14):**
   `outer_dvclose = inner_dvclose = 0.01` m for the GPU engine (T5.3 proposal,
   `docs/dev/precision.md`: on the Indus within 1.5e-4 m of the tight heads on
   daily steps, against 0.15 m at 5, for 20–40 % more time per period on
   rocm). The mf6 engine keeps 5, with a note in `indus.toml`. T5.4 makes it
   the per-engine default in the patched `config.jl`; T5.4 and T5.5a report
   the cost on the 1-year run and at global size, and the value is revisited
   if that cost is far from the Indus one.
10. **Domain of the global run (2026-11-15):** Greenland and the Arctic are not
    modelled. For now everything north of 60°N is masked out in both models
    (T5.5b card).
11. **Phase 7 scope (2026-11-16):** the longitude wrap-around (T7.1) is not
    needed: very little land crosses the date line. The 30″ goals (T7.3, T7.4)
    are skipped: not relevant for now. Spherical geometry (T7.2) and multi-GPU
    domain decomposition (T7.5) stay; T7.5 now targets the global 5′ model and
    carries its own design section (`docs/dev/plan/phase7.md`).

Still open:

- **Multi-GPU hardware:** which machine T7.5's scaling runs could use; this
  machine has one GPU, so development uses several MPI ranks on it.
- **Scope of Milestone C:** HFB, GNC, TVK/TVS, ATS, and the output options
  `BUDGETCSV`, `CSV_OUTER_OUTPUT`/`CSV_INNER_OUTPUT` (with the IMS convergence
  tables, open since T3.1) and `SAVE_SPECIFIC_DISCHARGE` (accepted, but
  DATA-SPDIS is not written; open since T2.8) have no card and are not listed
  as out of scope. None of the in-scope examples needs HFB.
