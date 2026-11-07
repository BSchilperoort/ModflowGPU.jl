# Phase 3: GPU execution and Milestone A

Task cards for this phase of [PLAN.md](../../PLAN.md). Read PLAN.md sections 1–4 and 6 for the goals, conventions, architecture and validation rules these cards refer to.

Size: S ≈ < 1 day of agent work, M ≈ 1–3 days, L ≈ > 3 days.
`‖` marks cards that can run in parallel with their siblings.

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
