# Phase 0: setup and reference environment

Task cards for this phase of [PLAN.md](../../../PLAN.md). Read PLAN.md sections 1–4 and 6 for the goals, conventions, architecture and validation rules these cards refer to.

Size: S ≈ < 1 day of agent work, M ≈ 1–3 days, L ≈ > 3 days.
`‖` marks cards that can run in parallel with their siblings.

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
  and ModflowGPU in one environment. Record the finding in `docs/dev/compat.md`.
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
- Check, and write the result to `docs/dev/reference_tooling.md`:
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
