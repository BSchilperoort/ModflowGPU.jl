# ModflowGPU.jl: notes for agents

A GPU port of MODFLOW 6 groundwater flow (GWF) in Julia with KernelAbstractions. The plan,
architecture and task cards are in `PLAN.md`; the conventions below summarise its section 2.
Read the card you are working on and the sections it references before starting.

## Git workflow (local, agreed with the owner)

- The remote is `origin` (`git@github.com:BSchilperoort/ModflowGPU.jl.git`). Do not open
  GitHub PRs; cards are merged locally.
- Each task card is done on a local branch `T<id>-<short-name>` (e.g. `T0.1-package-scaffold`),
  branched from `main`. Card branches are not pushed.
- After an independent review, the branch is merged into `main` with `git merge --no-ff`, with
  the card report as the merge commit message, and `main` is pushed to `origin`. Agents
  implementing a card do not merge their own branch or push.
- The card report covers: what was ported (Fortran routines), test results with numbers
  (including the local GPU test summary for anything touching kernels, solver or assembly),
  deviations from MF6, and open issues.
- Never commit anything under `reference/` (gitignored), or any input data.

## Commands

```sh
julia --project -e 'using Pkg; Pkg.test()'                    # CPU tests (this is what CI runs)
MFGPU_TEST_BACKEND=rocm julia --project=test test/runtests.jl # GPU tests, local machine only
julia --project=@runic -m Runic --inplace .                   # format (or `runic --inplace .`)
julia --project=benchmark benchmark/hardware.jl rocm          # hardware baseline (T0.4), also `cpu`
```

CI also runs the CPU tests on KernelAbstractions 0.9 (the version mGV allows; its CPU
backend is native Julia, while KA 0.10's is POCL). The local equivalent temporarily adds a
compat entry to `test/Project.toml`. `Pkg.resolve` does not downgrade an existing
manifest, so delete the shared root `Manifest.toml` (not committed) first, and again after
reverting the compat entry so the next resolve returns to KA 0.10:

```sh
rm -f Manifest.toml
julia --project=test -e 'using Pkg; Pkg.compat("KernelAbstractions", "0.9"); Pkg.resolve()'
julia --project=test test/runtests.jl                          # (MFGPU_TEST_BACKEND=rocm works too)
git checkout test/Project.toml && rm Manifest.toml             # undo
```

`test/` and `benchmark/` are workspace projects of the package (`[workspace]` in `Project.toml`), so there is
one `Manifest.toml` at the root (not committed). It includes AMDGPU, so the GPU test command
above works without extra setup. CI (`.github/workflows/`) runs only the CPU tests and Runic.

## Conventions (PLAN.md section 2)

- **Read before you write:** read the MF6 Fortran source for the feature
  (`reference/modflow6`, fetched by `validation/fetch_reference.sh`, T0.2). Port behaviour,
  not structure, and cite the Fortran routine in a comment where a formula comes from.
- **Style (as in mGV):** Runic formatting; docstrings on public functions; `@kernel`
  functions named `*_kernel!` with a launcher function next to them; `@Const` for read-only
  kernel arguments; `get_backend(array)` to launch; `Adapt.@adapt_structure` to move structs
  to the device; `copyto!` (not slicing) to update device arrays.
- **Backends:** the core package depends only on KernelAbstractions. GPU packages are weak
  dependencies; each `ext/*Ext.jl` calls `register_backend!` in `__init__`. Select a backend
  with `select_backend(name)` (`backend_options()` lists the names). Extension hooks:
  `pin_memory!`, `supports_float64`.
- **No scalar indexing on device arrays.** GPU tests run with `allowscalar(false)`.
- **Host vs device:** input parsing, connectivity, graph colouring and AMG setup run on the
  host. Everything done once per outer iteration (assembly, boundary formulation, SpMV,
  preconditioner, convergence checks, budgets) runs in KA kernels.
- **Compat:** `KernelAbstractions = "0.9.41, 0.10"`. Code must work on both: mGV only
  allows KA 0.9 (with AMDGPU 2.8), and Milestone B loads both in one environment. See
  `docs/compat.md` (e.g. `CPU()` is `POCLBackend` in KA 0.10).

## Float32 and Float64 (PLAN.md section 3.7)

- Both precisions are first-class. Engine structs are parametric in `T` (state, assembly),
  `Ts` (Krylov), `Tp` (preconditioner) and `Ti` (indices, default `Int32`).
- `resolve_precision(backend, preset)` maps `:double`, `:single`, `:auto` to `(; T, Ts, Tp)`
  (`:mixed` arrives with T4.4). `:auto` is `:double` if `supports_float64(backend)`, else
  `:single`; `:double` on a backend without Float64 is an error.
- Every unit and parity test runs in `:single` and `:double` via the test helper
  `for_each_precision(f)` (`test/testutils.jl`). **A card is not done if it passes only in
  Float64.**
- Formulation rules, in every preset: heads stored as offsets from a per-cell reference
  `r_n` computed on the host in Float64; flux-form residuals (never `b - A*h`); correction
  solves with zero initial guess; Float64 (or compensated) accumulation for budgets, dots
  and norms; time and `delt` stay Float64 on the host.
- **Kernel hygiene:** no Float64 literals or Float64-returning calls in kernels. Write
  `T(0.5)`, `one(T)`, `oftype(x, ...)`, not `0.5`.
- Host-side parsing, specs and setup stay Float64.

## Tests and data

- `Pkg.test()` runs on the CPU backend and must pass without a GPU.
- **GPU tests run only on this machine** (AMD RX 7800 XT, ROCm in `/opt/rocm`), never in CI:
  `MFGPU_TEST_BACKEND=rocm julia --project=test test/runtests.jl`. GPU results must match
  the CPU backend within 1e-10 relative for `:double` and within 1e-4 m in heads for
  `:single`.
- Indus static inputs: `INDUS_DATA=/home/bart/data/modflow/data`. Tests that need them skip
  with a message when `INDUS_DATA` is unset. Never copy the data into this repository.
  Global inputs are not available; use the synthetic models (T3.4).
- Parity tests that need `reference/` skip with a message when it is missing.

## ModflowCoupling changes: patch files only

Changes to ModflowCoupling are stored as patches in `patches/modflow-coupling/`. Never push
to SWIFT-WUR/Modflow-coupling, and never edit `/home/bart/data/modflow/ModflowCoupling`
(the owner's checkout).

1. `validation/fetch_reference.sh` clones `https://github.com/SWIFT-WUR/Modflow-coupling`
   into `reference/Modflow-coupling` at base commit
   `8dd8c010796fd3648b855bed51195c0f2e61dd60`.
2. `scripts/apply_coupling_patches.sh` resets that clone to the base commit and applies
   `patches/modflow-coupling/*.patch` in order with `git am`.
3. Make changes there as commits, then export them with
   `git format-patch <base>..HEAD -o patches/modflow-coupling/`, one numbered patch per card
   (`0001-T5.1-engine-switch.patch`, ...), and commit the patch files here.
4. The patched clone uses ModflowGPU from this repository through a `[sources]` entry
   (`ModflowGPU = {path = "../.."}`), added by the first patch.

`patches/modflow-coupling/README.md` lists each patch, its card and the base commit. Moving
the base to a newer upstream commit means rebasing all patches and updating the README in
one card.
