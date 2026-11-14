# Phase 5: ModflowCoupling integration and Milestone B

Task cards for this phase of [PLAN.md](../../PLAN.md). Read PLAN.md sections 1–4 and 6 for the goals, conventions, architecture and validation rules these cards refer to.

Size: S ≈ < 1 day of agent work, M ≈ 1–3 days, L ≈ > 3 days.
`‖` marks cards that can run in parallel with their siblings.

All ModflowCoupling changes follow the patch workflow of section 2: work in
`reference/Modflow-coupling` (base `8dd8c01`), and deliver numbered patches in
`patches/modflow-coupling/`. Each card's "done" check starts from a fresh
`scripts/apply_coupling_patches.sh` run, so the patch series always applies
cleanly. Run the tests with `INDUS_DATA=/home/bart/data/modflow/data`.

**T5.0 KernelAbstractions 0.10 as the minimum (S) ‖ (done 2026-11-14)**
- Depends: nothing. Owner decision of 2026-11-11 (PLAN.md section 8, decision 7): mGV
  moves to KA 0.10, so ModflowGPU no longer has to run on KA 0.9 with AMDGPU 2.8. Done
  on 2026-11-13: mGV `main` is a package (GPU weak dependencies, `ext/*Ext.jl`, BMI in
  `src/bmi/`) with compat `KernelAbstractions = "0.10.7"` and `AMDGPU = "2.11"`. The
  Phase 4 numbers were measured on KA 0.10.5 with AMDGPU 2.11.0; on AMDGPU 2.8 the
  reductions pay an extra host copy (about 32 µs each, T3.1).
- Deliver:
  1. `KernelAbstractions = "0.10"` in `Project.toml` and `benchmark/Project.toml`. Raise
     the AMDGPU compat to the lowest version that resolves with it and passes the rocm
     suite (2.11.0 is the tested one; 2.9–2.10 have KA only as a weak dependency). Check
     that the CUDA, Metal, oneAPI and OpenCL compat entries still resolve and that the
     five extensions load.
  2. Remove what only KA 0.9 or AMDGPU 2.8 needed: the `ka` matrix axis in
     `.github/workflows/test.yml`, the `applicable(unsafe_wrap, …)` fallbacks in
     `ext/AMDGPUExt.jl` (`host_mapped_zeros`, `pin_memory!`; keep the copy path of the
     reductions, which backends without the hook still use), the written-out reduction
     tree in `src/solver/krylov.jl` if KA 0.10 allows the simpler form, and the `"CPU"`
     name in `test/unit/backend_test.jl`.
  3. Update `docs/compat.md` (replace its "mGV in the same environment" section, which
     still describes mGV on KA 0.9, with the versions ModflowGPU + mGV `main` resolve to),
     the Commands and Compat parts of `CLAUDE.md`, the comment
     in `src/precision.jl` and the header of `test/unit/kernel_test.jl`.
- Done when: the CPU and rocm suites pass with bitwise unchanged parity results, CI runs
  on one KA version, and `benchmark/run.jl` (indus, `global5m`) is within noise of
  `benchmark/results/2026-11-08-T4.5.md`.

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
  in Float32. Output `groundwater_depth` as `top' − h'` from offsets. Use
  `get_value_ptr(sim, address; device = true)` (engine arrays in `T`, head-like values
  as offsets from `HREF`); the default `get_value_ptr` is a host mirror in absolute
  Float64 values (PLAN.md section 3.6) and stays for the mf6-like host path.
- From the T5.1 report:
  - The patched clone has no AMDGPU dependency, so its suite ran on the CPU backend
    only. Add AMDGPU to its test environment and a way to select the backend (for
    example `MODFLOWCOUPLING_BACKEND=rocm`), and run the engine tests on rocm. If T5.3
    is done first, it adds this.
  - In ModflowGPU, the "backtracking is not applied" and "ILUT is replaced by
    `:mc_ilu0`" warnings appear on every GPU build. Log them once (`maxlog = 1`), and
    log the ILUT one at info level when ILUT comes from COMPLEXITY and not from an
    explicit keyword.
- Done when: results match T5.1 on CPU and rocm, and a daily step has no host↔device
  transfer beyond the BMI calls that ask for host arrays (check with the profiler or
  timers).

**T5.3 Coupling parity on the Indus (S)**
- Depends: T5.1.
- Deliver: `test/coupling_parity.jl` (in the patched ModflowCoupling clone, as a patch) running 1979-01 to 1979-03
  monthly and 31 daily periods with tight solver settings on both engines, with
  the GPU engine in `:single`, `:mixed` and `:double`, on CPU and rocm (T5.1 ran
  neither `:single` nor rocm; see the AMDGPU note in T5.2).
  Report max differences in heads, `baseflow_to_routing`, and `capillary_rise`,
  and also the differences at production settings (`outer_dvclose = 5`).
- Closure recommendation (from the T5.1 report): at `outer_dvclose = 5` MF6 stopped up
  to 1.9 m from its tight-closure heads on monthly periods (3e-3 m at 0.5, 4e-7 m at
  0.01), while the GPU engine was within 8.2e-6 m. For both engines and for
  `outer_dvclose` = 5, 0.5 and 0.01, report the distance to the tight heads and the wall
  time per period (monthly and daily). Propose production closures for the GPU engine
  in `docs/precision.md`; the owner decides (PLAN.md section 8, still open).
- Done when: the section 4 coupling criteria pass (`:mixed` against the `:double`
  criteria), and the closure table and proposal are in `docs/precision.md`.

**T5.4 mGV + ModflowGPU on one GPU (M)**
- Depends: T5.2 and T5.0. mGV is on KA 0.10 since 2026-11-13 (PLAN.md section 8,
  decision 7); use mGV `main`. Do not patch mGV; if something in mGV blocks the card,
  stop and report.
- Read: `Modflow-coupling/examples/mgv_coupling/*.jl`, mGV `src/bmi/` and
  `src/backend_setup.jl` (both packages select their backend the same way).
- Deliver: the daily coupling example running both models on the rocm backend,
  exchanging device arrays (in `:single` the exchanged fields are Float32 on
  both sides; in `:mixed` and `:double`, convert Float32 mGV ↔ Float64 MODFLOW in a
  kernel; check that `align_to_grid` is the identity when grids match, and keep
  the host path otherwise). Both packages in one environment on KA 0.10; update
  `docs/compat.md` with the resolved versions.
- Re-run `benchmark/run.jl` (indus, `global5m`) in the shared environment and compare
  with `benchmark/results/2026-11-08-T4.5.md`, so the Phase 4 numbers are known to hold
  where Milestone B runs. Use the timings to decide whether T4.7 is needed.
- From the T5.1 report:
  - `finalize` only drops the reference, and device memory is freed when the GC runs.
    With two models on one 16 GB card that is not enough: free the device arrays in
    `finalize` (a backend hook next to `pin_memory!`), and test that the pool usage
    returns to its level before `initialize`.
  - ModflowGPU is a hard dependency of the patched ModflowCoupling. Decide here whether
    it becomes a weak dependency (extension), so the mf6 engine installs without it.
- Done when: the Indus 1-year coupled run with tight closures gives the same results as
  the mf6-engine run within T5.3 tolerances, and timings for both engines are reported
  at tight and at production closures.

**T5.5a Global-size dress rehearsal (M)**
- Depends: T4.3, T5.4.
- Run the full pipeline at global size on synthetic data: the `global5m` model of
  T3.4 driven through ModflowCoupling's GPU engine (build from arrays, not NetCDF),
  steady-state spin-up, then 1 year daily. Measure mGV's global memory footprint
  with its real global inputs (available since 2026-11-14, PLAN.md section 2, Data):
  parameters `/home/bart/data/mGV/vic_global_5min_params_lossless.nc`, forcing
  `/media/bart/Data/Work/SWIFT/WFDE5_v3.1_5arcmin.zarr`, starting from mGV's
  `configs/global_config_zarr.toml`.
- Check the real inputs early, so T5.5b does not find mismatches at the end. Read the
  MODFLOW files in `/home/bart/data/modflow/global_data` (`*_oc.nc`) and check that the
  grids of both models line up. Pick between the file variants (`_nogl`, `_v2`,
  `_v3`, `_v2clamped`; the `*.qa.json` files next to them describe them) and record the
  choice for T5.5b. The WFDE5 zarr holds one store per forcing variable (`lwdown`,
  `prec`, `psurf`, `swdown`, `tair`, `vp`, `wind`); map them onto the per-variable
  file entries of mGV's global config. There is no global routing parameter file yet:
  generate it with `routing/HydroSHEDS/make-routing.py` from
  [mGV-input-data](https://github.com/SWIFT-WUR/mGV-input-data) (needs `uv` and the
  HydroSHEDS 15″ DEM `hyd_glo_dem_15s.tif`, see its README) with a global `--bbox`
  and 5′ cells. Clone it, the DEM and the output under `/home/bart/data/mGV/`, not in
  this repository. The routing grid must be the grid of the mGV parameter file
  (`examples/mgv_coupling/coupling.jl` errors when they differ); check that, and check
  how the DEM's latitude coverage compares with the parameter grid. Record the
  command, the mGV-input-data commit and the output path in the card report.
- Check that the device memory of the spin-up simulation is released before the daily
  run starts (the `finalize` fix of T5.4), and report the peak over the whole sequence.
- Done when: a report gives device memory of both models together and wall time
  per daily step and for the spin-up, per precision preset, and states whether
  both fit in 16 GB. If
  they do not, it proposes a fix (`:single` or `:mixed` preset, mGV memory reduction,
  Phase 7).

**T5.5b Global 5′ run with real data (L)**
- Depends: T5.5a. The global inputs are available since 2026-11-14 (PLAN.md section 8,
  decisions 1 and 8): MODFLOW in `/home/bart/data/modflow/global_data`, mGV as in
  T5.5a, with the file choices T5.5a recorded.
- Deliver: a global ModflowCoupling config, a steady-state spin-up
  (`period = "steady"`) on the GPU, then 1 year daily coupled with mGV. There is no
  global coupled steady-state head yet: it comes from coupling mGV to MODFLOW
  globally, which only became possible once mGV fit in this card's VRAM, so this
  card produces the first one. As a sanity check, compare its Indus cut with the
  Indus steady states in `/home/bart/data/modflow/data` (`mf_ss_gwl_5minDRN.nc`,
  `steady_head_mgv_coupled.nc`; `NOTES.md` there describes the first), and
  report the differences rather than requiring a match (the domains and boundaries
  differ).
- From the T3.4 report: the synthetic steady state is well conditioned (dense
  coastline, strong drains), so the `global5m` times are optimistic for real data.
  Report inner iterations per outer iteration for the real steady state with the
  default `:mc_ilu0`, and switch to `:amg` if they are far above the synthetic ones.
- Done when: the run completes on the 16 GB GPU.

**Milestone B gate:** T5.0–T5.5b merged; the section 4 coupling criteria pass on the
Indus in `:single`, `:mixed` and `:double` on CPU and rocm; mGV and ModflowGPU run the
global 5′ steady state and one year of daily steps together on the 16 GB GPU.
