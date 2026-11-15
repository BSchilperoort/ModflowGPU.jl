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

**T5.2 Device-resident coupling (M) (done 2026-11-14)**
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
    example `MODFLOWCOUPLING_BACKEND=rocm`), and run the engine tests on rocm. Done in
    T5.3 (patch 0002: `MODFLOWCOUPLING_BACKEND=auto|cpu|rocm`, AMDGPU as a test
    dependency, test groups); reuse it.
  - In ModflowGPU, the "backtracking is not applied" and "ILUT is replaced by
    `:mc_ilu0`" warnings appear on every GPU build. Log them once (`maxlog = 1`), and
    log the ILUT one at info level when ILUT comes from COMPLEXITY and not from an
    explicit keyword.
- From the T5.3 report: the review ran only the parity group of the clone on rocm.
  Run the clone's full suite on rocm here.
- Done when: results match T5.1 on CPU and rocm, and a daily step has no host↔device
  transfer beyond the BMI calls that ask for host arrays (check with the profiler or
  timers).

**T5.3 Coupling parity on the Indus (S) (done 2026-11-14)**
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

**T5.4 mGV + ModflowGPU on one GPU (M) (merged 2026-11-14, one check open)**
- Open owner decision: the Done-when check below fails for baseflow in the closed loop
  (`:double`/`:mixed` 2.2e-6 against 1e-6 on 46 of 365 days, none before day 220; heads
  and capillary rise pass). The review traced it to drift of two coupled runs, seeded by
  one-ulp Float32 differences in the baseflow handed to mGV: GPU against GPU with one ulp
  added on day 3 gives 2.6e-6. Open loop (the GPU engine driven by the inputs recorded
  from the mf6 run, through the device exchange) passes with 1.8e-11. Proposal: for
  multi-month runs, use the open-loop replay plus the closed loop on the first 31 days
  and on the water-balance totals. The tolerance was not changed.
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
- From the T5.2 report:
  - The device path takes its inputs from host BMI calls only. Add setters that take
    device arrays (the mGV exchange) and the Float32 ↔ Float64 kernel this card asks for.
  - A host array from `BMI.get_value_ptr` is refreshed only on the next property read,
    so a caller that keeps the pointer across `update` sees old values, and inputs
    written straight into `m.recharge` (not through `set_value`) are ignored on the
    device path. Make this safe for the mGV coupling, or make the misuse an error.
  - The contract for writing device BOUND (between `prepare_time_step` and
    `prepare_solve`) is documented but not enforced, and device writes are unchecked.
  - `rec_map` keeps only the last record of a cell, where the host path adds up DRN
    records on one cell. Sum them, or error when a cell has more than one record.
  - The device path is not tested with `foc = false` (no EVT), and WEL only by its sum.
  - `maxlog = 1` is per call site, so the message for a different preconditioner of a
    later model in the same process is suppressed.
  - rocprofv3 shows three `hipMemcpy` calls per daily step: two BMI uploads and one not
    identified (probably the device-to-device `hold ← h` copy). Identify it. Monthly
    periods also copy `rho` from host to device when the step length changes.
  - In `:single` the river stage rounds to 1.2e-4 m where the cell reference is 3.2 km
    from the stage (the Float32 limit of the offset form with this reference).
- From the T5.3 report:
  - The owner adopted the closure proposal on 2026-11-14 (PLAN.md section 8,
    decision 9). Make `outer_dvclose = inner_dvclose = 0.01` the default of the GPU
    engine in the patched `config.jl` (the mf6 engine keeps 5, with a note in
    `indus.toml` that it stops up to 1.9 m from the converged heads on monthly
    periods), and update the proposal section of `docs/precision.md` to say it is
    adopted. Report its cost against 5 on the 1-year run; T5.5a repeats this at global
    size.
  - At closures 5 and 0.5, `:single` stops 3e-3 m from its tight heads on monthly
    periods, where `:mixed` stops 8.2e-6 m away. The Float32 INNER_RCLOSE floor is the
    likely cause; check it if `:single` is used for the coupled run.
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
- Report the wall time per daily step at `outer_dvclose = inner_dvclose` = 5, 0.01 (the
  GPU engine default, PLAN.md section 8, decision 9) and 1e-3 on `global5m`, so the
  owner can revisit the value at global size.
- From the T6.0 report: `ex-gwf-lgrv-gr` in `:single` does not converge with the
  default `:mc_ilu0` (500 outer iterations) and passes only with `:amg`. Under STRICT
  the Float32 INNER_RCLOSE floor (`eps32·max Σ|terms|`) is below the residual of the
  rounded exact heads, and a floor that converges leaves the heads 6.1e-3 m off,
  because Float32 CG with MC-ILU0 is too slow there. For the steady-state spin-up,
  decide how `:single` picks its preconditioner (for example `:amg` automatically for
  badly conditioned steady states) and whether the rclose floor should account for
  head rounding. `:mixed` with `:mc_ilu0` is fine.
- From the T5.4 report:
  - `cell_maps` in `examples/mgv_coupling/coupling.jl` runs `argmin(abs.(v .- x))` over
    all longitudes and latitudes for every mGV cell: about 1e10 operations and tens of
    GB allocated at global size. Replace it with `searchsorted` or index arithmetic
    before the first global run.
  - `gather_convert_kernel!` reads `src[map[i]]` under `@inbounds`. Check the map
    against `length(src)` once on the host where the map is built.
  - `mGV.finalize!` does not free mGV's device arrays (64 MiB on the Indus). mGV is not
    patched here: account for it in the memory report, and tell the owner if it decides
    whether both models fit.
  - A second DRN, RIV or EVT record on one cell is an error on the device path. Check
    the global inputs for duplicates (T5.5b uses them).
  - T4.7: on the Indus mGV takes 1.70 ms per day and MODFLOW 1.20 ms, so T4.7 is not
    needed there. Report the same ratio at global size (MODFLOW on `global5m` takes
    0.265 s per day) and say whether T4.7 is needed.
  - Coupled parity over more than a month is compared open loop until the owner decides
    the criterion (see the T5.4 card).
  - Smaller: `finalize!` in ModflowCoupling frees `d.sim` twice (harmless); the device
    BOUND check does not cover all of `riv_ck`; in `:single` the river stage rounds by up
    to 1.4e-4 m on perched rivers; CUDA's `free_memory!` is untested.
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
  T5.5a, with the file choices T5.5a recorded: `docs/global_inputs.md` (file choices,
  what does not line up, package sizes) and the draft config
  `examples/global/global.toml` of ModflowCoupling patch 0005. Before the run, decide the
  CHD heads (the chosen steady-head file has none on the 173,727 CHD ring cells; sea
  level 0 m is the proposal), and use the routing file and mGV config of the T5.5a
  report (`benchmark/results/2026-11-14-T5.5a.md`).
- From the T5.5a review: with MODFLOW's baseflow and capillary rise fed back, mGV's
  routing on the global routing file produces discharges up to 1e37 m³/s and NaN from
  day 2 (mGV alone stays finite), which stopped MODFLOW when it took mGV's discharge
  (T5.5a report, "mGV's routing blows up"). Resolve it (owner decision, mGV is not
  patched here) and check mGV's discharge for finite, plausible values every day.
- Owner decision (2026-11-15): Greenland and the Arctic are not modelled. For now, mask
  out everything north of 60°N in both models (575,055 of mGV's 2,288,857 active cells,
  all 100,306 Greenland cells among them; the parameter grid runs from 56°S to 84°N).
  - mGV: a derived parameter file with `run_cell` cleared north of 60°N, written next to
    the original under `/home/bart/data/mGV/` (do not change the original; mGV itself is
    not patched), with the forcing and routing inputs cut to the same cells.
  - MODFLOW: clear `ibound` north of 60°N in the config or a derived file. The boundary
    at the cut is no-flow (owner, 2026-11-15).
  - Routing: rivers that cross 60°N are cut, which the owner accepts for now. Northward
    rivers (Ob, Yenisei, Lena, Mackenzie) end in an outlet at the cut; southward ones
    lose their upstream inflow. Report how many links are cut.
  - Check whether the mask removes the routing blow-up below (it started on Greenland
    and Arctic cells, where the cells and channels are shortest), and report the
    shortest channel length that remains.
- Owner decisions (2026-11-15) after the blow-up investigation:
  - Spin-up: MODFLOW's steady state is reached by running the coupled system for
    100–200 years, not by a steady-state solve that has to converge first. Report how
    the heads and the baseflow drift per year, and stop when they level off. Forcing:
    the gridded store `/media/bart/Data/Work/SWIFT/WFDE5_v3.1_5arcmin.zarr` is mounted
    since 2026-11-15. It is one zarr v3 store with all seven variables as
    `(time, lat, lon)` int16 arrays on the 4320 × 1680 grid, 16,802 daily steps
    (1979–2024), 239 GB. Cycle through this record for the spin-up (the default, unless
    the owner says otherwise), and check first that mGV `main` reads it; the land-only
    1979 store in `/home/bart/data/mGV/` is the fallback.
  - River loss is limited to the water present in the channel: MODFLOW must not take
    more from a river than mGV has there, so the exchange creates no water. Design this
    in the card (for example no infiltration from a RIV cell on a day when the channel
    volume is below the potential loss) and report the created water per day, which
    must be zero.
  - CHD ring heads: sea level (0 m) is fine for now.
- From the blow-up investigation (2026-11-15, no code changed; replaces the "likely
  cause" of the T5.5a review bullet above):
  - Cause: mGV's explicit routing (`A_new = max(A + (Q_in − Q_old)·dt/L, 0)`, 600 s
    sub-steps, 6 m/s cap) sends up to 3600·A m³ downstream per sub-step, more than the
    stored A·L on channels shorter than 3.6 km; the clamp at 0 creates the difference and
    it multiplies along chains of short cells. Positive baseflow is the trigger (it
    raises A and the velocity). A constant 0.1 m³/s into every channel of mGV alone, with
    no MODFLOW, blows up as well.
  - The 60°N mask removes it: south of 60°N the shortest channel is 4639 m. With the
    mask the real coupled model stayed finite for 10 days on the GPU and MODFLOW
    converged in 3–4 outer iterations per day. Without the mask the `_nogl` domain has
    110,669 channels below 3.6 km and is marginal. If the mask is ever lifted, mGV needs
    more routing sub-steps (576 per day was stable in a host replica) or an adaptive
    step; that is the owner's change in mGV.
  - Cell sizes do not create water with the real inputs (MODFLOW against mGV cell area
    0.9955–1.0056; recharge volumes agree within 0.06 %). The synthetic `global5m` does
    over-count recharge at high latitude (its `cell_area` is DELR·DELC).
  - The models start far from equilibrium. From the provided heads the first feedback is
    487 km³/day of baseflow, 59 times mGV's whole channel storage (8.2 km³, cold start),
    up to 3455 m³/s per cell on the Amazon; it falls to 95 km³/day by day 10. Decide how
    both models are spun up before the daily run (mGV's channels are nearly empty at a
    cold start), and report baseflow against channel storage per day.
  - The steady-state spin-up of the real masked model did not converge in 2000 outer
    iterations at closure 0.01 or 0.1 (`:mc_ilu0`, 1979 mean recharge and discharge of
    mGV alone, CHD ring at 0 m). This is the first thing to solve in this card (try
    `:amg`, see the T3.4 bullet).
  - River losses larger than the water in a channel are applied only partly
    (`add_baseflow_kernel!`): about 3 km³/day of created water from the provided heads.
    The coupling changes the channel area without updating Q, which created 3.9e6 m³.
    Report the created water per day and decide whether MODFLOW's river loss must be
    limited to the water present.
- From the T5.5a report:
  - Missing values in the chosen files: aqdepth 39.5 %, cprate 10.5 %. Check how
    `build_parameters` fills them.
  - Run the spin-up in `:mixed`: on the real Indus inputs `:single` stops about 0.03 m
    from the converged heads with either preconditioner.
  - T5.5a's iteration counts and closure costs were measured with a synthetic river
    discharge on `global5m` (6.59 M nodes; the real model has 4.72 M). Measure them
    again with mGV's discharge: time per day at closures 5, 0.01 and 1e-3.
  - `mGV.finalize!` frees no device memory (2428 MiB at global size), and dropping the
    model did not free it in the rehearsal script; the globals in
    `global_rehearsal.jl:227-232` may hold it.
  - The gridded WFDE5 stores were not checked (`/media/bart/Data` was not mounted); the
    land-only `WFDE5_v3.1_5arcmin_land_1979.zarr` covers the one year.
  - The routing file was built from two longitude boxes split at −25° (the global run
    needed more than 52 GB of RAM). The seam adds 131 outlets, all on Greenland.
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
