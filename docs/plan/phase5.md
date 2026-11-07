# Phase 5: ModflowCoupling integration and Milestone B

Task cards for this phase of [PLAN.md](../../PLAN.md). Read PLAN.md sections 1–4 and 6 for the goals, conventions, architecture and validation rules these cards refer to.

Size: S ≈ < 1 day of agent work, M ≈ 1–3 days, L ≈ > 3 days.
`‖` marks cards that can run in parallel with their siblings.

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
