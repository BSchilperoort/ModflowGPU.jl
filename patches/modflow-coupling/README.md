# ModflowCoupling patches

Changes that ModflowGPU needs in
[SWIFT-WUR/Modflow-coupling](https://github.com/SWIFT-WUR/Modflow-coupling) are kept
here as `git format-patch` files. They are never pushed upstream and never made in the
owner's checkout (`/home/bart/data/modflow/ModflowCoupling`). See PLAN.md section 2.

**Base commit:** `8dd8c010796fd3648b855bed51195c0f2e61dd60` (2026-10-19,
"Make config contain all required info").

## Workflow

1. `validation/fetch_reference.sh` clones the upstream repository into
   `reference/Modflow-coupling` at the base commit.
2. `scripts/apply_coupling_patches.sh` resets that clone to the base commit (local
   branch `modflowgpu`) and applies `*.patch` from this directory in lexical order with
   `git am`. It discards uncommitted changes and runs `git clean -fdx`, which also
   deletes **ignored** files in the clone (for example a local `Manifest.toml` or
   test output).
3. Make your change in the clone as commits on top of the applied patches, then export
   all of them again, removing the old patch files first so renamed or dropped
   commits do not leave stale patches behind:

   ```sh
   rm -f patches/modflow-coupling/*.patch
   cd reference/Modflow-coupling
   git format-patch --zero-commit --no-signature \
       8dd8c010796fd3648b855bed51195c0f2e61dd60..HEAD -o ../../patches/modflow-coupling/
   ```

   (`--zero-commit --no-signature` keep the patch files stable between exports. The
   `../../patches` path assumes the default `REFERENCE_DIR=<repo root>/reference`;
   with another `REFERENCE_DIR`, pass the absolute path of `patches/modflow-coupling/`.)

   Use one numbered patch per card, named after the card
   (`0001-T5.1-engine-switch.patch`, ...): start the commit subject with the card id.
   Commit the patch files in this repository and add a row to the table below.
4. The first patch adds a `[sources]` entry `ModflowGPU = {path = "../.."}` so the
   patched clone uses ModflowGPU from this repository.

If the base must move to a newer upstream commit, rebase all patches, update the base
commit here, in `validation/fetch_reference.sh` and in
`scripts/apply_coupling_patches.sh`, in one PR.

## Patches

| Patch | Card | Description |
| --- | --- | --- |
| `0001-T5.1-engine-switch.patch` | T5.1 | `[sources]` ModflowGPU; `[solver] engine = "mf6" \| "gpu"`, `backend`, `precision`, `preconditioner` (`"auto"`); GPU engine built in memory (`build_spec`, builder API) and run through the stepping API; RCH per unit area; river stages absolute into the BOUND mirror (offsets from `GWF/HREF`); tests with both engines (`MODFLOWCOUPLING_ENGINES`), Manifest.toml updated |
| `0002-T5.3-coupling-parity.patch` | T5.3 | `test/coupling_parity.jl`: GPU (`:single`, `:mixed`, `:double`) vs mf6 engine on the Indus, 3 monthly and 31 daily periods, tight closures (section 4 coupling criteria) and production closures (logged), closure table with `MODFLOWCOUPLING_CLOSURE_TABLE=1`; AMDGPU as a test dependency and `MODFLOWCOUPLING_BACKEND` (`auto`, `cpu`, `rocm`); test groups (`Pkg.test(test_args = ["parity"])` or `MODFLOWCOUPLING_TESTS`) |
| `0003-T5.2-device-coupling.patch` | T5.2 | Device-resident coupling for the GPU engine (`src/device_coupling.jl`): `apply_stresses!` / `collect_outputs!` as KernelAbstractions kernels on the engine's device arrays (`get_value_ptr(...; device = true)`), record → cell maps on the device, river stage in offset form, `groundwater_depth` as `top' − h'`; inputs copied to the device in the BMI setters, host outputs refreshed when read, no engine host mirrors; `[solver] coupling = "auto" \| "device" \| "host"`; `BMI.get_value_ptr(m, name; device = true)`, `head_reference`; KernelAbstractions as a dependency (Manifest.toml re-resolved); tests device vs host coupling |
| `0004-T5.4-mgv-one-gpu.patch` | T5.4 | mGV and ModflowGPU on one GPU: `set_value_device!` / `get_value_device!` (device arrays of another model through a cell map, gather and Float32 ↔ engine-precision conversion in one kernel; host fallback without the device coupling); device BOUND writes through `ModflowGPU.device_bound!`; host arrays handed out (`m.head`, `get_value_ptr`) kept current across updates and direct input writes uploaded (`expose!`, `sync_inputs!`); an error for several records of a package on one cell (`record_map`); `finalize!` frees the coupling's device arrays; `outer_dvclose`/`inner_dvclose` default per engine (gpu 0.01, mf6 5; `dvclose`, `DEFAULT_DVCLOSE`), note in `indus.toml`; `examples/mgv_coupling`: mGV `main` (unstructured grid of active cells) cell maps, device exchange with the feedback applied to mGV's state in kernels (`exchange = :device`, host path kept), `[sources]` for ModflowCoupling, ModflowGPU and mGV (42aebaa), AMDGPU and its Manifest.toml; tests: device exchange vs host path (bitwise), host arrays across updates, `foc = false` with per-record WEL, dvclose defaults |
| `0005-T5.5a-global-rehearsal.patch` | T5.5a | Models from arrays: `StaticInputs(; ...)` keyword constructor, `GroundwaterModel(config, inputs)` (no NetCDF; `validate(config; files = false)`); `src/cell_maps.jl`: `cell_maps` by binary search (`coordinate_index`) instead of the per-cell `argmin`, `check_cell_map` (host check of the gather maps), used by `examples/mgv_coupling/coupling.jl`; `examples/mgv_coupling/global_rehearsal.jl` (mGV global + synthetic `global5m` MODFLOW on one GPU: memory per stage, spin-up, daily coupled steps); `examples/global/`: draft global MODFLOW config (`global.toml`), mGV global config, `check_inputs.jl` (global input checks), `merge_routing_tiles.py` (routing file from longitude boxes); `initialize_gpu` documents the `:single` spin-up preconditioner decision; tests: model from arrays bitwise equal to the one from files, cell maps against `argmin`, map checks, steady-state spin-up in `:single` and `:double` |
| `0006-T5.5b-global-run.patch` | T5.5b | River-loss limit: BMI input `river_water_volume` (m³ per cell, default Inf = no limit), `river_bottom` raises RIV RBOT to `stage − 0.999·volume/(cond·days)` on the host and device paths (the bottoms as read are restored where the limit does not apply), so a cell cannot lose more than its channel holds; `examples/mgv_coupling/coupling.jl`: `Coupling(...; limit_river_loss = true)` sends mGV's channel volumes every day, `add_baseflow!`/`add_baseflow_kernel!` count the created water (clamp at 0) and set mGV's discharge of the new channel area (`channel_discharge`); `examples/global/`: `make_60N_inputs.jl` (derived inputs of the 60°N mask, CHD heads at 0 m, DEM check), `global_60N.toml`, `mgv_global_60N.toml`, `run_global.jl` (coupled spin-up with the 1979 forcing cycled, checkpoints and resume, yearly drift log and stopping criterion; timed daily runs), `compare_indus.jl`; tests: river-loss limit (unset, Inf and a large volume bitwise equal; capped at half the loss; dry channels) on mf6 and on gpu `:single`/`:double` with the device and host coupling |
| `0007-T5.6-global-mf6-parity.patch` | T5.6 | Global parity and timing against MODFLOW 6 (examples only, no change in `src/` or `test/`): `examples/global/parity_mf6.jl` (`mode=record`: the coupled GPU run from the year-200 checkpoint, writing mGV's daily discharge, subsurface runoff and channel volume as sent to MODFLOW; `mode=replay`: MODFLOW alone on the mf6 or GPU engine with those inputs from the same heads, open loop, timed per stage: input, solve, output; per-day heads, baseflow, capillary rise and RIV flows saved), `parity_compare.jl` (`mode=compare`: per-day differences with the T5.3 metric, location, water-table depth and boundary records of the largest, budgets, river-loss limit against its formula); README section |

## ModflowGPU stays a hard dependency (T5.4 decision)

The patched ModflowCoupling keeps ModflowGPU as a regular dependency, not a weak one
(extension): `[solver] engine = "gpu"` must work from a config alone in BMI hosts that only
call `BMI.initialize(GroundwaterModel, config)` (an extension would need `using ModflowGPU`
first); the GPU engine types are fields of `GroundwaterModel` (`Simulation`,
`DeviceCoupling`) and half of the package (`gpu_engine.jl`, `device_coupling.jl`) would move
into the extension with untyped fields; ModflowGPU loads without a GPU (0.54 s, GPU packages
are its own weak dependencies), and its KernelAbstractions stack is already loaded by mGV in
the coupled environment. Revisit if the GPU engine goes upstream and mf6-only users object
to installing KernelAbstractions.
