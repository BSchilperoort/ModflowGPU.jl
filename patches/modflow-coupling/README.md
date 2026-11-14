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
