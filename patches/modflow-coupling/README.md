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
   branch `modflowgpu`, uncommitted changes discarded) and applies `*.patch` from this
   directory in lexical order with `git am`.
3. Make your change in the clone as commits on top of the applied patches, then export
   all of them again:

   ```sh
   cd reference/Modflow-coupling
   git format-patch 8dd8c010796fd3648b855bed51195c0f2e61dd60..HEAD -o ../../patches/modflow-coupling/
   ```

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
| (none yet) | | |
