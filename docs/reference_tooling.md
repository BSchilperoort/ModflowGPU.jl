# Reference tooling (T0.2)

How to fetch the MODFLOW 6 reference material, how to load each libmf6 version from
Julia, and which libmf6 memory addresses the parity tests can use. Checked on
2026-11-06 with Julia 1.12.6, ModflowInterface 1.0.0, BasicModelInterface 0.1.1.

## 1. Fetching `reference/`

```sh
validation/fetch_reference.sh                 # into <repo root>/reference (gitignored)
REFERENCE_DIR=/elsewhere validation/fetch_reference.sh
scripts/apply_coupling_patches.sh             # reset ModflowCoupling clone + git am patches
validation/setup_libmf6_envs.sh               # two Julia envs for libmf6 6.3.0 / 6.8.2
```

All three (`fetch_reference.sh`, `apply_coupling_patches.sh`, `setup_libmf6_envs.sh`)
are idempotent and stop on the first error. `fetch_reference.sh` puts every download
and every extraction in its own new directory, checks sha256 digests, rejects zip files
with absolute or `..` paths or symbolic links, removes its temporary directory on
error or interrupt, and writes `reference/.gitignore` (`*`) so git never tracks the
data. To refresh the rolling examples release, remove both `reference/archives/mf6examples`
and `reference/mf6examples` (an existing extraction is never overwritten) and rerun;
the recorded checksum is replaced. Result (about 1.1 GB):

| Path | What | Size |
| --- | --- | --- |
| `archives/<name>/<asset>.zip` | the downloaded release assets, untouched | 152 MB |
| `checksums.txt` | sha256 of each archive | |
| `mf6.8.2_linux/` | `mf6.8.2_linux.zip` extracted: `bin/{mf6,libmf6.so,mf5to6,zbud6}`, plus `src/`, `dfns/`, `examples/`, `doc/` | 499 MB |
| `mf6.8.2_dfns/` | `mf6.8.2_dfns.zip` extracted (`*.dfn`) | 1.2 MB |
| `mf6examples/` | `mf6examples.zip` extracted: one directory per example (`ex-gwf-toth/`, ...; 115 in total) | 271 MB |
| `modflow6/` | shallow clone of MODFLOW-ORG/modflow6 at tag `6.8.2` | 61 MB |
| `Modflow-coupling/` | clone of SWIFT-WUR/Modflow-coupling, base `8dd8c010` (branch `modflowgpu` after `apply_coupling_patches.sh`) | 0.5 MB |
| `julia-env/mf6.3.0`, `julia-env/mf6.8.2` | throwaway Julia environments (section 2) | |
| `probe/mf<ver>-<example>/` | `libmf6_probe.jl` output (section 3) | |

Release assets (checked against the GitHub API): `mf6.8.2_linux.zip` and
`mf6.8.2_dfns.zip` from release `6.8.2` (published 2026-10-30), and `mf6examples.zip`
from MODFLOW-ORG/modflow6-examples release `current`. The names match PLAN.md. The
6.8.2 digests are pinned in the script. `current` is a rolling release, so its digest
is read from the API at download time and recorded in `checksums.txt`; the copy
fetched on 2026-11-06 (published 2026-10-29) has sha256 `cf40b4b9...c406`.

`bin/libmf6.so` of 6.8.2 links only against libc (static Fortran runtime), so it loads
without extra libraries. `bin/mf6 --version` prints `mf6: 6.8.2`.

## 2. Loading libmf6 from Julia

ModflowCoupling drives MF6 through ModflowInterface.jl:
`mf = ModflowModel(dir); BMI.initialize(mf)`, pointers via
`BMI.get_value_ptr(mf, MF.get_var_address(mf, var, component; subcomponent_name))`,
and a time step as `MF.prepare_time_step(mf, 0.0)`, `MF.prepare_solve(mf)`, repeated
`MF.solve(mf)`, `MF.finalize_solve(mf)`, `MF.finalize_time_step(mf)`. ModflowInterface
calls `Modflow6_jll.libmf6`, so the library is chosen by Modflow6_jll.

### MF6 6.3.0 (Modflow6_jll 6.3.0, as ModflowCoupling)

Nothing special: an environment with `ModflowInterface`, `BasicModelInterface` and
`Modflow6_jll = 6.3.0` (`reference/julia-env/mf6.3.0`). The library is
`~/.julia/artifacts/77d38efc.../lib/libmf6.so`; it reports `VERSION 6.3.0 03/04/2022`
in `mfsim.lst` and does not export the XMI `get_version` symbol.

### MF6 6.8.2 (release `libmf6.so`): yes, through a JLLWrappers preference

The same packages, plus this `LocalPreferences.toml` next to the environment's
`Project.toml` (written by `validation/setup_libmf6_envs.sh` into
`reference/julia-env/mf6.8.2`):

```toml
[Modflow6_jll]
libmf6_path = "/abs/path/reference/mf6.8.2_linux/bin/libmf6.so"
mf6_path = "/abs/path/reference/mf6.8.2_linux/bin/mf6"
```

JLLWrappers (1.8.0 here) reads `<product>_path` preferences at precompile time, so
`Modflow6_jll` and `ModflowInterface` recompile for that environment and
`Modflow6_jll.libmf6_path` then points at the 6.8.2 library (`get_version` returns
`6.8.2`). The preference only applies to sessions started with `--project` on that
environment. **No global `~/.julia/artifacts/Overrides.toml` is used or needed**, and
none was left behind. Modflow6_jll must be listed in the environment's `[deps]` or
`[extras]` for the preference to be picked up. Tested: with Modflow6_jll only as an
indirect dependency (as in ModflowCoupling's `Project.toml`) the preference is
silently ignored and 6.3.0 loads; adding `Modflow6_jll = "81230ed9-c869-5493-bf63-2c1575eaf49b"`
under `[extras]` is enough to switch to 6.8.2. A Milestone B environment that wants
6.8.2 must do that.

Pitfalls:

- ModflowInterface reads `BMI_LENVARADDRESS` and the other string lengths from the
  library when it is **precompiled**. They differ between versions
  (`BMI_LENVARADDRESS` = 51 in 6.3.0, 68 in 6.8.2). Never `dlopen` a different
  libmf6 behind a ModflowInterface precompiled for another one; use the preference so
  both are recompiled together. (Direct `ccall` on the library path also works, but
  then the string lengths must be read from that library's own `BMI_LEN*` globals.)
- One libmf6 instance per Julia process (MF6 keeps global state). Use separate
  processes for different versions.
- `ModflowInterface.get_var_type` ignores the BMI status; an unknown address gives an
  empty type string and `get_value_ptr` then errors with "unsupported type". Check the
  type first (as `libmf6_probe.jl` does).

### Check 1: 6.3.0 runs the 6.8-era examples

`ex-gwf-toth` (flopy 3.10.0.dev5 inputs) initialises and runs to the end with libmf6
6.3.0 through ModflowInterface: 3 outer iterations, heads 9984.21 to 11014.47 m.
`ex-gwf-hanir` and `ex-gwf-twri01` also run unchanged with 6.3.0. Against 6.8.2 the
first-solve `AMAT` and `RHS` are **bitwise identical** for all three, and the final
heads agree within 1.6e-14 relative (toth 1.7e-10 m, twri01 9e-13 m, hanir 4e-16 m).
Only these three decks were tried with 6.3.0; T0.3 runs the full ladder with 6.8.2.

## 3. Exposed addresses

`get_input_var_names` and `get_output_var_names` return the same list in both versions:
every variable in the MF6 memory manager (toth: 461 names in 6.3.0, 613 in 6.8.2;
full lists in `reference/probe/mf<ver>-<example>/var_names.txt`). `<MODEL>` is the
upper-case GWF model name (`TOTH`, `EX-GWF-HANIR`, ...).

Result of `validation/libmf6_probe.jl` on ex-gwf-toth (200×1×100, 20000 nodes, 39700
connections), ex-gwf-hanir and ex-gwf-twri01, for both versions. "Listed" = in the
var-name list; "read" = `get_value_ptr` returns an array of the expected length.

| Address | 6.3.0 | 6.8.2 | Type, length (toth) | Notes |
| --- | --- | --- | --- | --- |
| `SLN_1/IA` | listed, read | listed, read | Int32, neq+1 = 20001 | 1-based CSR row pointers |
| `SLN_1/JA` | listed, read | listed, read | Int32, nja = 99400 | diagonal first in every row |
| `SLN_1/AMAT` | listed, read | listed, read | Float64, nja | valid only after the first `solve`; undefined before it (seen zeros and NaN) |
| `SLN_1/RHS` | listed, read | listed, read | Float64, neq | valid after the first `solve` |
| `SLN_1/X` | listed, read | listed, read | Float64, neq | |
| `<MODEL>/CON/IA` | listed, read | listed, read | Int32, nodes+1 | same as `SLN_1/IA` for one model |
| `<MODEL>/CON/JA` | listed, read | listed, read | Int32, nja | |
| `<MODEL>/CON/JAS` | listed, read | listed, read | Int32, nja | 0 on the diagonal |
| `<MODEL>/CON/IHC` | listed, read | listed, read | Int32, njas = 39700 | per symmetric connection |
| `<MODEL>/CON/CL1`, `CL2` | listed, read | listed, read | Float64, njas | |
| `<MODEL>/CON/HWVA` | listed, read | listed, read | Float64, njas | |
| `<MODEL>/NPF/CONDSAT` | listed, read | listed, read | Float64, njas | |
| `<MODEL>/NPF/K11` | listed, read | listed, read | Float64, nodes | |
| `<MODEL>/X` | listed, read | listed, read | Float64, nodes | heads |
| `<MODEL>/DIS/NODEUSER` | listed, read | listed, read | Int32, 1 | dummy of length 1 with an undefined value when no cell is inactive (ModflowCoupling handles this) |

So **all requested addresses work in both 6.3.0 and 6.8.2**, at the same paths. In
6.8.2 the solution matrix is a `SparseMatrixType` whose memory path is still
`SLN_1` (`src/Utilities/Matrix/SparseMatrix.f90`, `allocate_arrays`), so the address
did not move. `AMAT` is exposed, so the assembly oracle test of section 4 can use it,
in either version.

What `AMAT`/`RHS` contain after a `solve` call: each XMI `solve` call is one outer
iteration, and the arrays hold the system that iteration assembled and solved.
`validation/check_probe_system.jl` confirms this: with `X` after the first `solve`,
`‖RHS − AMAT·X‖∞ / max_row Σ|a x|` is 1.5e-7 (toth), 1.4e-11 (hanir), 3.6e-13 (twri01),
i.e. the inner-solver tolerance. MF6 sign convention: negative diagonal, positive
off-diagonals (toth: values in [-8.5, 4.0]). Constant-head rows are identity rows
(diagonal 1, off-diagonals stored as 0, `RHS` = the head); other rows keep their
coupling to constant-head nodes. With IMS `SCALING_METHOD` other than none, IMS scales
`AMAT`/`RHS` in place during the linear solve and unscales afterwards, which may change
the last bits; parity runs should use no scaling. Newton (PTC, under-relaxation) runs
were not probed.

### Recipe for later cards

```sh
validation/fetch_reference.sh
validation/setup_libmf6_envs.sh
julia --project=reference/julia-env/mf6.8.2 validation/libmf6_probe.jl \
    reference/mf6examples/ex-gwf-toth reference/probe/mf6.8.2-toth   # out dir must not exist
julia validation/check_probe_system.jl reference/probe/mf6.8.2-toth
```

`libmf6_probe.jl` copies the example to `<out>/model`, prints a table like the one
above, writes `probe.toml`, `var_names.txt`, raw dumps of `SLN_1/{IA,JA,AMAT,RHS,X}`
after the first `solve` (`first_solve_*.bin`, Int32/Float64) and the final heads
(`final_X.bin`). Its `main` function can be `include`d and reused. For oracle tests,
load the 6.8.2 environment, run `prepare_time_step`, `prepare_solve`, one `solve`, then
copy `SLN_1/AMAT` and `SLN_1/RHS` before the next `solve` overwrites them.

## 4. libmf6 as a test oracle (T2.1)

Tests that compare with libmf6 memory run libmf6 in a **subprocess**, in the 6.8.2
environment of section 2, so ModflowInterface and Modflow6_jll stay out of the test
dependencies and every simulation gets a fresh libmf6 instance:

- `validation/libmf6_connectivity.jl <simdir> <outdir>` copies the simulation to
  `<outdir>/model`, initialises it (no time step), and dumps every GWF model's
  `<MODEL>/CON/{IA,JA,JAS,ISYM,IHC,CL1,CL2,HWVA,ANGLEX,IAUSR,JAUSR}` and
  `<MODEL>/DIS/{NODEUSER,NODEREDUCED,TOP,BOT,AREA,XC,YC}` as raw Int32/Float64 files
  `<outdir>/<MODEL>/<COMP>_<NAME>.bin`, with `index.toml` (types, lengths, and the scalars
  `DIS/{NODES,NODESUSER,NJA,NJAS}`). About 1.5 s per simulation.
- `test/libmf6_oracle.jl` runs such a script with
  `julia --startup-file=no --project=$MFGPU_LIBMF6_ENV` (default
  `$REFERENCE_DIR/julia-env/mf6.8.2`), with `JULIA_LOAD_PATH`/`JULIA_PROJECT` removed
  from the environment so it also works under `Pkg.test()`, and reads the dumps back
  (`libmf6_connectivity(simdir)`). Tests skip with a message when the environment's
  `.setup-complete` marker or `LocalPreferences.toml` is missing.
- `write_dis_simulation(dir, dis)` (same file) writes a minimal simulation (DIS with
  INTERNAL arrays written to read back exactly, NPF, IC, IMS, one steady period) for
  grids made in tests, so synthetic grids (e.g. with IDOMAIN < 0) can be checked against
  libmf6 too. MF6 reads 2D/3D arrays one row per Fortran READ, so each row starts on a
  new line.

Run by hand: `julia --project=reference/julia-env/mf6.8.2 validation/libmf6_connectivity.jl
reference/mf6examples/ex-gwf-toth /tmp/toth-con`.
