# GPU guide

```@meta
CurrentModule = ModflowGPU
DocTestSetup = :(using ModflowGPU)
```

This page explains what changes when a model runs on a GPU: choosing the device, the
precision presets and what they mean for the accuracy of heads and budgets, how much GPU
memory a model needs, and what to do when something goes wrong. It assumes you have
installed a GPU package as described in [Installation](install.md).

Unless stated otherwise, the numbers on this page were measured on an AMD Radeon RX 7800 XT
(16 GB) with ROCm 7.2, on the synthetic regional and global test models of the
repository and on the MODFLOW 6 example models. Other GPUs have not been measured.

## Running on the GPU

[`run_simulation`](@ref) and [`initialize`](@ref) run on the CPU unless you pass a GPU
backend. Load the GPU package, select the backend and pass it on:

```julia
using AMDGPU, ModflowGPU      # needs an AMD GPU; CUDA, oneAPI, Metal or OpenCL likewise

backend = select_backend("auto")
result = run_simulation("path/to/simulation"; backend)
```

Everything else stays the same: the input files, the output files and the
[`RunResult`](@ref). The model is read on the CPU, copied to the GPU once, and every
iteration runs on the GPU; heads and budgets are copied back only when the output control
asks for them. The listing file states the backend and the precision in its header, for
example `BACKEND: ROCBackend, PRECISION: T = Float64, Ts = Float64, Tp = Float32`.

On the AMD GPU, heads were bitwise identical to those of the CPU backend on the test
models, in every precision.

## Precision presets

A GPU computes much faster in single precision (Float32, about 7 significant digits) than
in double precision (Float64, about 16 digits), and needs half the memory per value. Some
GPUs have no Float64 at all. ModflowGPU therefore has four precision presets, chosen with
the `precision` keyword:

| Preset | Heads and flows | Linear solver | Preconditioner | Needs Float64 |
| --- | --- | --- | --- | --- |
| `:double` | Float64 | Float64 | Float64 | yes |
| `:mixed` | Float64 | Float64 | Float32 | yes |
| `:single` | Float32 | Float32 | Float32 | no |
| `:auto` (default) | `:mixed` where the backend has Float64, else `:single` | | | |

[`resolve_precision`](@ref) shows what a preset means on a backend. MODFLOW 6 itself
always computes in Float64.

### What the presets mean for accuracy

- **`:double` and `:mixed`** give the accuracy of MODFLOW 6. Only the preconditioner, which
  speeds up the linear solver but does not decide the answer, is stored in Float32 in
  `:mixed`. With tight solver closures, heads agree with MODFLOW 6 within 4e-8 m on the
  test models, and within 4.4e-11 m on the real global model (3.7 million cells, 31 daily
  steps), where the budget terms agree to 1e-12 relative.
- **`:single`** gives heads within 2e-4 m of MODFLOW 6 on most test models, and within
  6e-4 m on all of them (the largest, `ex-gwf-advtidal`, has convertible cells and tides).
  On the real global model the largest head difference after 31 daily steps was 2.5e-5 m.
  Percent discrepancies are mostly below 0.003 %, but see the budget item below.

Float32 stores about 7 digits, so its error grows with the size of the numbers stored.
ModflowGPU stores each head as the difference from a reference head of its cell (normally
the starting head), which keeps those numbers small. The error of a `:single` head is
therefore roughly 1e-7 times how far the head moves away from its starting head: 1e-4 m
for a head that moves 1000 m, much less for most models.

What you will notice in `:single`:

- **The closures have a floor.** Float32 cannot resolve head changes below a certain
  size, so OUTER_DVCLOSE and INNER_DVCLOSE below 1e-6 m (or below 1e-7 times the largest
  head offset) are raised, with a logged message such as
  `Float32 run: OUTER_DVCLOSE 1.0e-9 and INNER_DVCLOSE 1.0e-10 raised to at least 1.0e-6 and 1.0e-6 (FLOAT32_CLOSURES)`.
  INNER_RCLOSE gets a similar floor. The model then often converges in fewer outer
  iterations than in MODFLOW 6.
- **Steady states can stop short.** On a badly conditioned steady state (a regional
  spin-up of the Indus basin with real inputs) the `:single` heads stopped 3 to 7 cm from
  the converged Float64 heads, whatever the preconditioner and closure. Run spin-ups and
  other long steady states in `:mixed` (the `:auto` default where Float64 exists).
- **Some models do not converge** with the default preconditioner in `:single`. On two
  of the MODFLOW 6 examples (`ex-gwf-lgrv-gr`, `ex-gwf-hwell-a` with the Jacobi
  preconditioner) the run fails with a [`ConvergenceError`](@ref); `preconditioner = :amg`
  or `:mixed` fixes it.
- **Budgets with very large coefficients are rounded coarsely.** A flow is computed from
  Float32 heads, so where a very large conductance or storage coefficient meets a small
  flow, the rounding of the heads shows in the flow. On a radial test grid with
  conductances of 5.5e7 ft²/d next to the well, single FLOW-JA-FACE values differed by
  up to 2e-4 of the total inflow, while the heads agreed within 1.5e-6 ft. On three of
  the MODFLOW 6 example models with convertible cells, the `:single` percent discrepancy
  reached 0.02 % to 48 % (the largest on `ex-gwf-rad-disu`), while the heads were fine. Use `:mixed` where budgets and cell-by-cell flows matter.

### Which preset to use

| Situation | Preset |
| --- | --- |
| Any model, on a GPU or CPU with Float64 (AMD, NVIDIA, the CPU) | `:auto` (= `:mixed`) |
| A GPU without Float64 (Apple, some Intel) | `:auto` (= `:single`; the only option) |
| Transient runs where speed matters more than the last 0.1 mm, after checking a run against `:mixed` | `:single` |
| Comparing with MODFLOW 6 to the last digit, or checking that `:mixed` is not the cause of a difference | `:double` |

Speed, on the synthetic global model (6.6 million cells, a steady state and 30 daily
steps, default preconditioner): `:double` 9.6 s, `:mixed` 8.7 s, `:single` 3.6 s in total
(`benchmark/results/2026-11-16-T4.7.md`). On the CPU backend `:mixed` and `:double` take
about the same time. More numbers are on the [Performance](performance.md) page.

## Preconditioner

The linear solver uses CG or BiCGSTAB as IMS asks, but not IMS's ILU0, MILU0 or ILUT
preconditioner: those do not run in parallel on a GPU. ModflowGPU replaces it with its own,
chosen with the `preconditioner` keyword:

| `preconditioner` | What it is | When to use it |
| --- | --- | --- |
| `:auto` (default) = `:mc_ilu0` | ILU0 in multicolour order | almost always: the fastest over a whole run on every benchmark model, backend and precision |
| `:amg` | algebraic multigrid | badly conditioned steady states, models that do not converge with `:mc_ilu0` (needs about 20 % more memory) |
| `:jacobi` | diagonal scaling | least memory; needs many more iterations |
| `:mc_sgs`, `:chebyshev` | multicolour Gauss–Seidel, Chebyshev polynomial | for experiments |

`precond_options` passes settings to the preconditioner: `relax` for `:mc_ilu0` (the MILU
relaxation factor, IMS RELAXATION_FACTOR, which is not taken from the IMS file), `degree`
for `:chebyshev`, and the AMG settings for `:amg`. The defaults are a good start.

Because the preconditioner differs, the number of inner iterations differs from MODFLOW 6.
The heads still agree within the closure criteria.

## Memory

The whole model lives on the GPU: the matrix, the solver vectors, the preconditioner and
the boundary data. Peak GPU memory per million active cells, measured on the synthetic
global model (6.6 million cells in 2 layers, with a river or a drain, recharge and
evapotranspiration on every land cell, `benchmark/results/2026-11-08-T4.4.md`):

| Preconditioner | `:double` | `:mixed` | `:single` |
| --- | --- | --- | --- |
| `:mc_ilu0` (default) | 0.57 GiB | 0.54 GiB | 0.37 GiB |
| `:amg` | 0.69 GiB | 0.65 GiB | 0.46 GiB |
| `:jacobi` | 0.46 GiB | 0.45 GiB | 0.29 GiB |

Later solver changes added about 0.06 GiB per million cells: with the default
preconditioner the same model now peaks at 4.2 GiB in `:double`, 4.0 GiB in `:mixed` and
2.8 GiB in `:single` (`benchmark/results/2026-11-16-T6.1.md`), 0.64, 0.61 and 0.43 GiB per
million cells. As a rule of thumb, plan for about 0.65 GiB per million cells in `:mixed`
and 0.45 GiB in `:single`; in `:mixed` the synthetic global model uses a quarter of a
16 GB card.
Models with more connections per cell (DISV, DISU) or many more boundary records need more.

## Sharing the GPU with another model

ModflowGPU can share one GPU, in one Julia process, with another model that is also
written with KernelAbstractions, such as the mGV hydrology model in ModflowCoupling. This
was tested with mGV on the AMD GPU: a one-year daily coupled run of the Indus basin, with
mGV using 64 MiB and ModflowGPU 23 MiB of GPU memory. Things to know:

- Both models must use the same versions of KernelAbstractions and the GPU package, so
  they must be in one Julia environment that resolves.
- Select the backend once and pass the same backend to both models.
- [`finalize_simulation`](@ref) releases the GPU memory of a simulation at once (with
  `free = true`, the default), instead of when Julia's garbage collector runs. Call it
  before the other model allocates a lot of memory.
- The [stepping API](coupling/stepping.md) can hand device arrays to the other model
  (`get_value_ptr(sim, address; device = true)`), so data need not pass through the CPU.

Two separate Julia processes on one GPU have not been tested.

## Troubleshooting

### The first run is slow

Julia compiles the code of ModflowGPU for your backend and precision the first time it is
used. On the CPU of the development machine (Ryzen 7 7700) the first run of the small
model of [Getting started](getting_started.md) takes about 20 s, the second less than a
second. A run in another precision or with another preconditioner compiles again, a few
seconds each. Keep one Julia session open and run several models in it, instead of
starting Julia for every run; the compiled code is kept for as long as the session runs.
Loading the package (`using ModflowGPU`) takes a few seconds more.

### `does not support Float64`

```
precision = :double was requested, but backend MetalBackend does not support Float64. Use precision = :single instead.
```

The GPU has no Float64 (Apple GPUs, some Intel GPUs and OpenCL devices), so `:double` and
`:mixed` cannot run. Use `precision = :single`, or leave the default `:auto`, which picks
`:single` there.

### Out of GPU memory

The GPU package throws an out-of-memory error when the model does not fit. Check the
[memory](@ref "Memory") table above, then:

- use `precision = :single` (about 30 % less memory), or `preconditioner = :jacobi`;
- call [`finalize_simulation`](@ref) on simulations you no longer need, and close other
  programs that use the GPU;
- run on the CPU (`backend = select_backend("cpu")`), which is slower but limited only by
  the computer's memory.

### `Scalar indexing is disallowed`

This error comes from the GPU package when Julia code reads or writes a GPU array one
element at a time, which would be extremely slow. ModflowGPU never does this itself; it
happens when you index a device array, for example one from
`get_value_ptr(sim, address; device = true)`, outside a kernel. Use the host copy instead
([`get_value`](@ref), or [`get_value_ptr`](@ref) without `device = true`), copy the whole
array with `Array(x)`, or use whole-array operations (`x .+= 1`, `sum(x)`), which run on
the GPU.

### The run stops with a `ConvergenceError`

```
ConvergenceError: the solution did not converge in time step 1 of stress period 1 (500 outer iterations); add CONTINUE to the mfsim.nam OPTIONS to continue
```

As in MODFLOW 6, a time step that does not converge within OUTER_MAXIMUM stops the run
after its output is written, unless `mfsim.nam` has the CONTINUE option. If the same input
converges in MODFLOW 6:

- in `:single`, try `:mixed` (on a GPU with Float64) or `preconditioner = :amg`;
- with the default preconditioner, try `preconditioner = :amg`;
- if the log shows `the correction or the heads are not finite (the linear solver
  diverged)`, the linear solver failed: the same two remedies apply.

### The CPU run is slow

The CPU backend runs the same code as the GPU, on the CPU threads Julia was started with.
Start Julia with `--threads=auto`. The CPU backend is meant for testing and for small
models; on large models a GPU is many times faster (see [Performance](performance.md)).
