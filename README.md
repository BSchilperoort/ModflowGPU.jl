# ModflowGPU.jl

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![AI Declaration Format](https://img.shields.io/badge/AI_Declaration_Format-validated-blue)](aidecl.yaml)

A GPU port of the groundwater-flow (GWF) model of [MODFLOW 6](https://github.com/MODFLOW-ORG/modflow6),
written in Julia with [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
It reads standard MF6 input files and runs on the CPU or on AMD, NVIDIA, Intel and Apple GPUs.

The first user is [ModflowCoupling.jl](https://github.com/SWIFT-WUR/Modflow-coupling), which couples
MODFLOW 6 to [mGV](https://github.com/SWIFT-WUR/mGV) through BMI for basin-to-global groundwater
models. MODFLOW 6 is the reference: every feature is accepted only when it reproduces MF6 heads and
budgets.

**Status:** under development, not yet registered. Supported now: DIS, confined NPF, STO, IC, CHD,
WEL, RCH, GHB, RIV, DRN, EVT, OC, the Picard outer loop with CG/BiCGSTAB, and budgets. See
[docs/support_matrix.md](docs/support_matrix.md) for details and [PLAN.md](PLAN.md) for the roadmap.

## Installation

Requires Julia 1.12 or newer.

```julia
using Pkg
Pkg.add(url = "https://github.com/BSchilperoort/ModflowGPU.jl")
```

GPU backends are optional package extensions. Add the package for your GPU alongside ModflowGPU:

| GPU | Package |
| --- | --- |
| AMD | `AMDGPU` |
| NVIDIA | `CUDA` |
| Intel | `oneAPI` |
| Apple | `Metal` |
| OpenCL devices | `OpenCL` |

```julia
Pkg.add("AMDGPU")  # for example
```

## Quick start

```julia
using AMDGPU, ModflowGPU   # omit the GPU package to run on the CPU

backend = ModflowGPU.select_backend("auto")   # first working GPU, else CPU
result = ModflowGPU.run("path/to/simulation"; backend)

result.converged
result.heads         # final heads per active cell (Float64)
```

`run` reads `mfsim.nam` and writes the head, budget and listing files that the OC package asks
for. Useful options:

- `precision`: `:auto` (default), `:double`, `:mixed` or `:single`. `:auto` picks `:mixed` (Float64
  state, Float32 preconditioner) on backends with Float64 support, otherwise `:single`.
- `preconditioner`: `:auto` (default, multicolour ILU0), `:jacobi`, `:amg`, and others.

For coupled models, `ModflowGPU.initialize` and the BMI-style stepping API run one time step at a
time. ModflowCoupling uses this API.

## Documentation

- [docs/support_matrix.md](docs/support_matrix.md): supported MF6 packages and options
- [docs/mf6_notes.md](docs/mf6_notes.md): how MF6 behaviour is mapped, and where it differs
- [docs/precision.md](docs/precision.md): precision presets and accuracy
- [docs/scaling.md](docs/scaling.md) and [benchmark/results/](benchmark/results/): performance

## Development

```sh
julia --project -e 'using Pkg; Pkg.test()'                     # CPU tests
MFGPU_TEST_BACKEND=rocm julia --project=test test/runtests.jl  # GPU tests (AMD)
```

More details on the tests, the reference data and the formatter (Runic) are in [CLAUDE.md](CLAUDE.md).

## AI use

This package is fully developed with AI coding agents, which implement and review
each task card of the (LLM generated) plan. Every feature is validated by LLM agents
against MODFLOW 6. Details are in [aidecl.yaml](aidecl.yaml), following
the [AI Declaration Format](https://ai-declaration.org/).
