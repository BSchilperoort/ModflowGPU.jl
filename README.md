# ModflowGPU.jl

[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://bschilperoort.github.io/ModflowGPU.jl/dev/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![AI Declaration Format](https://img.shields.io/badge/AI_Declaration_Format-validated-blue)](aidecl.yaml)

A GPU port of the groundwater-flow (GWF) model of [MODFLOW 6](https://github.com/MODFLOW-ORG/modflow6),
written in Julia with [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
It reads standard MF6 input files and runs on the CPU or on AMD, NVIDIA, Intel and Apple GPUs.

The first user is [ModflowCoupling.jl](https://github.com/SWIFT-WUR/Modflow-coupling), which couples
MODFLOW 6 to [mGV](https://github.com/SWIFT-WUR/mGV) through BMI for basin-to-global groundwater
models. MODFLOW 6 is the reference: every feature is accepted only when it reproduces MF6 heads and
budgets.

**Status:** under development, not yet registered. Supported now: DIS, DISV, DISU, NPF
(confined and convertible), STO, IC, CHD, WEL, RCH, GHB, RIV, DRN, EVT, OC, time series, the Picard
outer loop with CG/BiCGSTAB, and budgets. See
[docs/dev/support_matrix.md](docs/dev/support_matrix.md) for details and [PLAN.md](PLAN.md) for the roadmap.

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
```

`run` reads `mfsim.nam` and writes the head, budget and listing files that the OC package asks
for. Options (`precision`, `preconditioner`) and the stepping API for coupled models are
described in the [documentation](https://bschilperoort.github.io/ModflowGPU.jl/dev/).

## Documentation

The manual is at <https://bschilperoort.github.io/ModflowGPU.jl/dev/>. Pages that are not
written yet are covered by the developer notes in [docs/dev/](docs/dev/):

- [support_matrix.md](docs/dev/support_matrix.md): supported MF6 packages and options
- [mf6_notes.md](docs/dev/mf6_notes.md): how MF6 behaviour is mapped, and where it differs
- [precision.md](docs/dev/precision.md): precision presets and accuracy
- [scaling.md](docs/dev/scaling.md) and [benchmark/results/](benchmark/results/): performance

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
