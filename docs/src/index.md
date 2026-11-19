# ModflowGPU.jl

ModflowGPU.jl runs the groundwater-flow model (GWF) of [MODFLOW 6](https://github.com/MODFLOW-ORG/modflow6)
on a GPU or a CPU. It reads ordinary MODFLOW 6 input files (`mfsim.nam` and the files it
lists), writes the usual head, budget and listing files, and can also be driven one time step
at a time from Julia, for example to couple it to a land-surface or hydrology model. It is
written in Julia with [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
so the same code can run on AMD, NVIDIA, Intel and Apple GPUs, on OpenCL devices and on the CPU.
Only AMD (ROCm) and the CPU are tested so far.

MODFLOW 6 is the reference. A feature is only accepted when it reproduces MODFLOW 6 heads
and volumetric budgets on the same input. The first user is
[ModflowCoupling](https://github.com/SWIFT-WUR/Modflow-coupling), which couples MODFLOW 6 to the
[mGV](https://github.com/SWIFT-WUR/mGV) hydrology model for basin-to-global groundwater models.

## Status

The package is under development and not yet registered; install it from its Git URL (see
[Installation](install.md)). Supported today:

- one GWF model with one IMS solution, on DIS, DISV and DISU grids;
- NPF (confined and convertible cells, wetting and drying, anisotropy), STO, IC and OC;
- the boundary packages CHD, WEL, GHB, RIV, DRN, RCH and EVT, with time series (TS) and
  time-array series (TAS);
- the Picard outer iteration and the Newton–Raphson formulation, with backtracking and
  under-relaxation, and CG or BiCGSTAB inner solvers, in double, mixed or single
  precision, with volumetric budgets;
- the stepping API and BMI for coupling.

Not supported yet: XT3D, several models with exchanges, and observation
(OBS) output. Input that uses something unsupported is rejected with an error that names it,
never silently ignored. Packages for other processes (SFR, LAK, MAW, UZF, MVR, CSUB and the
transport models) are out of scope. See [Supported packages](compatibility/packages.md).

## When to use it

- You have a large MODFLOW 6 GWF model (millions of cells, for example a regional or global
  model) and a GPU, and want shorter run times.
- You run MODFLOW 6 inside another model, one time step at a time, and the groundwater part
  has become the slow part.
- You want to write your own coupling in Julia and need direct access to heads, boundary
  data and budgets as arrays.

## When not to use it

- Your model needs a package or option listed as unsupported. Use MODFLOW 6.
- Your model is small. A GPU only pays off when there is enough work to keep it busy, so
  on a model of a few thousand cells do not expect a speed-up over MODFLOW 6 on the CPU.
- You need identical iteration counts or listing files. ModflowGPU uses a different
  preconditioner, so iteration counts differ; heads and budgets agree within the closure
  criteria of the model.
- You need MODFLOW 6's transport models or its full set of output files.

## How to cite

There is no paper yet. Until a release is tagged, cite the repository:

> Schilperoort, B. ModflowGPU.jl: MODFLOW 6 groundwater flow on GPUs in Julia.
> <https://github.com/BSchilperoort/ModflowGPU.jl>

Please also cite MODFLOW 6 (Langevin et al., 2017,
[doi:10.3133/tm6A55](https://doi.org/10.3133/tm6A55)), whose equations and input format this
package implements.

## AI declaration

This package is developed with AI coding agents. The plan was written with AI assistance
and is carried out one task at a time: an agent writes the code, tests and documentation on
a branch, and a separate agent reviews it. The owner sets the scope and makes design
decisions. Correctness is checked against MODFLOW 6 itself, not by the agents' own
judgement. All of the code was written by AI. The full declaration, in the
[AI Declaration Format](https://ai-declaration.org/), is in
[aidecl.yaml](https://github.com/BSchilperoort/ModflowGPU.jl/blob/main/aidecl.yaml).

## Where to go next

- [Installation](install.md) and [Getting started](getting_started.md) for a first run.
- [Supported packages](compatibility/packages.md) to check that your model is supported.
- [Stepping API](coupling/stepping.md) to drive the model from Julia.
- [For developers](developers.md) if you want to change the code.
