# Compatibility: KernelAbstractions, GPU packages and mGV

Checked 2026-11-14 for T5.0 on the development machine (Julia 1.12.6, RX 7800 XT,
ROCm 7.2.2 in `/opt/rocm`). T0.1 (2026-11-06) supported KA 0.9 and 0.10; since T5.0
KernelAbstractions (KA) 0.10 is the minimum (PLAN.md section 8, decision 7), because mGV
moved to KA 0.10.

## Compat entries

| Package | Compat | Why this lower bound |
| --- | --- | --- |
| KernelAbstractions | `0.10` | KA 0.9 support dropped in T5.0 |
| AMDGPU | `2.11` | the tested version, and mGV's minimum (2.9 and 2.10 resolve with KA 0.10, see below) |
| CUDA | `6.5` | lowest release that resolves with KA 0.10 (6.4 and older need KA 0.9) |
| Metal | `1.12` | lowest release that resolves with KA 0.10 |
| oneAPI | `2.10` | lowest release that resolves with KA 0.10 |
| OpenCL | `1` | OpenCL 0.10 needs KA 0.9 (T0.1 had `0.10`, which cannot resolve with KA 0.10) |

The CUDA, Metal and oneAPI bounds were found by resolving ModflowGPU with each release in
a fresh environment; the older bounds (`6.2.0`, `1`, `2`) still resolved, but only to these
releases, so the entries now say so. They also match mGV's compat (CUDA 6.5.1, Metal 1.12,
oneAPI 2.10, OpenCL 1).

Resolved versions (fresh resolves, 2026-11-14):

| Environment | KA | AMDGPU | Others | Result |
| --- | --- | --- | --- | --- |
| Root workspace (`test/`, `benchmark/`) | 0.10.7 | 2.11.0 | GPUArrays 12.2.0, KernelInterface 0.4.6, GPUCompiler 2.22.2, pocl_standalone_jll 7.2.3 | CPU (POCL) and `MFGPU_TEST_BACKEND=rocm` suites pass (numbers in the T5.0 report) |
| ModflowGPU + all five GPU packages + `pocl_jll` | 0.10.7 | 2.11.0 | CUDA 6.5.1, Metal 1.12.0, oneAPI 2.10.0, OpenCL 1.0.0 | all five extensions load; `select_backend("rocm")` and `select_backend("opencl")` (POCL device, Float64 supported) work |
| ModflowGPU + mGV `main` (42aebaa) + AMDGPU + GPUArrays | 0.10.7 | 2.11.0 | GPUArrays 12.2.0; adding CUDA, Metal, oneAPI, OpenCL resolves to the versions above | both `select_backend("rocm")` return `ROCBackend()` |

Which AMDGPU goes with which KA (General registry, checked by resolving):

| AMDGPU | Relation to KA |
| --- | --- |
| 1.3.4 – 2.8 | hard dependency, `KernelAbstractions = "0.9.2 - 0.9"`: cannot load with ModflowGPU any more |
| 2.9 – 2.10 | depends on KernelInterface 0.4; KA is a weak dependency (extension) with weak compat `0.10`. Resolves with ModflowGPU and the extension loads (2.9.2 and 2.10.0 also unregister wrapped host memory), but the rocm suite was not run on them |
| ≥ 2.11 | hard dependency, `KernelAbstractions = "0.10"` |

## mGV in the same environment (Milestone B)

mGV `main` (since 2026-11-13) is a package with GPU weak dependencies and compat
`KernelAbstractions = "0.10.7"`, `AMDGPU = "2.11"`, `CUDA = "6.5.1"`, `Metal = "1.12"`,
`oneAPI = "2.10"`, `OpenCL = "1"`. Every entry overlaps with ModflowGPU's, so the two
resolve together to KA 0.10.7 and AMDGPU 2.11.0 (table above), and the shared environment
of T5.4 runs on the same versions as the ModflowGPU tests and benchmarks.

## KA 0.10 behaviour that matters here

- **`CPU` is POCL.** In KA 0.10, `CPU` is an alias for `POCLBackend`: CPU kernels are
  compiled to OpenCL/SPIR-V and run through POCL. `backend_name(CPU())` is
  `"POCLBackend"`. CPU numbers depend on the thread setting (`JULIA_KA_CPU_THREADS`);
  record it with every CPU measurement.
- **`KernelAbstractions.supports_float64` defaults to `false`** (via KernelInterface) for
  backends that do not define it. So `ModflowGPU.supports_float64` has its own default of
  `true` (section 3.7 of PLAN.md) and does not forward to KA. Metal, oneAPI and OpenCL get
  explicit methods in the extensions.
- **`@synchronize` is a plain barrier call**, so it may sit in a loop or a helper function
  (`_tree_reduce!` in `src/solver/krylov.jl`). KA 0.9's CPU backend only accepted it in
  the kernel body, which is why the reduction tree used to be written out.
- **`@localmem` with an element type held in a kernel-local variable** fails at run time on
  POCL ("undefined variable"); write the type expression in the macro call.
