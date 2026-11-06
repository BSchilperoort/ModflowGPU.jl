# Compatibility: KernelAbstractions, AMDGPU and mGV

Checked 2026-11-06 for T0.1 on the development machine (Julia 1.12.6, RX 7800 XT,
ROCm 7.2.2 in `/opt/rocm`).

## Compat entry

ModflowGPU sets `KernelAbstractions = "0.9.41, 0.10"`. Both series resolve and pass the
test suite. Each case was resolved in a temporary environment with ModflowGPU
`Pkg.develop`ed and KernelAbstractions (KA) forced to the version shown:

| Environment | KA | AMDGPU | `test/runtests.jl` |
| --- | --- | --- | --- |
| `test/` workspace (default resolve) | 0.10.5 | 2.11.0 | CPU 42/42 pass; `MFGPU_TEST_BACKEND=rocm` 47/47 pass |
| KA forced to `0.9.41` (lower bound) | 0.9.41 | none | CPU 42/42 pass |
| KA forced to `0.9` | 0.9.45 | none | CPU 42/42 pass |
| KA forced to `0.9` + AMDGPU | 0.9.45 | 2.8.0 | `MFGPU_TEST_BACKEND=rocm` 47/47 pass |
| ModflowGPU + mGV + AMDGPU | 0.9.45 | 2.8.0 | both `select_backend("rocm")` return `ROCBackend()` |
| All five GPU packages + `pocl_jll` | 0.9.45 | 2.8.0 | all five extensions load (CUDA 6.3.1, Metal 1.10.3, oneAPI 2.9.0, OpenCL 0.10.11) |

With KA 0.9 the resolver picks AMDGPU 2.8.0 (the registry bounds AMDGPU 1.3.4–2.8 to KA
0.9 and AMDGPU ≥ 2.11 to KA 0.10; AMDGPU 2.9/2.10 did not resolve with KA 0.9 here). Both
combinations run ModflowGPU on the RX 7800 XT.

## mGV in the same environment (Milestone B)

mGV pins `KernelAbstractions = "0.9.41"`, which in Julia's caret semantics means
`[0.9.41, 0.10)`. Therefore:

- **mGV and ModflowGPU can share an environment today**, but only on KA 0.9: the resolver
  picks KA 0.9.45 and **AMDGPU 2.8.0** (not 2.10/2.11). Loading both and selecting `"rocm"`
  works.
- **mGV cannot share an environment with KA 0.10 / AMDGPU ≥ 2.10.** Forcing KA 0.10 with mGV
  fails: `Unsatisfiable requirements detected for package KernelAbstractions ... restricted
  to versions 0.9.41 - 0.9 by mGV`.
- To use KA 0.10 / AMDGPU ≥ 2.10 together with mGV, mGV's compat must become
  `KernelAbstractions = "0.9.41, 0.10"` (and its code must work on 0.10, see below). That is
  a change to mGV, outside this repository; until then Milestone B environments run on
  KA 0.9 + AMDGPU 2.8. ModflowGPU must therefore keep working on KA 0.9 (T5.4 tests the
  shared environment).

## KA 0.9 vs 0.10 differences that matter here

- **`CPU` is a different backend.** In KA 0.10, `CPU` is an alias for `POCLBackend`: CPU
  kernels are compiled to OpenCL/SPIR-V and run through POCL, not as Julia loops as in
  0.9. `backend_name(CPU())` is `"POCLBackend"` in 0.10 and `"CPU"` in 0.9; do not test for
  the string `"CPU"`. `KernelAbstractions.supports_float64(CPU())` is `true` on this machine.
  CPU performance numbers (T0.4) must state the KA version.
- **`KernelAbstractions.supports_float64` default flipped.** KA 0.9 defaults to `true`;
  KA 0.10 (via KernelInterface) defaults to `false` for backends that do not define it. So
  `ModflowGPU.supports_float64` has its own default of `true` (section 3.7 of PLAN.md) and
  does not forward to KA. Metal, oneAPI and OpenCL get explicit methods in the extensions.
- KA 0.10 adds a dependency on KernelInterface (host API: `supports_float64`, `allocate`,
  `synchronize`, ...). The kernel language (`@kernel`, `@index`, `@Const`) used so far is
  the same in both series.
