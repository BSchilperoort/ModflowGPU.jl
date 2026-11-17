# Installation

```@meta
CurrentModule = ModflowGPU
DocTestSetup = :(using ModflowGPU)
```

ModflowGPU is a Julia package. You need Julia, the package itself and, to run on a GPU,
the Julia package for your GPU. Without a GPU package everything runs on the CPU, which is
enough to try the package and to follow [Getting started](getting_started.md).

## Julia

ModflowGPU needs **Julia 1.12 or newer**. The easiest way to install and update Julia is
[juliaup](https://github.com/JuliaLang/juliaup), the official installer; the
[Julia download page](https://julialang.org/install/) has the one-line install command for
your operating system. Check the version in a terminal:

```sh
julia --version   # julia version 1.12.x or newer
```

If it shows an older version, run `juliaup update` (or `juliaup add release` followed by
`juliaup default release`).

ModflowGPU is developed and tested on Linux. It contains no Linux-specific code, but it has
not been tested on Windows or macOS.

## The package

ModflowGPU is not yet in Julia's General registry, so you add it by its Git URL. Use a
project environment of your own, so that the packages of different projects do not get in
each other's way. In a terminal, in the folder where you keep your work:

```sh
julia --project=.
```

and then in Julia:

```julia
using Pkg
Pkg.add(url = "https://github.com/BSchilperoort/ModflowGPU.jl")
```

This downloads ModflowGPU and its dependencies and writes `Project.toml` and
`Manifest.toml` in the folder. The first time takes one to a few minutes, because Julia also
precompiles the packages. Start Julia with `julia --project=.` in that folder from then on.

To get a newer version of ModflowGPU later, run `Pkg.update("ModflowGPU")`.

Check that it works:

```jldoctest
julia> using ModflowGPU

julia> backend = select_backend("cpu");

julia> backend_name(backend)
"POCLBackend"
```

The CPU backend is [POCL](https://portablecl.org), a portable OpenCL implementation that
is installed with ModflowGPU's dependencies; you do not need to install anything else for
it. It uses as many threads as Julia was started with: start Julia with
`julia --project=. --threads=auto` to use all cores.

## A GPU

Each GPU vendor has its own Julia package. ModflowGPU supports them as optional
*extensions*: you add the GPU package to the same environment and load it next to
ModflowGPU, and ModflowGPU then finds the GPU. Pick the section for your GPU below.

!!! warning "Only AMD GPUs are tested"
    ModflowGPU is developed and tested on an AMD GPU (Radeon RX 7800 XT, ROCm 7.2) and on
    the CPU. For NVIDIA (CUDA), Intel (oneAPI), Apple (Metal) and other OpenCL devices,
    the extension is written and loads together with the other GPU packages, but no
    model has been run on such a GPU yet. Please report what you find in an
    [issue](https://github.com/BSchilperoort/ModflowGPU.jl/issues).

| GPU | Julia package | Backend name | Float64 | Tested on hardware |
| --- | --- | --- | --- | --- |
| AMD (Radeon, Instinct) | `AMDGPU` (2.11 or newer) | `"rocm"` | yes | yes |
| NVIDIA | `CUDA` (6.5 or newer) | `"cuda"` | yes | no |
| Intel (Arc, Data Center GPU) | `oneAPI` (2.10 or newer) | `"oneapi"` | depends on the GPU | no |
| Apple (M-series) | `Metal` (1.12 or newer) | `"metal"` | no | no |
| Other OpenCL devices | `OpenCL` (1 or newer) | `"opencl"` | depends on the device | no |

The Float64 column matters for the [precision presets](@ref "Precision presets"): on a GPU
without Float64, ModflowGPU runs in single precision (`:single`), which is less accurate.

### AMD: AMDGPU.jl

AMDGPU.jl uses the ROCm libraries installed on your system. Install ROCm first, following
[AMD's ROCm installation guide](https://rocm.docs.amd.com) for Linux, and check that
`rocminfo` lists your GPU. ROCm is normally installed in `/opt/rocm`, where AMDGPU.jl finds
it. Then add the package:

```julia
Pkg.add("AMDGPU")
```

Check that Julia sees the GPU (needs an AMD GPU):

```julia
using AMDGPU
AMDGPU.functional()    # true
AMDGPU.versioninfo()   # ROCm version and the GPUs found
```

Float64 is available on AMD GPUs.

### NVIDIA: CUDA.jl

CUDA.jl needs only the NVIDIA driver; it downloads the matching CUDA toolkit itself. Check
that `nvidia-smi` lists your GPU, then add the package:

```julia
Pkg.add("CUDA")
```

Check (needs an NVIDIA GPU):

```julia
using CUDA
CUDA.functional()    # true
CUDA.versioninfo()   # driver, toolkit and the GPUs found
```

Float64 is available on NVIDIA GPUs. On many consumer cards it is much slower than
Float32; this has not been measured with ModflowGPU.

### Intel: oneAPI.jl

oneAPI.jl needs the Intel GPU driver with Level Zero support (Linux). Add the package:

```julia
Pkg.add("oneAPI")
```

Check (needs an Intel GPU):

```julia
using oneAPI
oneAPI.functional()    # true
oneAPI.versioninfo()
```

Some Intel GPUs, for example the Arc series, have no Float64. ModflowGPU asks the device
and then runs in single precision (see [`supports_float64`](@ref)).

### Apple: Metal.jl

Metal.jl works on Macs with Apple silicon (M-series). Add the package:

```julia
Pkg.add("Metal")
```

Check (needs an Apple GPU):

```julia
using Metal
Metal.functional()    # true
Metal.versioninfo()
```

Apple GPUs have no Float64, so ModflowGPU always runs in single precision on them.

### Other OpenCL devices: OpenCL.jl

OpenCL.jl runs on any device with an OpenCL driver that can run SPIR-V kernels. Install
the driver of your device, then add the package:

```julia
Pkg.add("OpenCL")
```

Float64 depends on the device (the `cl_khr_fp64` extension); ModflowGPU asks the device.

## Check that the GPU is found

Load the GPU package before or together with ModflowGPU, then ask ModflowGPU which backend
it picks:

```julia
using AMDGPU, ModflowGPU         # or CUDA, oneAPI, Metal, OpenCL; needs a GPU

backend = select_backend("auto") # the first GPU that works, else the CPU
backend_name(backend)            # "ROCBackend" on an AMD GPU, "POCLBackend" for the CPU
supports_float64(backend)        # true on AMD and NVIDIA GPUs
resolve_precision(backend)       # the precision `:auto` gives on this backend
```

[`backend_options`](@ref) lists every name [`select_backend`](@ref) accepts, also for GPU
packages that are not loaded:

```jldoctest
julia> backend_options()
7-element Vector{String}:
 "auto"
 "cpu"
 "cuda"
 "rocm"
 "metal"
 "oneapi"
 "opencl"
```

`select_backend("auto")` never fails: without a working GPU it returns the CPU backend.
Ask for a GPU by name to get an error that says what is missing instead:

```jldoctest
julia> select_backend("rocm")
ERROR: Backend 'rocm' requires AMDGPU.jl. Add it to your environment and load it alongside ModflowGPU, e.g. `using AMDGPU, ModflowGPU`.
[...]
```

If the package is loaded but cannot use the GPU (no driver, no device), the error says
`AMDGPU is not functional on this machine`; check the package's own `functional()` and
`versioninfo()` as above.

## Next steps

- [Getting started](getting_started.md): run a first model and read its results.
- [GPU guide](gpu.md): precision, memory use and troubleshooting.
