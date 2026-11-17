# For backend authors

```@meta
CurrentModule = ModflowGPU
DocTestSetup = :(using ModflowGPU)
```

ModflowGPU runs on any KernelAbstractions backend. Each GPU package is a weak dependency
with a package extension in `ext/` (for example `ext/AMDGPUExt.jl`), which Julia loads
when the user loads the GPU package. The extension registers its backend in `__init__`
and adds methods to the hooks below where the default does not fit its backend:

- [`ModflowGPU.register_backend!`](@ref) (required): makes the backend selectable by name;
- [`supports_float64`](@ref): `false` for devices without Float64 (the default is `true`);
- [`ModflowGPU.pin_memory!`](@ref): page-locked host memory for faster transfers;
- [`ModflowGPU.free_memory!`](@ref): release device memory immediately;
- [`ModflowGPU.host_mapped_zeros`](@ref): host memory that kernels can write directly.

None of these are exported; an extension calls them qualified.
[`ModflowGPU.free_device_arrays!`](@ref), which frees every device array of an object
through [`ModflowGPU.free_memory!`](@ref), is also useful to coupling code.

```@docs
ModflowGPU.register_backend!
ModflowGPU.pin_memory!
ModflowGPU.free_memory!
ModflowGPU.host_mapped_zeros
ModflowGPU.free_device_arrays!
```
