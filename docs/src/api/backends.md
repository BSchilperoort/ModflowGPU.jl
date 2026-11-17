# Backends and precision

```@meta
CurrentModule = ModflowGPU
DocTestSetup = :(using ModflowGPU)
```

A backend is the device a simulation runs on, as a KernelAbstractions `Backend`. GPU
backends become available when their package is loaded next to ModflowGPU (for example
`using AMDGPU, ModflowGPU`); [`select_backend`](@ref) picks one by name. The precision
preset sets the floating-point types of the model state, the linear solver and the
preconditioner; [`resolve_precision`](@ref) shows what a preset means on a backend.

```jldoctest
julia> backend = select_backend("cpu");

julia> resolve_precision(backend, :auto)
(T = Float64, Ts = Float64, Tp = Float32)
```

```@docs
select_backend
backend_options
backend_name
supports_float64
resolve_precision
ModflowGPU.PRECISION_PRESETS
```
