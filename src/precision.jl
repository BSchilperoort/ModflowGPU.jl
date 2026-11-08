# Precision presets (PLAN.md section 3.7). Engine structs are parametric in `T` (state and
# assembly), `Ts` (Krylov) and `Tp` (preconditioner); a preset fixes all three.

"""
    supports_float64(backend)::Bool

Whether `backend` can run kernels in Float64. Extension hook like [`pin_memory!`](@ref):
the default is `true` (CPU, CUDA, ROCm). The Metal extension returns `false`, and the oneAPI
and OpenCL extensions query the device.

This is deliberately not `KernelAbstractions.supports_float64`: its default is `true` in
KernelAbstractions 0.9 but `false` in 0.10, and ModflowGPU supports both.
"""
supports_float64(::Backend) = true

"""
Precision presets accepted by [`resolve_precision`](@ref).
"""
const PRECISION_PRESETS = (:auto, :double, :mixed, :single)

"""
    resolve_precision(backend, preset = :auto) -> (; T, Ts, Tp)

Resolve a precision preset to the floating-point types for state and assembly (`T`), the
Krylov solver (`Ts`) and the preconditioner (`Tp`):

| Preset    | `T`     | `Ts`    | `Tp`    |
| --------- | ------- | ------- | ------- |
| `:double` | Float64 | Float64 | Float64 |
| `:mixed`  | Float64 | Float64 | Float32 |
| `:single` | Float32 | Float32 | Float32 |
| `:auto`   | `:mixed` if `supports_float64(backend)`, else `:single` |

`:mixed` keeps the state, assembly and Krylov vectors in Float64 (the `:double` parity
tolerances hold) and stores every preconditioner's data (Jacobi `minv`, Chebyshev and
multicolour operator copies and factors, the AMG hierarchy) in Float32. It is the
`:auto` default on backends with Float64 because it is as accurate as `:double` and 10–20 %
faster with less memory (measured in docs/precision.md, T4.4).

The result is a `NamedTuple`, so `T, Ts, Tp = resolve_precision(backend, preset)` works.
Requesting `:double` or `:mixed` on a backend without Float64 support throws an error.
"""
function resolve_precision(backend::Backend, preset::Symbol = :auto)
    if preset === :auto
        preset = supports_float64(backend) ? :mixed : :single
    end
    if preset === :double
        supports_float64(backend) || error(
            "precision = :double was requested, but backend $(backend_name(backend)) does " *
                "not support Float64. Use precision = :single instead.",
        )
        return (; T = Float64, Ts = Float64, Tp = Float64)
    elseif preset === :mixed
        supports_float64(backend) || error(
            "precision = :mixed was requested, but backend $(backend_name(backend)) does " *
                "not support Float64. Use precision = :single instead.",
        )
        return (; T = Float64, Ts = Float64, Tp = Float32)
    elseif preset === :single
        return (; T = Float32, Ts = Float32, Tp = Float32)
    else
        error(
            "Unknown precision preset :$preset. Valid presets are: " *
                "$(join(repr.(PRECISION_PRESETS), ", ")).",
        )
    end
end

resolve_precision(backend::Backend, preset::AbstractString) = resolve_precision(backend, Symbol(lowercase(preset)))
