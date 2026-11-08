# Helpers shared by the test files. Expects `TEST_BACKEND` to be defined (see runtests.jl).

"""
Precision presets every test runs in. `:mixed` differs from `:double` only in the
preconditioner type `Tp`, so it runs only where `Tp` matters (`SOLVER_TEST_PRESETS`).
"""
const TEST_PRESETS = (:single, :double)

"""
Presets of the solver, preconditioner and end-to-end tests (groups krylov, amg, picard,
run, stepping, synthetic): `TEST_PRESETS` plus `:mixed` (T4.4).
"""
const SOLVER_TEST_PRESETS = (:single, :mixed, :double)

"""
    for_each_precision(f; backend = TEST_BACKEND, presets = TEST_PRESETS)

Run `f(p)` in a separate testset for each precision preset, where `p` is a `NamedTuple`
`(; preset, T, Ts, Tp)` (see `ModflowGPU.resolve_precision`). Presets that need Float64
are skipped with a message on backends without Float64.

```julia
for_each_precision() do p
    x = KernelAbstractions.zeros(TEST_BACKEND, p.T, 10)
    @test eltype(x) == p.T
end
```
"""
function for_each_precision(f; backend = TEST_BACKEND, presets = TEST_PRESETS)
    for preset in presets
        if preset !== :single && !ModflowGPU.supports_float64(backend)
            @info "Skipping precision = :$preset: backend $(ModflowGPU.backend_name(backend)) has no Float64"
            continue
        end
        types = ModflowGPU.resolve_precision(backend, preset)
        @testset "precision = :$preset" begin
            f((; preset, types...))
        end
    end
    return nothing
end

"Write `text` to `name` in `dir` and return the path."
function write_input(dir, name, text)
    path = joinpath(dir, name)
    mkpath(dirname(path))
    write(path, text)
    return path
end

"The `ModflowGPU.MF6InputError` thrown by `f()` (test fails if there is none)."
function input_error_of(f)
    try
        f()
    catch e
        e isa ModflowGPU.MF6InputError && return e
        rethrow()
    end
    @test false
    return nothing
end
