# Optional timing instrumentation of the hot loop (T4.1).
#
# The engine marks its phases (assembly, SpMV, preconditioner setup, Krylov updates and
# reductions, host transfers, ...) with `@timed_section name expr`. With no section timer
# installed (the default) a section costs one load of a global `Ref` and a branch: no
# synchronisation, no closure, no allocation. A benchmark installs a timer with
# `set_section_timer!(t)`; the engine then calls `section_begin(t, name)` before and
# `section_end(t, name)` after the expression. The timer decides what to do there (the
# benchmark harness, benchmark/run.jl, synchronises the device and records the time in a
# TimerOutputs tree). The package itself depends on no timing library.

const _SECTION_TIMER = Ref{Any}(nothing)

"""
    set_section_timer!(t) -> previous
    set_section_timer!(nothing)

Install `t` as the section timer of `@timed_section` (or remove it with `nothing`). `t`
must have methods of `ModflowGPU.section_begin(t, name::Symbol)` and
`ModflowGPU.section_end(t, name::Symbol)`; sections nest, and `section_end` is called with
the name of the innermost open section. Returns the previous timer. Global state: one
timer for all simulations of the process (a benchmark tool, not a per-model setting).
"""
function set_section_timer!(t)
    prev = _SECTION_TIMER[]
    _SECTION_TIMER[] = t
    return prev
end

"""
    section_timer() -> timer or nothing

The installed section timer (`set_section_timer!`), `nothing` when timing is off.
"""
section_timer() = _SECTION_TIMER[]

"""
    section_begin(t, name::Symbol)
    section_end(t, name::Symbol)

Hooks of a section timer `t` (`set_section_timer!`), called around each
`@timed_section name expr`. The package defines no methods; a timer type adds them.
"""
function section_begin end

@doc (@doc section_begin) function section_end end

"""
    @timed_section name expr

Evaluate `expr` (and return its value) as the section `name` (a `Symbol`) of the installed
section timer (`set_section_timer!`). Without a timer this is `expr` plus one branch. The
expression is not wrapped in a closure, so variables it assigns stay in the enclosing
scope and the disabled path keeps its types. An exception inside `expr` leaves the section
open (benchmark use only).
"""
macro timed_section(name, ex)
    return quote
        local _st = _SECTION_TIMER[]
        _st === nothing || section_begin(_st, $(esc(name)))
        local _sv = $(esc(ex))
        _st === nothing || section_end(_st, $(esc(name)))
        _sv
    end
end
