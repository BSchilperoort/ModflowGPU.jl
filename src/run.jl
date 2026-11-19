# Running a simulation: `run_simulation(simdir)`, alias `ModflowGPU.run` (the time loop of MF6's `Sim`, with
# `tdis_set_timestep`, `sln_ca` and `gwf_ot` per step), on the stepping API (src/stepping.jl).

"""
    RunResult

What [`run_simulation`](@ref) returns. Fields:

- `converged::Bool`: every time step converged;
- `heads::Vector{Float64}`: the absolute heads at the end of the run, per model node
  (cells with IDOMAIN > 0, in MF6's reduced node order), in the model's length units;
- `steps::Vector{StepResult}`: per time step its [`StepResult`](@ref) (time, iterations);
- `budgets::Vector{ModelBudget}`: per time step the volumetric budget
  ([`ModelBudget`](@ref));
- `head_file`, `budget_file`, `listing_file`: the paths of the output files (`""` when
  not written);
- `T`, `Ts`, `Tp`: the precision types used (see [`resolve_precision`](@ref));
- `model`: the model, still on the device (internal; its memory is released by the
  garbage collector).

# Example
```julia
result = run_simulation("path/to/simulation")
result.converged
result.budgets[end].terms   # budget terms of the last time step
```
"""
struct RunResult{M <: GwfModel}
    model::M
    T::Type
    Ts::Type
    Tp::Type
    steps::Vector{StepResult}
    budgets::Vector{ModelBudget}
    heads::Vector{Float64}
    converged::Bool
    head_file::String
    budget_file::String
    listing_file::String
end

"""
    ConvergenceError(kper, kstp, iterations)

Thrown by [`run_simulation`](@ref) (and `BasicModelInterface.update`) when time step
`kstp` of stress period `kper` did not converge within OUTER_MAXIMUM (`iterations` outer
iterations) and the simulation has no CONTINUE option. As MF6, the run stops after writing
the output of that step.

# Example
```julia
try
    run_simulation("path/to/simulation")
catch e
    e isa ConvergenceError || rethrow()
    @warn "stopped in stress period \$(e.kper), time step \$(e.kstp)"
end
```
"""
struct ConvergenceError <: Exception
    kper::Int
    kstp::Int
    iterations::Int
end

Base.showerror(io::IO, e::ConvergenceError) = print(
    io, "ConvergenceError: the solution did not converge in time step ", e.kstp, " of stress period ", e.kper,
    " (", e.iterations, " outer iterations); add CONTINUE to the mfsim.nam OPTIONS to continue",
)

"""
    run_simulation(simdir::AbstractString; kwargs...) -> RunResult
    run_simulation(spec::SimulationSpec; kwargs...) -> RunResult

Run the MF6 simulation in directory `simdir` (the directory with `mfsim.nam`), or a
loaded or built [`SimulationSpec`](@ref), from start to end. The simulation has one GWF
model and one IMS solution. Each time step applies the stress-period data, solves the
heads with the outer (Picard) iterations, computes flows and the budget, and writes the
output that the OC package asks for: the head file, the budget file and the model listing
file (budget tables), under the names given in the input.

Keywords:
- `backend = CPU()`: the KernelAbstractions backend, e.g. from [`select_backend`](@ref);
- `precision = :auto`: a precision preset, see [`resolve_precision`](@ref) (`:auto` is
  `:mixed` on backends with Float64, else `:single`);
- `outdir = simdir`: the directory for the output files (`spec.simdir` for a spec); a
  directory inside the one named by the environment variable `REFERENCE_DIR` (the MF6
  reference runs of the parity tests) is an `ArgumentError`;
- `write_output = true`: `false` writes no files;
- `closures = nothing`: a `NamedTuple` that overrides IMS settings, with any of
  `outer_dvclose`, `outer_maximum`, `inner_dvclose`, `inner_rclose`, `inner_maximum`
  (closures in the model's units); in `:single`, closures below what Float32 can reach
  are raised (logged);
- `preconditioner = :auto`: the linear-solver preconditioner, replacing the ILU0, MILU0
  or ILUT of IMS (a ModflowGPU option, not in MF6 input): `:auto` (`:mc_ilu0`),
  `:jacobi`, `:chebyshev`, `:mc_sgs`, `:mc_ilu0` or `:amg`;
- `precond_options = (;)`: keywords for the preconditioner: `degree` for `:chebyshev`,
  `relax` for `:mc_sgs` and `:mc_ilu0`, the AMG options for `:amg` (developer notes,
  docs/dev/mf6_notes.md).

Returns a [`RunResult`](@ref) with the heads, the budgets and the iteration counts. A
time step that does not converge throws a [`ConvergenceError`](@ref) after writing its
output, unless the simulation has the CONTINUE option. Invalid or unsupported input
throws an [`ModflowGPU.MF6InputError`](@ref) or an [`MF6UnsupportedError`](@ref) before
anything runs.

[`ModflowGPU.run`](@ref) is the same function under its earlier name.

# Example
```julia
using ModflowGPU
backend = select_backend("auto")
result = run_simulation("path/to/simulation"; backend)
result.converged

# on a GPU without Float64, or to save memory:
result = run_simulation("path/to/simulation"; backend, precision = :single)
```
"""
function run_simulation(
        simdir::AbstractString; backend::Backend = CPU(), precision::Symbol = :auto, outdir::AbstractString = simdir,
        closures = nothing, write_output::Bool = true, preconditioner::Symbol = :auto,
        precond_options::NamedTuple = (;),
    )
    sim = load_simulation(simdir)
    return run_simulation(sim; backend, precision, outdir, closures, write_output, preconditioner, precond_options)
end

function run_simulation(
        sim::SimulationSpec; backend::Backend = CPU(), precision::Symbol = :auto, outdir::AbstractString = sim.simdir,
        closures = nothing, write_output::Bool = true, preconditioner::Symbol = :auto,
        precond_options::NamedTuple = (;),
    )
    s = initialize(sim; backend, precision, outdir, closures, write_output, preconditioner, precond_options)
    try
        while s.istep < length(s.steps)
            update(s) || (r = last(s.results); throw(ConvergenceError(r.step.kper, r.step.kstp, r.outer.iterations)))
        end
    finally
        finalize_simulation(s; free = false)   # the model is returned (RunResult)
    end
    out = s.output
    paths = isnothing(out) ? ("", "", "") : (out.head_path, out.budget_path, out.listing_path)
    allconv = all(r -> r.outer.converged, s.results)
    return RunResult(s.model, s.T, s.Ts, s.Tp, s.results, s.budgets, absolute_heads(s.model), allconv, paths...)
end

"""
    ModflowGPU.run(simdir; kwargs...) -> RunResult

The same function as [`run_simulation`](@ref), under the name used before it was
exported. Not exported, because it would clash with `Base.run`; it is kept (not
deprecated) for code that calls `ModflowGPU.run`.
"""
const run = run_simulation

function _listing_header(io::IO, m::GwfModel, ps::PicardSolver, T, Ts, Tp, backend)
    st = ps.settings
    println(io, "ModflowGPU listing file (MODFLOW 6 groundwater flow on KernelAbstractions)")
    println(io, " MODEL: ", m.spec.name, ", ", m.conn.nodes, " active cells, ", m.conn.nja, " connections")
    println(io, " BACKEND: ", backend_name(backend), ", PRECISION: T = ", T, ", Ts = ", Ts, ", Tp = ", Tp)
    k = st.krylov
    @printf(io, " OUTER_DVCLOSE = %.4e, OUTER_MAXIMUM = %d, UNDER_RELAXATION = %s\n", st.dvclose, st.maxiter, uppercase(String(st.ims.under_relaxation)))
    @printf(io, " LINEAR_ACCELERATION = %s (%s), INNER_MAXIMUM = %d, INNER_DVCLOSE = %.4e, INNER_RCLOSE = %.4e (option %d)\n", uppercase(String(k.method)), k.preconditioner === :jacobi ? "Jacobi" : String(k.preconditioner), k.maxiter, k.dvclose, k.rclose, k.icnvgopt)
    return nothing
end

function _write_step(out::ModelOutput, m::GwfModel, step::TimeStep, r::OuterResult, b::ModelBudget, sim::SimulationSpec)
    io = out.listing
    oc = m.spec.oc
    @printf(io, "\n %d CALLS TO NUMERICAL SOLUTION IN TIME STEP %d STRESS PERIOD %d\n %d TOTAL ITERATIONS\n", r.iterations, step.kstp, step.kper, r.inner)
    r.converged || @printf(io, "\n Solution did not converge for stress period %d and time step %d\n", step.kper, step.kstp)
    oc_selected(oc, :save, "BUDGET", step) && write_budget(out, m, step)
    # Heads are copied from the device only when OC saves or prints them (T5.2: a step
    # with budget output only moves no array between device and host).
    save_head, print_head = oc_selected(oc, :save, "HEAD", step), oc_selected(oc, :print, "HEAD", step)
    heads = (save_head || print_head) ? absolute_heads(m) : nothing
    save_head && write_heads(out, m, step, heads)
    print_head && write_head_listing(io, m, step, heads)
    # set_print_flag: also at the end of a period and after a failed step (without CONTINUE).
    if oc_selected(oc, :print, "BUDGET", step) || step.endofperiod || (!r.converged && !sim.continue_on_nonconvergence)
        terms = [(t.name, t.package, t.cum_in, t.cum_out, t.rate_in, t.rate_out) for t in b.terms]
        write_listing_budget(io, step.kstp, step.kper, terms)
    end
    r.converged || @printf(io, "\n         ****FAILED TO MEET SOLVER CONVERGENCE CRITERIA IN TIME STEP %d OF STRESS PERIOD %d****\n", step.kstp, step.kper)
    @printf(io, "\n TIME STEP LENGTH = %.6g, STRESS PERIOD TIME = %.6g, TOTAL SIMULATION TIME = %.6g\n", step.delt, step.pertim, step.totim)
    flush(io)
    return nothing
end
