# Running a simulation: `ModflowGPU.run(simdir)` (the time loop of MF6's `Sim`, with
# `tdis_set_timestep`, `sln_ca` and `gwf_ot` per step), on the stepping API (src/stepping.jl).

"""
    RunResult

What `ModflowGPU.run` returns: `model` (the `GwfModel`, still on the device), the
precision types `T`, `Ts`, `Tp`, `steps` (`StepResult` per time step), `budgets`
(`ModelBudget` per time step), `heads` (absolute Float64 heads per reduced node at the end
of the run), `converged` (every step converged) and the output paths `head_file`,
`budget_file`, `listing_file` (`""` when not written).
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

A time step did not converge within OUTER_MAXIMUM and the simulation has no CONTINUE
option (MF6 stops the simulation after writing the step's output).
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
    ModflowGPU.run(simdir; backend = CPU(), precision = :auto, outdir = simdir,
                   closures = nothing, write_output = true, preconditioner = :auto,
                   precond_options = (;)) -> RunResult

Run the MF6 simulation in `simdir` (one GWF model with one IMS solution; exchanges arrive
in Phase 3) on `backend` in precision `precision` (`resolve_precision`: `:auto` =
`:mixed` on backends with Float64, else `:single`; `:double`, `:mixed`, `:single`). Per time step: stress period data once per period, the outer
iterations (`solve_timestep!`), flows and budget (`calc_flows!`, `model_budget!`), and the
output selected by OC: HDS and CBC files and the model listing file (budget tables at the
end of each period, with PRINT BUDGET, and after a failed step), written to `outdir` under
the names of the input (the listing file under its name-file name or `<model>.lst`).

`closures` overrides IMS closures (see `PicardSettings`); in `:single` closures below
`FLOAT32_CLOSURES` are raised. A step that does not converge throws a `ConvergenceError`
after its output, unless the simulation has the CONTINUE option.

`preconditioner` (a ModflowGPU option, not in MF6 input) selects the inner preconditioner
in place of the ILU0/MILU0/ILUT of IMS: `:auto` (the default, `DEFAULT_PRECONDITIONER` =
`:mc_ilu0`), `:jacobi`, `:chebyshev`, `:mc_sgs`, `:mc_ilu0` or `:amg` (`PRECONDITIONERS`). `precond_options` is a NamedTuple of keywords
for its constructor (see `build_preconditioner`; for `:amg` those of `AMGOptions`). See
`KrylovSettings` and docs/mf6_notes.md (IMS linear solver mapping).
"""
function run(
        simdir::AbstractString; backend::Backend = CPU(), precision::Symbol = :auto, outdir::AbstractString = simdir,
        closures = nothing, write_output::Bool = true, preconditioner::Symbol = :auto,
        precond_options::NamedTuple = (;),
    )
    sim = load_simulation(simdir)
    return run(sim; backend, precision, outdir, closures, write_output, preconditioner, precond_options)
end

function run(
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
        finalize(s; free = false)   # the model is returned (RunResult)
    end
    out = s.output
    paths = isnothing(out) ? ("", "", "") : (out.head_path, out.budget_path, out.listing_path)
    allconv = all(r -> r.outer.converged, s.results)
    return RunResult(s.model, s.T, s.Ts, s.Tp, s.results, s.budgets, absolute_heads(s.model), allconv, paths...)
end

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
