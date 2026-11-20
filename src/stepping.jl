# Stepping API (PLAN.md section 3.6): the time loop of `run_simulation` cut into the
# libmf6/XMI calls (srcbmi/mf6xmi.F90, mf6core.f90), with libmf6 memory addresses.
#
#   initialize               load, build the model on the device (`Mf6Initialize`)
#   prepare_time_step(dt)    next TDIS step; period data at the first step of a period
#                            (`Mf6PrepareTimestep`: `tdis_set_timestep`, `model_rp`)
#   prepare_solve            start the step: storage `rho`, `hold ← h`, CHD heads
#                            (`prepareSolve`: `model_ad`)
#   solve -> converged       one outer iteration (`xmi_solve`: `sln_ca` loop body)
#   finalize_solve           flows and budget (`finalizeSolve`: `model_cq`, `model_bd`)
#   finalize_time_step       output (`Mf6FinalizeTimestep`: `model_ot`)
#   do_time_step, update     the steps above in one call; finalize_simulation
#
# Values. Addresses are libmf6's, e.g. "RDE/X", "RDE/DIS/NODEUSER", "RDE/RIV/BOUND",
# "RDE/DRN/SIMVALS", "SLN_1/MXITER", "TDIS/DELT" (`get_var_address`). `get_value_ptr`
# returns a host mirror in Float64 with absolute heads (head-like BOUND columns and X are
# absolute, BOUND before the MULTIPLIER aux variable, as given in the input). The engine
# keeps offsets in precision T on the device (PLAN.md 3.7, rule 1), so the mirror is a
# copy, synchronised at the call boundaries:
#
#   - writes into a writable mirror (X, BOUND) are sent to the device at the start of
#     `prepare_time_step` and of `prepare_solve` (when the mirror differs from what was
#     last synchronised), through the same conversion as period data (`set_bound!`);
#   - a mirror is filled when `get_value_ptr` creates it, and refreshed from the engine
#     at the end of `prepare_time_step`, `prepare_solve` and `finalize_solve` (so X during the outer
#     iterations is the X at the start of the step; read it with `get_value`). BOUND and
#     NODELIST mirrors are versioned (T4.6): they are copied only when the engine value
#     changed (new period data, a write) or, in `finalize_solve`, when the mirror differs.
#   - writes into a mirror between `prepare_solve` and `finalize_solve` are discarded by
#     the refresh in `finalize_solve`, with a warning (use `set_value` there).
#
# `get_value_ptr(sim, address; device = true)` returns the engine arrays themselves
# (device, precision T, head-like values as offsets from "<model>/HREF", BOUND with the
# multiplier applied); writes there bypass the conversions and the mirrors. Device writes
# into BOUND (T5.2: ModflowCoupling's device-resident coupling writes them in kernels) go
# through `device_bound!` (T5.4), which opens the array for writing between
# `prepare_time_step` and `prepare_solve` only; `prepare_solve` checks the written records
# on the device (finite values, `riv_ck`). They are replaced by new period data, and are
# not seen by host reads of BOUND (`get_value`, mirrors), which keep the last values set on
# the host. A host mirror of the same BOUND does not overwrite them unless it is written.
# `finalize_simulation` frees the device arrays (T5.4: two models on one GPU). A caller that
# keeps no host mirrors (reads through `get_value` only) makes a time step move no array
# between host and device: the mirrors are the only per-step copies of the stepping API,
# and the output of `finalize_time_step` copies the heads only when OC saves or prints them.

"""
    StepResult

The result of one time step, in `RunResult.steps` and `Simulation.results`.

Fields:
- `step`: the TDIS time step, with `kper`, `kstp` (stress period and time step numbers),
  `delt` (step length), `pertim` and `totim` (time in the period and in the simulation at
  the end of the step), in the model's time units;
- `outer`: the outer-iteration result, with `converged::Bool`, `iterations` (outer
  iterations) and `inner` (total inner, linear-solver, iterations).

# Example
```julia
result = run_simulation("path/to/simulation")
[(r.step.kper, r.step.kstp, r.outer.iterations) for r in result.steps]
```
"""
struct StepResult
    step::TimeStep
    outer::OuterResult
end

# A variable reachable by address: `get(sim)` returns a new host array (Float64 values
# absolute; Int32 indices), `set(sim, values)` writes the engine (nothing: read-only),
# `device(sim)` the engine array (nothing: host only). Versioned variables (T4.6: BOUND,
# NODELIST) also have `version(sim)`, a counter that changes whenever the engine value
# changes, and `source(sim)`, the engine's host array itself (equal to `get(sim)`, not
# copied).
struct _Variable
    get::Function
    set::Union{Nothing, Function}
    device::Union{Nothing, Function}
    units::String
    version::Union{Nothing, Function}
    source::Union{Nothing, Function}
end
_Variable(get, set, device, units) = _Variable(get, set, device, units, nothing, nothing)

# A host mirror (`get_value_ptr`), its value at the last synchronisation (`synced`; empty
# for a versioned variable, whose engine array `source` is that value) and the engine
# version at the last synchronisation.
struct _Mirror
    var::_Variable
    host::Array
    synced::Array
    version::Base.RefValue{Int}
end
function _Mirror(s, var::_Variable)
    h = var.get(s)
    isnothing(var.version) && return _Mirror(var, h, copy(h), Ref(0))
    return _Mirror(var, h, similar(h, 0), Ref(var.version(s)))
end

"""
    Simulation

A simulation driven one time step at a time, created by [`initialize`](@ref) and used by
the other functions of the stepping API. It is also the model type of the BMI methods
(`BasicModelInterface.initialize(Simulation, dir)`).

Fields that are useful to read (do not change them):
- `spec`: the [`SimulationSpec`](@ref) it was built from;
- `T`, `Ts`, `Tp`: the precision types (see [`resolve_precision`](@ref));
- `steps`: the TDIS time steps, and `istep`, the index of the current one (0 before the
  first [`prepare_time_step`](@ref));
- `results` and `budgets`: per completed time step its [`StepResult`](@ref) and
  [`ModelBudget`](@ref) (in a multi-model solution one per model and step, in the order of
  the solution's models; `ModelBudget.model` names the model);
- `models`: the names of the models, in the solution's order (T6.6).

The other fields (the model on the device, the solver, the output files, the call stage)
are internal.

# Example
```julia
sim = initialize("path/to/simulation")
sim.istep, length(sim.steps)   # (0, number of time steps)
```
"""
mutable struct Simulation{M <: GwfModel, P <: PicardSolver, A <: BudgetAccumulator}
    const spec::SimulationSpec
    const model::M
    const solver::P
    const acc::A
    const output::Union{Nothing, ModelOutput}
    const T::Type
    const Ts::Type
    const Tp::Type
    const steps::Vector{TimeStep}
    istep::Int
    stage::Symbol
    outer::OuterState
    const results::Vector{StepResult}
    const budgets::Vector{ModelBudget}
    const mirrors::Dict{String, _Mirror}
    # T5.4: packages whose device BOUND was opened for writing in this step
    # (`device_bound!`), checked at `prepare_solve`; the check's result buffer (lazy).
    const device_writes::Vector{Int}
    check::Any
    # T6.6: the models (`model_views`: the model itself, or the `ModelView`s of a
    # multi-model solution), their names, outputs and budget accumulators, and the
    # exchange flows (`ExchangeFlows`, `nothing` for a single model). `output` and `acc`
    # are those of the first model.
    const views::Vector{Any}
    const models::Vector{String}
    const outputs::Vector{ModelOutput}
    const accs::Vector{BudgetAccumulator}
    const exflows::Any
    # T6.6: the observation output of the exchanges (`ExchangeObservations`), or `nothing`.
    const exobs::Any
    # T6.9: without `write_output`, the observation output of each model (`ObservationOutput`
    # or `nothing`): MF6 writes OBS6 output in every run.
    const obsonly::Vector{Any}
end

Base.show(io::IO, s::Simulation) = print(
    io, "Simulation(\"", join(s.models, "\", \""), "\", T = ", s.T, ", step ", s.istep, " of ", length(s.steps), ", ", s.stage, ")",
)

"""
    initialize(simdir::AbstractString; kwargs...) -> Simulation
    initialize(spec::SimulationSpec; kwargs...) -> Simulation

Start a simulation to be run one time step at a time: load the MF6 simulation in directory
`simdir` (or take a loaded or built `spec`; GWF models in one IMS solution, connected by
GWF6-GWF6 exchanges when there are several), build the model, the outer solver and the
budget accumulator on the device, and open the output files (libmf6's `initialize`). The
models of a multi-model solution are solved as one system (T6.6); each writes its own
output files, and its variables have its own addresses (`"<model>/X"`, ...).

The keywords are those of [`run_simulation`](@ref): `backend`, `precision`, `outdir`,
`closures`, `write_output`, `preconditioner` and `precond_options`.

Before the first [`prepare_time_step`](@ref) (or [`ModflowGPU.update`](@ref)) the heads are the
starting heads and no period data is applied. End with [`finalize_simulation`](@ref),
which closes the output files and frees the device memory.

Throws an [`ModflowGPU.MF6InputError`](@ref) or [`MF6UnsupportedError`](@ref) for invalid
or unsupported input (also for a simulation with more than one solution, or models with
convertible cells or NEWTON in a multi-model solution).

# Example
```julia
sim = initialize("path/to/simulation"; backend = select_backend("auto"))
while get_current_time(sim) < get_end_time(sim)
    ModflowGPU.update(sim) || error("time step did not converge")
end
finalize_simulation(sim)
```
"""
function initialize(
        simdir::AbstractString; backend::Backend = CPU(), precision::Symbol = :auto, outdir::AbstractString = simdir,
        closures = nothing, write_output::Bool = true, preconditioner::Symbol = :auto,
        precond_options::NamedTuple = (;),
    )
    return initialize(load_simulation(simdir); backend, precision, outdir, closures, write_output, preconditioner, precond_options)
end

function initialize(
        sim::SimulationSpec; backend::Backend = CPU(), precision::Symbol = :auto, outdir::AbstractString = sim.simdir,
        closures = nothing, write_output::Bool = true, preconditioner::Symbol = :auto,
        precond_options::NamedTuple = (;),
    )
    (length(sim.solution_groups) == 1 && length(only(sim.solution_groups).solutions) == 1) ||
        throw(MF6UnsupportedError(sim.simdir, "unsupported: more than one solution (the models must be in one IMS solution)"))
    multi = length(sim.models) > 1 || !isempty(sim.exchanges)
    ims = only(only(sim.solution_groups).solutions)
    write_output && _check_output_dir(outdir)
    # NumericalSolution.f90 `sln_ar`: the Newton formulation (T6.2) and XT3D (T6.5, `gwf_get_iasym`;
    # T6.6: also XT3D on an exchange, `gwf_gwf_connection` of the interface model) give a
    # nonsymmetric matrix.
    for spec in sim.models
        (spec.newton || spec.npf.xt3d) && ims.linear_acceleration === :cg && throw(
            MF6InputError(
                ims.file, 0,
                "MODEL **$(uppercase(spec.name))** PRODUCES AN ASYMMETRIC COEFFICIENT MATRIX, BUT THE CONJUGATE GRADIENT METHOD WAS SELECTED. USE BICGSTAB INSTEAD.",
            ),
        )
    end
    for ex in sim.exchanges
        ex.xt3d && ims.linear_acceleration === :cg && throw(
            MF6InputError(
                ims.file, 0,
                "MODEL **$(ex.model1)** PRODUCES AN ASYMMETRIC COEFFICIENT MATRIX, BUT THE CONJUGATE GRADIENT METHOD WAS SELECTED. USE BICGSTAB INSTEAD.",
            ),
        )
    end
    T, Ts, Tp = resolve_precision(backend, precision)
    steps = time_steps(sim.tdis)
    m = multi ? GwfModel(backend, T, solution_setup(sim), nper(sim.tdis)) : GwfModel(backend, T, only(sim.models), nper(sim.tdis))
    ps = PicardSolver(m, PicardSettings(ims, T; closures, preconditioner, precond_options); Ts, Tp)
    views = model_views(m)
    accs = BudgetAccumulator[BudgetAccumulator(v) for v in views]
    outs = ModelOutput[]
    try
        write_output && for v in views
            push!(outs, ModelOutput(v, sim.simdir, outdir))
        end
    catch
        foreach(close, outs)
        rethrow()
    end
    out = isempty(outs) ? nothing : first(outs)
    exflows = multi ? ExchangeFlows(m) : nothing
    # T6.9: observations are written also without `write_output` (MF6 writes OBS6 output in
    # every run, e.g. through libmf6), into `outdir`.
    hasobs = any(v -> _has_obs(_vspec(v)), views) || (multi && any(ex -> !isnothing(ex.spec.obs), m.solution.exchanges))
    !write_output && hasobs && _check_output_dir(outdir)
    obsonly = Any[]
    try
        write_output || for v in views
            push!(obsonly, ObservationOutput(v, sim.simdir, outdir))
        end
    catch
        foreach(o -> isnothing(o) || close(o), obsonly)
        rethrow()
    end
    exobs = multi ? ExchangeObservations(m.solution, sim.simdir, outdir) : nothing
    s = Simulation(
        sim, m, ps, first(accs), out, T, Ts, Tp, steps, 0, :initialized, OuterState(ps), StepResult[], ModelBudget[],
        Dict{String, _Mirror}(), Int[], nothing, views, String[_vname(v) for v in views], outs, accs, exflows, exobs,
        obsonly,
    )
    for (v, o) in zip(views, outs)
        _listing_header(o.listing, v, ps, T, Ts, Tp, backend)
    end
    return s
end

# T6.2 (from T6.1): output written next to the inputs (the default `outdir`) once overwrote
# an MF6 reference run of the parity tests. With the environment variable REFERENCE_DIR set
# (the reference tree of validation/fetch_reference.sh) an `outdir` inside it is an error.
function _check_output_dir(outdir::AbstractString)
    ref = get(ENV, "REFERENCE_DIR", "")
    (isempty(ref) || !isdir(ref)) && return nothing
    r = realpath(ref)
    o = abspath(outdir)
    o = ispath(o) ? realpath(o) : o
    (o == r || startswith(o, joinpath(r, ""))) && throw(
        ArgumentError(
            "refusing to write output into $outdir: it is inside REFERENCE_DIR ($ref), the MF6 reference runs; pass another `outdir` or `write_output = false`",
        ),
    )
    return nothing
end

function _check_stage(s::Simulation, f::Symbol, allowed::Symbol...)
    s.stage in allowed ||
        throw(ArgumentError("$f: called after $(s.stage) (allowed after: $(join(allowed, ", ")))"))
    return nothing
end

"""
    prepare_time_step(sim::Simulation, dt = 0.0) -> sim

Advance to the next TDIS time step and, at the first step of a stress period, apply the
period data (libmf6's `prepare_time_step`). `dt` is ignored, as in libmf6 (no adaptive
time stepping). Values set through the stepping API before this call are kept unless the
period has new data for that package (as in MF6). Values written into host mirrors
([`get_value_ptr`](@ref)) are sent to the device first.

Allowed after [`initialize`](@ref) and after [`finalize_time_step`](@ref); throws an
`ArgumentError` otherwise, or when the simulation has ended.

# Example
One time step, call by call (what [`ModflowGPU.update`](@ref) does):
```julia
prepare_time_step(sim)
prepare_solve(sim)
for _ in 1:only(get_value(sim, "SLN_1/MXITER"))
    ModflowGPU.solve(sim) && break
end
finalize_solve(sim)
finalize_time_step(sim)
```
"""
function prepare_time_step(s::Simulation, dt::Real = 0.0)
    _check_stage(s, :prepare_time_step, :initialized, :finalized_time_step)
    s.istep < length(s.steps) || throw(ArgumentError("prepare_time_step: the simulation has ended (time $(get_end_time(s)))"))
    _push_mirrors!(s)
    s.istep += 1
    step = s.steps[s.istep]
    window = time_step_window(s.steps, s.istep)
    step.kstp == 1 && @timed_section :period_data apply_period!(s.model, step.kper; window)
    # Observations (T6.7, `bnd_rp_obs`): the records of the package observations.
    if step.kstp == 1
        for o in s.outputs
            isnothing(o.obs) || obs_rp!(o.obs, s.model)
        end
        for o in s.obsonly
            isnothing(o) || obs_rp!(o, s.model)
        end
    end
    # Time series (T6.3): `idm_ts_advance` after `tdis_set_timestep`.
    @timed_section :period_data advance_time_series!(s.model, window)
    s.stage = :prepared_time_step
    _pull_mirrors!(s)
    return s
end

"""
    prepare_solve(sim::Simulation, isol = 1) -> sim

Start the outer iterations of the current time step (libmf6's `prepare_solve`): the
storage term for the step length, the heads of the previous step and the constant heads;
it resets the iteration counter. Values written into host mirrors since the last call are
sent to the device first, and BOUND arrays opened with [`ModflowGPU.device_bound!`](@ref)
are checked. `isol` is the solution number; only 1 exists.

Allowed after [`prepare_time_step`](@ref); see there for an example.
"""
function prepare_solve(s::Simulation, isol::Integer = 1)
    _check_solution(isol)
    _check_stage(s, :prepare_solve, :prepared_time_step)
    _check_device_writes!(s)
    _push_mirrors!(s)
    @timed_section :start_timestep start_timestep!(s.model, s.steps[s.istep].delt)
    s.outer = OuterState(s.solver)
    s.stage = :prepared_solve
    _pull_mirrors!(s)
    return s
end

"""
    ModflowGPU.solve(sim::Simulation, isol = 1) -> Bool

One outer (Picard) iteration of the current time step (libmf6's `solve`); returns whether
the step has converged. Call it up to MXITER times (`get_value(sim, "SLN_1/MXITER")`).
`isol` is the solution number; only 1 exists.

Not exported, because `solve` is also exported by CommonSolve (the SciML packages) and
would be ambiguous next to it: write `ModflowGPU.solve`. See [`prepare_time_step`](@ref)
for an example.
"""
function solve(s::Simulation, isol::Integer = 1)
    _check_solution(isol)
    _check_stage(s, :solve, :prepared_solve, :solving)
    s.stage = :solving
    return outer_iteration!(s.outer, s.solver, s.model, s.steps[s.istep].kstp)
end

"""
    finalize_solve(sim::Simulation, isol = 1) -> Bool

End the outer iterations of the current time step (libmf6's `finalize_solve`): compute the
flows (the `SIMVALS` of the stress packages) and the budget of the step. Returns whether
the step converged (libmf6 reports a failure status when it did not). `isol` is the
solution number; only 1 exists. See [`prepare_time_step`](@ref) for an example.
"""
function finalize_solve(s::Simulation, isol::Integer = 1)
    _check_solution(isol)
    _check_stage(s, :finalize_solve, :solving, :prepared_solve)
    m = s.model
    step = s.steps[s.istep]
    @timed_section :flows calc_flows!(m)
    if isnothing(s.exflows)
        b = @timed_section :budget model_budget!(s.acc, m, step)
        push!(s.budgets, b)
    else
        # T6.6: the exchange flows, then the budget of every model.
        @timed_section :flows exchange_flows!(s.exflows, m)
        for (v, acc) in zip(s.views, s.accs)
            push!(s.budgets, @timed_section :budget model_budget!(acc, v, step, s.exflows))
        end
    end
    push!(s.results, StepResult(step, OuterResult(s.outer)))
    s.stage = :finalized_solve
    _pull_mirrors!(s; lost = true)
    return s.outer.converged
end

"""
    finalize_time_step(sim::Simulation) -> Bool

End the current time step (libmf6's `finalize_time_step`): write the output selected by
OC (head file, budget file, listing file). Returns `false` when the step did not converge
and the simulation has no CONTINUE option (libmf6 returns a failure status;
[`run_simulation`](@ref) throws a [`ConvergenceError`](@ref) there), `true` otherwise.
See [`prepare_time_step`](@ref) for an example.
"""
function finalize_time_step(s::Simulation)
    _check_stage(s, :finalize_time_step, :finalized_solve)
    r = last(s.results)
    nv = length(s.views)
    for (k, (v, o)) in enumerate(zip(s.views, s.outputs))
        b = s.budgets[end - nv + k]
        @timed_section :output _write_step(o, v, r.step, r.outer, b, s.spec; exflows = s.exflows)
    end
    # T6.9: the observations of a run without output (`gwf_ot_obs`).
    for o in s.obsonly
        isnothing(o) || write_observations(o, s.model, r.step.totim)
    end
    isnothing(s.exobs) || write_exchange_observations(s.exobs, s.exflows, r.step.totim)
    s.stage = :finalized_time_step
    ok = r.outer.converged || s.spec.continue_on_nonconvergence
    if !r.outer.converged && s.spec.continue_on_nonconvergence
        @warn "Time step $(r.step.kstp) of stress period $(r.step.kper) did not converge ($(r.outer.iterations) outer iterations); continuing (CONTINUE)"
    end
    return ok
end

"""
    do_time_step(sim::Simulation) -> Bool

Solve the current time step: [`prepare_solve`](@ref), up to MXITER
[`ModflowGPU.solve`](@ref) calls, [`finalize_solve`](@ref) (libmf6's `do_time_step`).
Returns whether the step converged. Call [`prepare_time_step`](@ref) before it and
[`finalize_time_step`](@ref) after it.

# Example
```julia
prepare_time_step(sim)
# ... set boundary values of this step with set_value ...
do_time_step(sim) || @warn "no convergence"
finalize_time_step(sim)
```
"""
function do_time_step(s::Simulation)
    prepare_solve(s)
    for _ in 1:s.solver.settings.maxiter
        (solve(s) || s.outer.diverged) && break
    end
    return finalize_solve(s)
end

"""
    ModflowGPU.update(sim::Simulation) -> Bool

Run one full time step: [`prepare_time_step`](@ref), [`do_time_step`](@ref),
[`finalize_time_step`](@ref) (libmf6's `update`). Returns the result of
`finalize_time_step`: `false` when the step did not converge and the simulation has no
CONTINUE option.

Not exported, because CUDA.jl exports an `update` (for CUDA graphs) and
`using CUDA, ModflowGPU` would make `update` ambiguous: write `ModflowGPU.update`.

This is not `BasicModelInterface.update`, which throws a [`ConvergenceError`](@ref) in
that case and returns `nothing`.

# Example
```julia
sim = initialize("path/to/simulation")
ModflowGPU.update(sim) # the first time step
get_current_time(sim)  # the time at its end
```
"""
function update(s::Simulation)
    prepare_time_step(s)
    do_time_step(s)
    return finalize_time_step(s)
end

"""
    finalize_simulation(sim::Simulation; free = true) -> sim

End a simulation started with [`initialize`](@ref) (libmf6's `finalize`): close the
output files and, with `free` (the default), release the device memory of the model now
([`ModflowGPU.free_device_arrays!`](@ref)) rather than when the garbage collector runs.
This matters when another model on the same GPU allocates next.

Afterwards the simulation cannot be stepped, and with `free = true` not read either
([`get_value`](@ref) and [`get_value_ptr`](@ref) throw); device arrays obtained before
(`get_value_ptr(...; device = true)`) must not be used, while host mirrors keep their
last values. With `free = false` the model stays readable ([`run_simulation`](@ref)
returns it) and its memory is released by the garbage collector. A second call does
nothing.

[`ModflowGPU.finalize`](@ref) is the same function under its earlier name.

# Example
```julia
sim = initialize("path/to/simulation")
ModflowGPU.update(sim)
finalize_simulation(sim)
```
"""
function finalize_simulation(s::Simulation; free::Bool = true)
    s.stage === :freed && return s
    foreach(close, s.outputs)
    foreach(o -> isnothing(o) || close(o), s.obsonly)
    isnothing(s.exobs) || close(s.exobs)
    s.stage = :finalized
    if free
        backend = get_backend(s.model.h)
        backend isa CPU || KernelAbstractions.synchronize(backend)
        free_device_arrays!((s.model, s.solver, s.accs, s.check, s.outer, s.outputs, s.exflows, s.obsonly))
        s.stage = :freed
    end
    return s
end

"""
    ModflowGPU.finalize(sim::Simulation; free = true) -> sim

The same function as [`finalize_simulation`](@ref), under the name used before it was
exported. Not exported, because it would clash with `Base.finalize`; it is kept (not
deprecated) for code that calls `ModflowGPU.finalize`.
"""
const finalize = finalize_simulation

"""
    ModflowGPU.device_bound!(sim, address) -> device BOUND array

For coupling on the device: open the device BOUND array at `address`
(`"<model>/<package>/BOUND"`) for writing in the current time step, so that kernels of
the coupled model can write boundary values without a copy to the host. Allowed
only between `prepare_time_step` and `prepare_solve` (otherwise an `ArgumentError`).
Returns the engine array, as `get_value_ptr(sim, address; device = true)`: precision `T`,
`(ncol, maxbound)`, head-like columns as offsets from the reference of the record's cell
(`rrec`), multiplier applied. `prepare_solve` then checks the first `NBOUND` records of
every opened BOUND on the device (one kernel and one host-mapped flag per package): all
values finite, and for RIV stage ≥ bottom and conductance ≥ 0 (`riv_ck`); a violation
throws an `MF6InputError` naming the first bad record. The written values are used by the
step and kept in later steps until new period data or a host write of BOUND replace them
(T6.9: a time series of the package rewrites only its own entries, and RCH/EVT records
moved to another cell keep their values), and are not seen by host reads of BOUND
(`get_value`, mirrors), which keep the values last set on the host. Writes into a device
BOUND that was not opened this way are not checked (the stepping API cannot see them).

Not exported: it is for code that writes device arrays in its own kernels.

# Example
```julia
prepare_time_step(sim)
bound = ModflowGPU.device_bound!(sim, "GWF/RIV-1/BOUND")   # (3, maxbound) on the device
my_stage_kernel!(bound, ...)   # write the stages (row 1) as offsets from the references
do_time_step(sim)
finalize_time_step(sim)
```
"""
function device_bound!(s::Simulation, address::AbstractString)
    _check_readable(s)
    parts = split(uppercase(address), '/')
    (length(parts) == 3 && parts[3] == "BOUND") ||
        throw(ArgumentError("device_bound!: \"$address\" is not a BOUND address (<model>/<package>/BOUND)"))
    s.stage === :prepared_time_step ||
        throw(ArgumentError("device_bound!: device BOUND writes are allowed only between prepare_time_step and prepare_solve (called after $(s.stage))"))
    arr = _variable(s, address).device(s)
    ip = _find_package(s, parts[1], parts[2])
    ip in s.device_writes || push!(s.device_writes, ip)
    s.model.packages[ip].device_written = true
    return arr
end

# Whether record `i` of a device BOUND is acceptable (the device form of `check_records`
# for what ModflowCoupling writes; offsets compare as the absolute values do, since a
# record's head-like columns share one reference).
@inline function _finite_record(bound, i)
    ok = true
    for c in axes(bound, 1)
        ok &= isfinite(bound[c, i])
    end
    return ok
end
@inline _device_record_ok(::BoundaryKind, bound, i) = _finite_record(bound, i)
@inline _device_record_ok(::BoundaryKind{:RIV}, bound, i) =
    _finite_record(bound, i) && bound[1, i] >= bound[3, i] && bound[2, i] >= zero(eltype(bound))

@kernel function device_bound_check_kernel!(flag, @Const(bound), kind)
    i = @index(Global)
    @inbounds _device_record_ok(kind, bound, i) || (flag[1] = Int32(1))
end

# The flag of the device BOUND check: a host-mapped Int32 where the backend allows it
# (no copy to read it), else a device array.
function _check_flag(s::Simulation)
    isnothing(s.check) || return s.check
    backend = get_backend(s.model.h)
    mapped = host_mapped_zeros(backend, Int32, 1)
    s.check = isnothing(mapped) ? (KernelAbstractions.zeros(backend, Int32, 1), nothing) : mapped
    return s.check
end

# `prepare_solve`: check the device BOUND arrays opened with `device_bound!` in this step.
function _check_device_writes!(s::Simulation)
    isempty(s.device_writes) && return nothing
    backend = get_backend(s.model.h)
    flag, host = _check_flag(s)
    for ip in s.device_writes
        pkg = s.model.packages[ip]
        pkg.nbound > 0 || continue
        isnothing(host) ? fill!(flag, Int32(0)) : (host[1] = Int32(0))
        device_bound_check_kernel!(backend, _WG)(flag, pkg.bound, BoundaryKind{boundary_kind(pkg)}(); ndrange = pkg.nbound)
        KernelAbstractions.synchronize(backend)
        bad = isnothing(host) ? Array(flag)[1] : host[1]
        bad == 0 || _device_bound_error(pkg)
    end
    empty!(s.device_writes)
    return nothing
end

# Error path: find the first bad record on the host and name it.
function _device_bound_error(pkg::BoundaryPackage{K}) where {K}
    bound = Array(pkg.bound)
    for i in 1:pkg.nbound
        _device_record_ok(BoundaryKind{K}(), bound, i) && continue
        rec = Float64.(bound[:, i])
        reason = !all(isfinite, rec) ? "non-finite value" :
            rec[1] < rec[3] ? "RIVER STAGE IS LESS THAN RIVER BOTTOM" : "CONDUCTANCE IS LESS THAN ZERO"
        throw(MF6InputError(pkg.spec.file, 0, "$(pkg.spec.name): BOUND record $i written on the device: $reason (values as offsets: $(join(rec, ", ")))"))
    end
    return nothing
end

_check_readable(s::Simulation) = s.stage === :freed &&
    throw(ArgumentError("the simulation was finalized and its device memory freed (finalize_simulation(sim; free = false) keeps it readable)"))

_check_solution(isol) = isol == 1 || throw(ArgumentError("solution $isol: the simulation has one solution (SLN_1)"))

"""
    get_current_time(sim::Simulation) -> Float64

The simulation time at the end of the current time step (TDIS `totim`, in the model's
time units; 0 before the first [`prepare_time_step`](@ref)).

# Example
```julia
sim = initialize("path/to/simulation")
get_current_time(sim)   # 0.0
ModflowGPU.update(sim)
get_current_time(sim)   # the length of the first time step
```
"""
get_current_time(s::Simulation) = s.istep == 0 ? 0.0 : s.steps[s.istep].totim

"""
    get_start_time(sim::Simulation) -> Float64

The start time of the simulation: always 0.0, in the model's time units.
"""
get_start_time(::Simulation) = 0.0

"""
    get_end_time(sim::Simulation) -> Float64

The end time of the simulation: the sum of the stress-period lengths (TDIS PERLEN), in
the model's time units.

# Example
```julia
while get_current_time(sim) < get_end_time(sim)
    ModflowGPU.update(sim)
end
```
"""
get_end_time(s::Simulation) = isempty(s.steps) ? 0.0 : last(s.steps).totim

"""
    get_time_step(sim::Simulation) -> Float64

The length of the current time step (TDIS `delt`, in the model's time units, as libmf6's
`get_time_step`; before the first step, that of the first step).
"""
get_time_step(s::Simulation) = isempty(s.steps) ? 0.0 : s.steps[max(s.istep, 1)].delt

# ---------------------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------------------

"""
    get_var_address(sim::Simulation, var, component; subcomponent_name = "") -> String

The address of variable `var` of `component` (a model name, `"SLN_1"` or `"TDIS"`) and
an optional subcomponent (a package name), as libmf6's `get_var_address`: the names in
upper case, joined by `/`. The address is not checked; [`var_addresses`](@ref) lists the
valid ones.

# Example
```julia
get_var_address(sim, "bound", "gwf"; subcomponent_name = "riv-1")   # "GWF/RIV-1/BOUND"
get_var_address(sim, "x", "gwf")                                     # "GWF/X"
```
"""
get_var_address(::Simulation, var::AbstractString, component::AbstractString; subcomponent_name::AbstractString = "") =
    join(filter(!isempty, uppercase.([component, subcomponent_name, var])), "/")

_scalar(x) = [x]

_dis_names(m::GwfModel) = _dis_names(m.spec)
function _dis_names(spec::GwfSpec)
    names = Set(["DIS"])
    for p in spec.packages
        p.ftype in ("DIS6", "DISV6", "DISU6") && push!(names, uppercase(p.name))
    end
    return names
end

_tdis_step(s::Simulation) = s.steps[max(s.istep, 1)]

_find_package(m::GwfModel, name) = findfirst(p -> uppercase(p.spec.name) == name, m.packages)

# T6.6: the model `mname` (upper case) of the simulation (its view), or `nothing`.
_find_view(s::Simulation, mname) = (k = findfirst(v -> uppercase(_vname(v)) == mname, s.views); isnothing(k) ? nothing : s.views[k])

# The index in `s.model.packages` of package `pname` of model `mname` (upper case), or `nothing`.
function _find_package(s::Simulation, mname, pname)
    v = _find_view(s, mname)
    isnothing(v) && return nothing
    m = _vmodel(v)
    i = findfirst(ip -> uppercase(m.packages[ip].spec.name) == pname, _vpackages(v))
    return isnothing(i) ? nothing : _vpackages(v)[i]
end

# The variable at `address` (uppercase), or an error listing the supported addresses.
function _variable(s::Simulation, address::AbstractString)
    parts = split(uppercase(address), '/')
    m = s.model
    T = s.T
    # T6.6: a model of a multi-model solution.
    if length(parts) in (2, 3) && !isnothing(m.solution)
        v = _find_view(s, parts[1])
        isnothing(v) || return _view_variable(s, v, address, parts)
    end
    mname = uppercase(m.spec.name)
    r = m.setup.r
    if length(parts) == 2 && parts[1] == "SLN_1"
        parts[2] == "MXITER" && return _Variable(_ -> _scalar(Int32(s.solver.settings.maxiter)), nothing, nothing, "-")
    elseif length(parts) == 2 && parts[1] == "TDIS"
        step = _tdis_step
        v = parts[2]
        v == "DELT" && return _Variable(s -> _scalar(get_time_step(s)), nothing, nothing, "T")
        v == "TOTIM" && return _Variable(s -> _scalar(get_current_time(s)), nothing, nothing, "T")
        v == "PERTIM" && return _Variable(s -> _scalar(s.istep == 0 ? 0.0 : step(s).pertim), nothing, nothing, "T")
        v == "KPER" && return _Variable(s -> _scalar(Int32(s.istep == 0 ? 0 : step(s).kper)), nothing, nothing, "-")
        v == "KSTP" && return _Variable(s -> _scalar(Int32(s.istep == 0 ? 0 : step(s).kstp)), nothing, nothing, "-")
        v == "NPER" && return _Variable(s -> _scalar(Int32(nper(s.spec.tdis))), nothing, nothing, "-")
    elseif length(parts) == 2 && parts[1] == mname
        v = parts[2]
        v == "X" && return _Variable(
            s -> absolute_heads(s.model),
            (s, x) -> copyto!(s.model.h, T.(to_offsets(r, vec(x)))),
            s -> s.model.h, "L",
        )
        v == "XOLD" && return _Variable(s -> r .+ Float64.(Array(s.model.hold)), nothing, s -> s.model.hold, "L")
        v == "HREF" && return _Variable(s -> copy(r), nothing, s -> r, "L")
        v == "IBOUND" && return _Variable(s -> copy(s.model.ibound_host), nothing, s -> s.model.ibound, "-")
    elseif length(parts) == 3 && parts[1] == mname && parts[2] in _dis_names(m)
        c = m.conn
        v = parts[3]
        v == "NODEUSER" && return _Variable(_ -> copy(c.nodeuser), nothing, nothing, "-")
        v == "NODEREDUCED" && return _Variable(_ -> copy(c.nodereduced), nothing, nothing, "-")
        v == "TOP" && return _Variable(_ -> copy(c.top), nothing, nothing, "L")
        v == "BOT" && return _Variable(_ -> copy(c.bot), nothing, nothing, "L")
        v == "AREA" && return _Variable(_ -> copy(c.area), nothing, nothing, "L2")
        v == "NODES" && return _Variable(_ -> _scalar(Int32(c.nodes)), nothing, nothing, "-")
        v == "NODESUSER" && return _Variable(_ -> _scalar(Int32(c.nodesuser)), nothing, nothing, "-")
    elseif length(parts) == 3 && parts[1] == mname && !isnothing(_find_package(m, parts[2]))
        pkg = let ip = _find_package(m, parts[2])
            s -> s.model.packages[ip]
        end
        v = parts[3]
        v == "BOUND" && return _Variable(
            s -> copy(bound_user!(pkg(s))),
            (s, x) -> (copyto!(pkg(s).bound_user, x); set_bound!(pkg(s), s.model.setup.r)),
            s -> pkg(s).bound, "-", s -> pkg(s).bound_version, s -> bound_user!(pkg(s)),
        )
        v == "NODELIST" && return _Variable(
            s -> copy(pkg(s).nodelist_host), nothing, s -> pkg(s).nodelist, "-", s -> pkg(s).bound_version, s -> pkg(s).nodelist_host,
        )
        v == "SIMVALS" && return _Variable(s -> Float64.(Array(pkg(s).simvals)), nothing, s -> pkg(s).simvals, "L3/T")
        v == "NBOUND" && return _Variable(s -> _scalar(Int32(pkg(s).nbound)), nothing, nothing, "-")
        v == "MAXBOUND" && return _Variable(s -> _scalar(Int32(length(pkg(s).nodelist_host))), nothing, nothing, "-")
    end
    throw(ArgumentError("unknown address \"$address\"; supported: $(join(var_addresses(s), ", "))"))
end

# T6.6: the variables of model `v` of a multi-model solution: its slice of the merged
# model's arrays (`X`, `XOLD`, `IBOUND`: device views), its own grid and packages. NODELIST
# holds the model's reduced nodes on the host and the solution's on the device.
function _view_variable(s::Simulation, v::ModelView, address::AbstractString, parts)
    T = s.T
    rg = _vrange(v)
    off = _voffset(v)
    r = _vsetup(v).r
    nn = length(rg)
    if length(parts) == 2
        x = parts[2]
        x == "X" && return _Variable(
            s -> absolute_heads(v),
            (s, xs) -> copyto!(s.model.h, off + 1, T.(to_offsets(r, vec(xs))), 1, nn),
            s -> view(s.model.h, rg), "L",
        )
        x == "XOLD" && return _Variable(s -> r .+ Float64.(Array(s.model.hold)[rg]), nothing, s -> view(s.model.hold, rg), "L")
        x == "HREF" && return _Variable(s -> copy(r), nothing, s -> r, "L")
        x == "IBOUND" && return _Variable(s -> s.model.ibound_host[rg], nothing, s -> view(s.model.ibound, rg), "-")
    elseif parts[2] in _dis_names(_vspec(v))
        c = _vconn(v)
        x = parts[3]
        x == "NODEUSER" && return _Variable(_ -> copy(c.nodeuser), nothing, nothing, "-")
        x == "NODEREDUCED" && return _Variable(_ -> copy(c.nodereduced), nothing, nothing, "-")
        x == "TOP" && return _Variable(_ -> copy(c.top), nothing, nothing, "L")
        x == "BOT" && return _Variable(_ -> copy(c.bot), nothing, nothing, "L")
        x == "AREA" && return _Variable(_ -> copy(c.area), nothing, nothing, "L2")
        x == "NODES" && return _Variable(_ -> _scalar(Int32(c.nodes)), nothing, nothing, "-")
        x == "NODESUSER" && return _Variable(_ -> _scalar(Int32(c.nodesuser)), nothing, nothing, "-")
    else
        ip = _find_package(s, parts[1], parts[2])
        if !isnothing(ip)
            pkg = s -> s.model.packages[ip]
            x = parts[3]
            x == "BOUND" && return _Variable(
                s -> copy(bound_user!(pkg(s))),
                (s, xs) -> (copyto!(pkg(s).bound_user, xs); set_bound!(pkg(s), s.model.setup.r)),
                s -> pkg(s).bound, "-", s -> pkg(s).bound_version, s -> bound_user!(pkg(s)),
            )
            x == "NODELIST" && return _Variable(
                s -> Int32[n > 0 ? n - off : n for n in pkg(s).nodelist_host], nothing, s -> pkg(s).nodelist, "-",
            )
            x == "SIMVALS" && return _Variable(s -> Float64.(Array(pkg(s).simvals)), nothing, s -> pkg(s).simvals, "L3/T")
            x == "NBOUND" && return _Variable(s -> _scalar(Int32(pkg(s).nbound)), nothing, nothing, "-")
            x == "MAXBOUND" && return _Variable(s -> _scalar(Int32(length(pkg(s).nodelist_host))), nothing, nothing, "-")
        end
    end
    throw(ArgumentError("unknown address \"$address\"; supported: $(join(var_addresses(s), ", "))"))
end

"""
    var_addresses(sim::Simulation) -> Vector{String}

Every address that [`get_value`](@ref), [`get_value_ptr`](@ref) and the BMI accept, as
libmf6 names them, for a model `<model>` with stress packages `<package>`:

- `<model>/X`, `XOLD` (heads now and at the end of the previous step), `HREF` (the
  per-cell reference heads), `IBOUND`;
- `<model>/DIS/NODEUSER`, `NODEREDUCED`, `TOP`, `BOT`, `AREA`, `NODES`, `NODESUSER`;
- `<model>/<package>/BOUND`, `NODELIST`, `SIMVALS` (simulated flows), `NBOUND`,
  `MAXBOUND`;
- `SLN_1/MXITER` and `TDIS/DELT`, `TOTIM`, `PERTIM`, `KPER`, `KSTP`, `NPER`.

With several models in the solution (T6.6) the model addresses are there for every model
(`<model>` its name). The device arrays of `X`, `XOLD` and `IBOUND` (`get_value_ptr(...;
device = true)`) are then views of the solution's arrays, and the device NODELIST holds the
solution's node numbers (the host NODELIST the model's).

The writable ones are listed by [`input_addresses`](@ref).

# Example
```julia
sim = initialize("path/to/simulation")
filter(contains("/RIV"), var_addresses(sim))
```
"""
function var_addresses(s::Simulation)
    a = String[]
    for mv in s.views
        mn = uppercase(_vname(mv))
        append!(a, ["$mn/$v" for v in ("X", "XOLD", "HREF", "IBOUND")])
        append!(a, ["$mn/DIS/$v" for v in ("NODEUSER", "NODEREDUCED", "TOP", "BOT", "AREA", "NODES", "NODESUSER")])
        for ip in _vpackages(mv)
            p = _vmodel(mv).packages[ip]
            append!(a, ["$mn/$(uppercase(p.spec.name))/$v" for v in ("BOUND", "NODELIST", "SIMVALS", "NBOUND", "MAXBOUND")])
        end
    end
    push!(a, "SLN_1/MXITER")
    append!(a, ["TDIS/$v" for v in ("DELT", "TOTIM", "PERTIM", "KPER", "KSTP", "NPER")])
    return a
end

"""
    input_addresses(sim::Simulation) -> Vector{String}

The addresses that [`set_value`](@ref) accepts (and whose [`get_value_ptr`](@ref) mirrors
may be written): `"<model>/X"` and `"<model>/<package>/BOUND"` of every stress package.
"""
input_addresses(s::Simulation) = filter(a -> !isnothing(_variable(s, a).set), var_addresses(s))

"""
    get_value_ptr(sim::Simulation, address; device = false) -> Array

A host array that mirrors the variable at `address` (see [`var_addresses`](@ref)), as
libmf6's `get_value_ptr`. Values are Float64 with absolute heads (in the model's length
units), indices Int32; BOUND is `(ncol, maxbound)`, with the values before the MULTIPLIER
auxiliary variable is applied.

The same array is returned on every call and stays valid until
[`finalize_simulation`](@ref). It is a copy of the engine data, filled when it is created
and refreshed at the end of [`prepare_time_step`](@ref), [`prepare_solve`](@ref) and
[`finalize_solve`](@ref). Writes into the mirror of `X` or `BOUND` are sent to the device
at the next `prepare_time_step` or `prepare_solve`; writes between `prepare_solve` and
`finalize_solve` are discarded with a warning (use [`set_value`](@ref) there).

With `device = true`: the engine array itself, on the device (values in the precision `T`
of the simulation, indices Int32), with head-like values as offsets from `"<model>/HREF"` and BOUND with the
multiplier applied; only for `X`, `XOLD`, `HREF`, `IBOUND`, `BOUND`, `NODELIST` and
`SIMVALS` (`HREF` is the host Float64 vector of the references). No mirror is made.
Write a device BOUND only through [`ModflowGPU.device_bound!`](@ref).

# Example
```julia
heads = get_value_ptr(sim, "GWF/X")          # updated by every time step
riv = get_value_ptr(sim, "GWF/RIV-1/BOUND")  # stage, cond, rbot per river cell
prepare_time_step(sim)
riv[1, :] .+= 0.1                            # raise the stages for this step
do_time_step(sim)
finalize_time_step(sim)
```
"""
function get_value_ptr(s::Simulation, address::AbstractString; device::Bool = false)
    _check_readable(s)
    var = _variable(s, address)
    if device
        isnothing(var.device) && throw(ArgumentError("get_value_ptr: \"$address\" has no device array (use device = false)"))
        return var.device(s)
    end
    key = uppercase(address)
    mir = get!(() -> _Mirror(s, var), s.mirrors, key)
    return mir.host
end

"""
    get_value(sim::Simulation, address) -> Array
    get_value(sim::Simulation, address, dest) -> dest

A new host array (or `dest`, filled) with the current value of the variable at `address`
(see [`var_addresses`](@ref)), read from the engine now. Values are as in
[`get_value_ptr`](@ref): Float64 with absolute heads, Int32 indices; scalars are
one-element vectors.

# Example
```julia
h = get_value(sim, "GWF/X")                     # heads per model node
q = get_value(sim, "GWF/RIV-1/SIMVALS")         # river flows of the last step
kper = only(get_value(sim, "TDIS/KPER"))
```
"""
get_value(s::Simulation, address::AbstractString) = (_check_readable(s); _variable(s, address).get(s))
get_value(s::Simulation, address::AbstractString, dest::AbstractArray) = copyto!(dest, get_value(s, address))

"""
    set_value(sim::Simulation, address, values) -> sim

Set the variable at `address` (one of [`input_addresses`](@ref)) and send it to the
device now. `X` takes absolute heads per model node (cells with IDOMAIN > 0); `BOUND`
takes `(ncol, maxbound)` values (or the same values flattened), before the MULTIPLIER
auxiliary variable, with head-like columns (stage, head, elevation) absolute, in the model's units. A host mirror of
the address ([`get_value_ptr`](@ref)) is updated as well. The values are used from the
next solve until new period data for the package replaces them.

# Example
```julia
prepare_time_step(sim)
bound = get_value(sim, "GWF/WEL-1/BOUND")
bound[1, :] .= -100.0                   # pumping rate of every well
set_value(sim, "GWF/WEL-1/BOUND", bound)
do_time_step(sim)
finalize_time_step(sim)
```
"""
function set_value(s::Simulation, address::AbstractString, values::AbstractArray)
    _check_readable(s)
    var = _variable(s, address)
    isnothing(var.set) && throw(ArgumentError("set_value: \"$address\" is read-only"))
    cur = var.get(s)
    length(values) == length(cur) || throw(DimensionMismatch("set_value: $(length(values)) values for \"$address\" of length $(length(cur))"))
    copyto!(cur, Float64.(vec(values)))
    var.set(s, cur)
    mir = get(s.mirrors, uppercase(address), nothing)
    if !isnothing(mir)
        copyto!(mir.host, cur)
        isnothing(var.version) ? copyto!(mir.synced, cur) : (mir.version[] = var.version(s))
    end
    return s
end

function _push_mirrors!(s::Simulation)
    isempty(s.mirrors) && return nothing
    @timed_section :mirrors _push_mirrors_all!(s)
    return nothing
end

# Whether two host arrays hold the same bits (a memcmp for bits types; -0.0 and 0.0
# differ, which only costs an unneeded push).
function _same_bits(a::Array{T}, b::Array{T}) where {T}
    size(a) == size(b) || return false
    isbitstype(T) || return isequal(a, b)
    return GC.@preserve a b ccall(:memcmp, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), a, b, sizeof(a)) == 0
end
_same_bits(a, b) = isequal(a, b)

# The value a mirror was last synchronised with (versioned: the engine array, unless the
# engine changed since; then `nothing`).
function _synced_value(s::Simulation, mir::_Mirror)
    var = mir.var
    isnothing(var.version) && return mir.synced
    return var.version(s) == mir.version[] ? var.source(s) : nothing
end

# Send the writable mirrors that were written since the last synchronisation (dirty: they
# differ from the synchronised value). A versioned mirror whose engine value changed
# outside the API (no stepping call in between) is not pushed; the next pull replaces it.
function _push_mirrors_all!(s::Simulation)
    for mir in values(s.mirrors)
        isnothing(mir.var.set) && continue
        ref = _synced_value(s, mir)
        (isnothing(ref) || _same_bits(mir.host, ref)) && continue
        mir.var.set(s, mir.host)
        if isnothing(mir.var.version)
            copyto!(mir.synced, mir.host)
        else
            mir.version[] = mir.var.version(s)
        end
    end
    return nothing
end

# `lost = true` (finalize_solve): warn about writes into a writable mirror since
# `prepare_solve`, which the refresh discards (they are not sent during the outer iterations).
function _pull_mirrors!(s::Simulation; lost::Bool = false)
    isempty(s.mirrors) && return nothing
    @timed_section :mirrors _pull_mirrors_all!(s, lost)
    return nothing
end

#
# Versioned mirrors are refreshed only when the engine value changed (new period data, a
# write) or, with `lost`, when the mirror was written: no copy otherwise.
function _pull_mirrors_all!(s::Simulation, lost::Bool)
    for (key, mir) in s.mirrors
        var = mir.var
        ref = _synced_value(s, mir)
        dirty = lost && !isnothing(ref) && !_same_bits(mir.host, ref)
        lost && !isnothing(var.set) && dirty &&
            @warn "\"$key\" was written through get_value_ptr after prepare_solve; the write is discarded (write before prepare_solve, or use set_value)"
        if isnothing(var.version)
            v = var.get(s)
            copyto!(mir.host, v)
            copyto!(mir.synced, v)
        elseif isnothing(ref) || dirty
            copyto!(mir.host, var.source(s))
            mir.version[] = var.version(s)
        end
    end
    return nothing
end
