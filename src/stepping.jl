# Stepping API (PLAN.md section 3.6): the time loop of `ModflowGPU.run` cut into the
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
#   do_time_step, update     the steps above in one call; finalize
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
#   - every mirror is refreshed from the engine at the end of `initialize`,
#     `prepare_time_step`, `prepare_solve` and `finalize_solve` (so X during the outer
#     iterations is the X at the start of the step; read it with `get_value`). BOUND and
#     NODELIST mirrors are versioned (T4.6): they are copied only when the engine value
#     changed (new period data, a write) or, in `finalize_solve`, when the mirror differs.
#   - writes into a mirror between `prepare_solve` and `finalize_solve` are discarded by
#     the refresh in `finalize_solve`, with a warning (use `set_value` there).
#
# `get_value_ptr(sim, address; device = true)` returns the engine arrays themselves
# (device, precision T, head-like values as offsets from "<model>/HREF", BOUND with the
# multiplier applied); writes there bypass the conversions and the mirrors. Device writes
# into BOUND (T5.2: ModflowCoupling's device-resident coupling writes them in kernels)
# between `prepare_time_step` and `prepare_solve` are used by the step; they are not
# checked (`riv_ck` etc.), are replaced by new period data, and are not seen by host reads
# of BOUND (`get_value`, mirrors), which keep the last values set on the host. A host
# mirror of the same BOUND does not overwrite them unless it is written. A caller that
# keeps no host mirrors (reads through `get_value` only) makes a time step move no array
# between host and device: the mirrors are the only per-step copies of the stepping API,
# and the output of `finalize_time_step` copies the heads only when OC saves or prints them.

"""
    StepResult

Per time step of a run: the `step` (`TimeStep`) and the `outer` result (`OuterResult`:
convergence, outer and inner iterations).
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

A simulation driven step by step (the stepping API, `initialize`): the `spec`, the
`model` (`GwfModel`), the outer solver (`PicardSolver`), the budget accumulator, the
output files (or `nothing`), the precision types `T`, `Ts`, `Tp`, the TDIS `steps` and the
index `istep` of the current one (0 before the first `prepare_time_step`), the `stage` of
the call sequence, the outer-iteration state of the current step, and per completed
step its `StepResult` and `ModelBudget` (`results`, `budgets`).
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
end

Base.show(io::IO, s::Simulation) = print(
    io, "Simulation(\"", s.model.spec.name, "\", T = ", s.T, ", step ", s.istep, " of ", length(s.steps), ", ", s.stage, ")",
)

"""
    initialize(simdir or spec::SimulationSpec; backend = CPU(), precision = :auto,
               outdir = simdir, closures = nothing, write_output = true,
               preconditioner = :auto, precond_options = (;)) -> Simulation

Load the simulation (one GWF model with one IMS solution), build the model, the outer
solver and the budget accumulator on `backend` in `precision` (`resolve_precision`), and
open the output files (`write_output`; see `ModflowGPU.run` for the keywords, including
`preconditioner` and `precond_options`). Before the first `prepare_time_step` the heads are the starting heads and no period data is applied.
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
    (length(sim.models) == 1 && isempty(sim.exchanges)) ||
        throw(MF6UnsupportedError(sim.simdir, "unsupported: $(length(sim.models)) models and $(length(sim.exchanges)) exchanges (multi-model simulations arrive in T6.6)"))
    spec = only(sim.models)
    ims = only(only(sim.solution_groups).solutions)
    T, Ts, Tp = resolve_precision(backend, precision)
    steps = time_steps(sim.tdis)
    m = GwfModel(backend, T, spec, nper(sim.tdis))
    ps = PicardSolver(m, PicardSettings(ims, T; closures, preconditioner, precond_options); Ts, Tp)
    acc = BudgetAccumulator(m)
    out = write_output ? ModelOutput(m, sim.simdir, outdir) : nothing
    s = Simulation(
        sim, m, ps, acc, out, T, Ts, Tp, steps, 0, :initialized, OuterState(ps), StepResult[], ModelBudget[],
        Dict{String, _Mirror}(), Int[], nothing,
    )
    isnothing(out) || _listing_header(out.listing, m, ps, T, Ts, Tp, backend)
    return s
end

function _check_stage(s::Simulation, f::Symbol, allowed::Symbol...)
    s.stage in allowed ||
        throw(ArgumentError("$f: called after $(s.stage) (allowed after: $(join(allowed, ", ")))"))
    return nothing
end

"""
    prepare_time_step(sim::Simulation, dt = 0.0) -> sim

Advance to the next TDIS time step and, at the first step of a stress period, apply the
period data (`Mf6PrepareTimestep`). `dt` is ignored, as in libmf6 (no ATS). Values set
through the stepping API before this call are kept unless the period has new data for
that package (as in MF6).
"""
function prepare_time_step(s::Simulation, dt::Real = 0.0)
    _check_stage(s, :prepare_time_step, :initialized, :finalized_time_step)
    s.istep < length(s.steps) || throw(ArgumentError("prepare_time_step: the simulation has ended (time $(get_end_time(s)))"))
    _push_mirrors!(s)
    s.istep += 1
    step = s.steps[s.istep]
    step.kstp == 1 && @timed_section :period_data apply_period!(s.model, step.kper)
    s.stage = :prepared_time_step
    _pull_mirrors!(s)
    return s
end

"""
    prepare_solve(sim::Simulation, isol = 1) -> sim

Start the outer iterations of the current time step (`prepareSolve`, `model_ad`): the
storage term for the step length, `hold ← h`, the constant heads; resets the iteration
counter. Values written into host mirrors since the last call are sent to the device
first.
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
    solve(sim::Simulation, isol = 1) -> Bool

One outer iteration of the current time step (`xmi_solve`); returns whether the step has
converged. Call it up to `MXITER` (`"SLN_1/MXITER"`) times.
"""
function solve(s::Simulation, isol::Integer = 1)
    _check_solution(isol)
    _check_stage(s, :solve, :prepared_solve, :solving)
    s.stage = :solving
    return outer_iteration!(s.outer, s.solver, s.model, s.steps[s.istep].kstp)
end

"""
    finalize_solve(sim::Simulation, isol = 1) -> Bool

End the outer iterations of the current time step (`finalizeSolve`): flows (`calc_flows!`,
SIMVALS) and the budget of the step; returns whether the step converged (libmf6 reports a
failure status when it did not).
"""
function finalize_solve(s::Simulation, isol::Integer = 1)
    _check_solution(isol)
    _check_stage(s, :finalize_solve, :solving, :prepared_solve)
    m = s.model
    step = s.steps[s.istep]
    @timed_section :flows calc_flows!(m)
    b = @timed_section :budget model_budget!(s.acc, m, step)
    push!(s.results, StepResult(step, OuterResult(s.outer)))
    push!(s.budgets, b)
    s.stage = :finalized_solve
    _pull_mirrors!(s; lost = true)
    return s.outer.converged
end

"""
    finalize_time_step(sim::Simulation) -> Bool

End the current time step (`Mf6FinalizeTimestep`): the output selected by OC (HDS, CBC,
listing). Returns `false` when the step did not converge and the simulation has no
CONTINUE option (libmf6 returns a failure status; `ModflowGPU.run` throws a
`ConvergenceError` there), `true` otherwise.
"""
function finalize_time_step(s::Simulation)
    _check_stage(s, :finalize_time_step, :finalized_solve)
    r = last(s.results)
    isnothing(s.output) || @timed_section :output _write_step(s.output, s.model, r.step, r.outer, last(s.budgets), s.spec)
    s.stage = :finalized_time_step
    ok = r.outer.converged || s.spec.continue_on_nonconvergence
    if !r.outer.converged && s.spec.continue_on_nonconvergence
        @warn "Time step $(r.step.kstp) of stress period $(r.step.kper) did not converge ($(r.outer.iterations) outer iterations); continuing (CONTINUE)"
    end
    return ok
end

"""
    do_time_step(sim::Simulation) -> Bool

`prepare_solve`, up to MXITER `solve` calls, `finalize_solve` (`Mf6DoTimestep`); returns
whether the step converged.
"""
function do_time_step(s::Simulation)
    prepare_solve(s)
    for _ in 1:s.solver.settings.maxiter
        (solve(s) || s.outer.diverged) && break
    end
    return finalize_solve(s)
end

"""
    update(sim::Simulation) -> Bool

One full time step: `prepare_time_step`, `do_time_step`, `finalize_time_step` (libmf6
`update`). Returns the result of `finalize_time_step`.
"""
function update(s::Simulation)
    prepare_time_step(s)
    do_time_step(s)
    return finalize_time_step(s)
end

"""
    finalize(sim::Simulation; free = true) -> sim

Close the output files and, with `free` (the default), release the device memory of the
model, the solver and the budget accumulator now (`free_device_arrays!`, through the
backend hook `free_memory!`) rather than when the garbage collector runs: with two models
on one GPU (T5.4) the memory must be back in the pool before the other model allocates.
Afterwards the simulation cannot be stepped or read (`get_value`, `get_value_ptr` throw);
device arrays obtained before (`get_value_ptr(...; device = true)`) must not be used, while
host mirrors keep their last values. With `free = false` the model stays readable
(`ModflowGPU.run` returns it) and its memory is released by the garbage collector.
"""
function finalize(s::Simulation; free::Bool = true)
    s.stage === :freed && return s
    isnothing(s.output) || close(s.output)
    s.stage = :finalized
    if free
        backend = get_backend(s.model.h)
        backend isa CPU || KernelAbstractions.synchronize(backend)
        free_device_arrays!((s.model, s.solver, s.acc, s.check, s.outer))
        s.stage = :freed
    end
    return s
end

"""
    device_bound!(sim, address) -> device BOUND array

Open the device BOUND array at `address` (`"<model>/<package>/BOUND"`) for writing in the
current time step: the enforced form of the device write contract (T5.2, T5.4). Allowed
only between `prepare_time_step` and `prepare_solve` (otherwise an `ArgumentError`).
Returns the engine array, as `get_value_ptr(sim, address; device = true)`: precision `T`,
`(ncol, maxbound)`, head-like columns as offsets from the reference of the record's cell
(`rrec`), multiplier applied. `prepare_solve` then checks the first `NBOUND` records of
every opened BOUND on the device (one kernel and one host-mapped flag per package): all
values finite, and for RIV stage ≥ bottom and conductance ≥ 0 (`riv_ck`); a violation
throws an `MF6InputError` naming the first bad record. The written values are used by the
step, are replaced by new period data, and are not seen by host reads of BOUND
(`get_value`, mirrors), which keep the values last set on the host. Writes into a device
BOUND that was not opened this way are not checked (the stepping API cannot see them).
"""
function device_bound!(s::Simulation, address::AbstractString)
    _check_readable(s)
    parts = split(uppercase(address), '/')
    (length(parts) == 3 && parts[3] == "BOUND") ||
        throw(ArgumentError("device_bound!: \"$address\" is not a BOUND address (<model>/<package>/BOUND)"))
    s.stage === :prepared_time_step ||
        throw(ArgumentError("device_bound!: device BOUND writes are allowed only between prepare_time_step and prepare_solve (called after $(s.stage))"))
    arr = _variable(s, address).device(s)
    ip = _find_package(s.model, parts[2])
    ip in s.device_writes || push!(s.device_writes, ip)
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
    throw(ArgumentError("the simulation was finalized and its device memory freed (finalize(sim; free = false) keeps it readable)"))

_check_solution(isol) = isol == 1 || throw(ArgumentError("solution $isol: the simulation has one solution (SLN_1)"))

"""
    get_current_time(sim) -> Float64

The simulation time at the end of the current time step (TDIS `totim`; 0 before the first
`prepare_time_step`).
"""
get_current_time(s::Simulation) = s.istep == 0 ? 0.0 : s.steps[s.istep].totim

"""
    get_start_time(sim) -> 0.0
    get_end_time(sim) -> Float64

The start (0) and end (sum of PERLEN) of the simulation.
"""
get_start_time(::Simulation) = 0.0
get_end_time(s::Simulation) = isempty(s.steps) ? 0.0 : last(s.steps).totim

"""
    get_time_step(sim) -> Float64

The length of the current time step (TDIS `delt`, as libmf6's `get_time_step`; before the
first step, that of the first step).
"""
get_time_step(s::Simulation) = isempty(s.steps) ? 0.0 : s.steps[max(s.istep, 1)].delt

# ---------------------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------------------

"""
    get_var_address(sim, var, component; subcomponent_name = "") -> String

The address of variable `var` of `component` (a model name, `"SLN_1"`, `"TDIS"`) and
optional subcomponent (a package name), as libmf6: `"RDE/RIV/BOUND"`.
"""
get_var_address(::Simulation, var::AbstractString, component::AbstractString; subcomponent_name::AbstractString = "") =
    join(filter(!isempty, uppercase.([component, subcomponent_name, var])), "/")

_scalar(x) = [x]

function _dis_names(m::GwfModel)
    names = Set(["DIS"])
    for p in m.spec.packages
        p.ftype in ("DIS6", "DISV6", "DISU6") && push!(names, uppercase(p.name))
    end
    return names
end

_tdis_step(s::Simulation) = s.steps[max(s.istep, 1)]

_find_package(m::GwfModel, name) = findfirst(p -> uppercase(p.spec.name) == name, m.packages)

# The variable at `address` (uppercase), or an error listing the supported addresses.
function _variable(s::Simulation, address::AbstractString)
    parts = split(uppercase(address), '/')
    m = s.model
    T = s.T
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

"""
    var_addresses(sim) -> Vector{String}

Every address the stepping API and the BMI support (writable ones: `input_addresses`).
"""
function var_addresses(s::Simulation)
    m = s.model
    mn = uppercase(m.spec.name)
    a = ["$mn/$v" for v in ("X", "XOLD", "HREF", "IBOUND")]
    append!(a, ["$mn/DIS/$v" for v in ("NODEUSER", "NODEREDUCED", "TOP", "BOT", "AREA", "NODES", "NODESUSER")])
    for p in m.packages
        append!(a, ["$mn/$(uppercase(p.spec.name))/$v" for v in ("BOUND", "NODELIST", "SIMVALS", "NBOUND", "MAXBOUND")])
    end
    push!(a, "SLN_1/MXITER")
    append!(a, ["TDIS/$v" for v in ("DELT", "TOTIM", "PERTIM", "KPER", "KSTP", "NPER")])
    return a
end

"""
    input_addresses(sim) -> Vector{String}

The writable addresses: `"<model>/X"` and `"<model>/<package>/BOUND"`.
"""
input_addresses(s::Simulation) = filter(a -> !isnothing(_variable(s, a).set), var_addresses(s))

"""
    get_value_ptr(sim, address; device = false) -> Array

A host mirror of the variable at `address` (Float64 with absolute heads, Int32 indices;
BOUND as `(ncol, maxbound)`, values before the MULTIPLIER aux variable). The same array
is returned on every call and stays valid until `finalize`; see the header of
src/stepping.jl for when it is synchronised with the device. Writes into the mirror of
`X` or `BOUND` take effect at the next `prepare_time_step` or `prepare_solve`.

With `device = true`: the engine array on the device (precision `T`, head-like values as
offsets from `"<model>/HREF"`, BOUND with the multiplier applied); `X`, `XOLD`, `HREF`,
`IBOUND`, `BOUND`, `NODELIST`, `SIMVALS` only (`HREF` is the host Float64 vector of the
references). Writes into a device BOUND between `prepare_time_step` and `prepare_solve`
are used by the step without checks or conversions, and host reads of BOUND do not see
them (see the header of src/stepping.jl). No host mirror is created.
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
    get_value(sim, address) -> Array
    get_value(sim, address, dest) -> dest

A copy of the current value at `address` on the host (as `get_value_ptr`, but read from the
engine now).
"""
get_value(s::Simulation, address::AbstractString) = (_check_readable(s); _variable(s, address).get(s))
get_value(s::Simulation, address::AbstractString, dest::AbstractArray) = copyto!(dest, get_value(s, address))

"""
    set_value(sim, address, values) -> sim

Set the variable at `address` (`X` in absolute heads, `BOUND` as `(ncol, maxbound)` or
flat, values before the multiplier, head-like columns absolute) and send it to the device
now (`set_bound!`: the conversion of period data). A host mirror of the address is
updated as well.
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
