# CSV output (T6.17): OC BUDGETCSV, IMS CSV_OUTER_OUTPUT and CSV_INNER_OUTPUT, WEL
# AUTO_FLOW_REDUCE_CSV, with MF6's headers and columns.
#
# Ports:
# - Budget.f90 `writecsv` / `write_csv_header` (from gwf.f90 `gwf_ot_bdsummary`, every time
#   step): `time`, the rate in of every budget term, the rate out of every term (names
#   `TERM(PACKAGE)_IN` / `_OUT`), TOTAL_IN, TOTAL_OUT and PERCENT_DIFFERENCE, from the
#   Float64 rates of the step's `ModelBudget`;
# - NumericalSolution.f90 `writeCSVHeader`, the outer-iteration row of `sln_ca` (one row per
#   outer iteration: total inner iterations of the run, time, period, step, outer
#   iteration, inner iterations, `hncg`, the model number and user node of `lrch`; the
#   package column is empty, no supported package has a package convergence test) and
#   `csv_convergence_summary` (one row per inner iteration, from an `InnerTrace`);
# - gwf-wel.f90 `wel_afr_csv_init` / `wel_afr_csv_write` (from `wel_bd_obs`, every time
#   step): a row per record in an active cell whose rate was reduced, `q·mult − rate < 0`.
#
# Numbers are written as MF6's G0 list (`'(*(G0,:,","))'`, Intel build): reals with 16
# significant digits (`obs_format`; zero is `.000000000000000`), the IMS step lengths as
# `g15.7`. The outer CSV costs nothing extra (`hncg` and `lrch` are on the host after the
# outer reduction); the inner CSV costs two argmax reductions per model and inner iteration,
# each with a host read (`InnerTrace`); the WEL CSV copies the package's rates to the host
# once per time step. Runs without CSV output build none of this.

_csv_g0(x::Real) = x == 0 ? ".000000000000000" : obs_format(Float64(x), 16)
_csv_g15(x::Real) = obs_format(Float64(x), 7; lead0 = true)
_csv_row(io::IO, items...) = (join(io, items, ","); println(io))

"""
    CsvOutput

The CSV files of a simulation (T6.17): IMS `CSV_OUTER_OUTPUT` / `CSV_INNER_OUTPUT` of the
solution (`outer`, `inner`, with the `InnerTrace` the inner solves fill), OC `BUDGETCSV`
of each model (`budget`, by model view) and WEL `AUTO_FLOW_REDUCE_CSV` (`afr`: view,
package index, file). `paths` lists the files in the order they were opened. Internal;
built by `initialize` when the input asks for any of them (in every run, as MF6 writes
them, also without `write_output`).
"""
mutable struct CsvOutput
    const outer::Union{Nothing, IOStream}
    const inner::Union{Nothing, IOStream}
    const trace::Union{Nothing, InnerTrace}
    const bcgs::Bool
    const budget::Vector{Union{Nothing, IOStream}}
    const budget_header::Vector{Bool}
    const afr::Vector{Tuple{Int, Int, IOStream}}
    const paths::Vector{String}
    # per model view: MF6's model number, its rows in the solution and its user nodes
    const ids::Vector{Int}
    const ranges::Vector{UnitRange{Int}}
    const nodeuser::Vector{Vector{Int32}}
    const names::Vector{String}
    # total inner iterations of the run (`itertot_sim`)
    itertot::Int
end

"""
    has_csv_output(sim::SimulationSpec) -> Bool

Whether the input asks for a CSV file: IMS CSV_OUTER_OUTPUT or CSV_INNER_OUTPUT, OC
BUDGETCSV, WEL AUTO_FLOW_REDUCE_CSV.
"""
function has_csv_output(sim::SimulationSpec)
    ims = only(only(sim.solution_groups).solutions)
    (isempty(ims.csv_outer_output) && isempty(ims.csv_inner_output)) || return true
    return any(sim.models) do spec
        (!isnothing(spec.oc) && !isempty(spec.oc.budgetcsv_file)) || any(b -> !isempty(b.afrcsv_file), spec.boundaries)
    end
end

"""
    CsvOutput(sim::SimulationSpec, m::GwfModel, ps::PicardSolver, outdir) -> CsvOutput or nothing

Open the CSV files the input of `sim` asks for (`nothing` when none), under `outdir` (the
names as given, relative to the simulation directory), and write the headers MF6 writes
before the first time step (`writeCSVHeader`, `wel_afr_csv_init`); the budget CSV header
follows with the first row (`writecsv`).
"""
function CsvOutput(sim::SimulationSpec, m::GwfModel, ps::PicardSolver, outdir::AbstractString)
    has_csv_output(sim) || return nothing
    ims = only(only(sim.solution_groups).solutions)
    views = model_views(m)
    names = String[_vname(v) for v in views]
    ids = Int[something(findfirst(s -> s.name == n, sim.models), k) for (k, n) in enumerate(names)]
    ranges = UnitRange{Int}[_vrange(v) for v in views]
    nodeuser = Vector{Int32}[_vconn(v).nodeuser for v in views]
    paths = String[]
    ios = IOStream[]
    function open_csv(name)
        p = _output_path(sim.simdir, outdir, name)
        io = open(p, "w")
        push!(paths, p)
        push!(ios, io)
        return io
    end
    try
        outer = isempty(ims.csv_outer_output) ? nothing : open_csv(ims.csv_outer_output)
        inner = isempty(ims.csv_inner_output) ? nothing : open_csv(ims.csv_inner_output)
        bcgs = ims.linear_acceleration !== :cg
        isnothing(outer) || _csv_row(
            outer, "total_inner_iterations", "totim", "kper", "kstp", "nouter", "inner_iterations", "solution_outer_dvmax",
            "solution_outer_dvmax_model", "solution_outer_dvmax_package", "solution_outer_dvmax_node",
        )
        trace = nothing
        if !isnothing(inner)
            cols = Any[
                "total_inner_iterations", "totim", "kper", "kstp", "nouter", "ninner", "solution_inner_dvmax",
                "solution_inner_dvmax_model", "solution_inner_dvmax_node", "solution_inner_rmax", "solution_inner_rmax_model",
                "solution_inner_rmax_node", "solution_inner_alpha",
            ]
            bcgs && push!(cols, "solution_inner_omega")
            if length(views) > 1
                for n in names
                    u = uppercase(n)
                    append!(cols, (u * "_inner_dvmax", u * "_inner_dvmax_node", u * "_inner_rmax", u * "_inner_rmax_node"))
                end
            end
            _csv_row(inner, cols...)
            trace = InnerTrace(m.backend, ranges, eltype(ps.krylov.sa))
        end
        budget = Union{Nothing, IOStream}[]
        afr = Tuple{Int, Int, IOStream}[]
        for (k, v) in enumerate(views)
            oc = _vspec(v).oc
            push!(budget, (isnothing(oc) || isempty(oc.budgetcsv_file)) ? nothing : open_csv(oc.budgetcsv_file))
            for ip in _vpackages(v)
                pkg = m.packages[ip]
                (boundary_kind(pkg) === :WEL && !isempty(pkg.spec.afrcsv_file)) || continue
                io = open_csv(pkg.spec.afrcsv_file)
                println(io, "time,period,step,boundnumber,cellnumber,rate-requested,rate-actual,wel-reduction")
                push!(afr, (k, ip, io))
            end
        end
        foreach(flush, ios)
        return CsvOutput(outer, inner, trace, bcgs, budget, fill(false, length(views)), afr, paths, ids, ranges, nodeuser, names, 0)
    catch
        foreach(close, ios)
        rethrow()
    end
end

function Base.close(c::CsvOutput)
    for io in (c.outer, c.inner, c.budget...)
        isnothing(io) || close(io)
    end
    foreach(a -> close(a[3]), c.afr)
    return nothing
end

# MF6's model number and user node of solution row `row` (`sln_get_nodeu`); (0, 0) for 0.
function _csv_location(c::CsvOutput, row::Integer)
    row > 0 || return (0, 0)
    for (k, rg) in enumerate(c.ranges)
        row in rg && return (c.ids[k], Int(c.nodeuser[k][row - first(rg) + 1]))
    end
    return (0, 0)
end

"""
    csv_outer_iteration!(c::CsvOutput, state::OuterState, step::TimeStep, inner0)

After an outer iteration (`state`, the iteration just done; `inner0` the inner iterations
of the step before it): the row of the outer CSV and the rows of the inner CSV of its
inner iterations (the end of `sln_ca`).
"""
function csv_outer_iteration!(c::CsvOutput, state::OuterState, step::TimeStep, inner0::Integer)
    iter = state.inner - inner0
    icsv0 = c.itertot + 1
    c.itertot += iter
    if !isnothing(c.outer)
        hncg = state.hncg
        mid, node = hncg != 0 && !isnan(hncg) ? _csv_location(c, state.lrch) : (0, 0)
        _csv_row(c.outer, c.itertot, _csv_g0(step.totim), step.kper, step.kstp, state.kiter, iter, _csv_g0(hncg), mid, "", node)
        flush(c.outer)
    end
    t = c.trace
    if !isnothing(c.inner) && !isnothing(t)
        nm = length(t.ranges)
        io = c.inner
        for k in eachindex(t.alpha)
            # The solution summary: the first model with the largest magnitude.
            dv = r = 0.0
            ldv = lr = 0
            for j in 1:nm
                i = (k - 1) * nm + j
                abs(t.dv[i]) > abs(dv) && ((dv, ldv) = (t.dv[i], t.ldv[i]))
                abs(t.r[i]) > abs(r) && ((r, lr) = (t.r[i], t.lr[i]))
            end
            dv == 0 && (ldv = 0)
            r == 0 && (lr = 0)
            mdv, ndv = _csv_location(c, ldv)
            mr, nr = _csv_location(c, lr)
            acc = c.bcgs ? _csv_g15(t.alpha[k]) * "," * _csv_g15(t.omega[k]) : _csv_g15(t.alpha[k])
            join(io, (icsv0 + k - 1, _csv_g0(step.totim), step.kper, step.kstp, state.kiter, k, _csv_g0(dv), mdv, ndv, _csv_g0(r), mr, nr, acc), ",")
            if nm > 1
                for j in 1:nm
                    i = (k - 1) * nm + j
                    print(io, ",", _csv_g0(t.dv[i]), ",", last(_csv_location(c, t.ldv[i])), ",", _csv_g0(t.r[i]), ",", last(_csv_location(c, t.lr[i])))
                end
            end
            println(io)
        end
        flush(io)
    end
    isnothing(t) || empty!(t)
    return nothing
end

"""
    write_budget_csv(io, b::ModelBudget, totim; header = false)

One row of the budget CSV (Budget.f90 `writecsv`; with `header`, `write_csv_header`
first): the time, the rates in, the rates out, their totals and the percent difference.
"""
function write_budget_csv(io::IO, b::ModelBudget, totim::Float64; header::Bool = false)
    if header
        print(io, "time,")
        foreach(t -> print(io, t.name, "(", t.package, ")_IN,"), b.terms)
        foreach(t -> print(io, t.name, "(", t.package, ")_OUT,"), b.terms)
        println(io, "TOTAL_IN,TOTAL_OUT,PERCENT_DIFFERENCE")
    end
    totrin = totrout = 0.0
    for t in b.terms
        totrin += t.rate_in
        totrout += t.rate_out
    end
    avg = (totrin + totrout) / 2
    pdiff = avg != 0 ? 100 * (totrin - totrout) / avg : 0.0
    print(io, _csv_g0(totim))
    foreach(t -> print(io, ",", _csv_g0(t.rate_in)), b.terms)
    foreach(t -> print(io, ",", _csv_g0(t.rate_out)), b.terms)
    println(io, ",", _csv_g0(totrin), ",", _csv_g0(totrout), ",", _csv_g0(pdiff))
    flush(io)
    return nothing
end

"""
    write_afr_csv(io, pkg::BoundaryPackage{:WEL}, m::GwfModel, step::TimeStep)

The AUTO_FLOW_REDUCE_CSV rows of a time step (gwf-wel.f90 `wel_afr_csv_write`): per
record in an active cell with `q·mult − rate < 0` (the WEL-REDUCTION of the record), the
time, period, step, record number, user node, requested rate, actual rate and reduction.
The rates are copied from the device once.
"""
function write_afr_csv(io::IO, pkg::BoundaryPackage, m::GwfModel, step::TimeStep)
    nb = pkg.nbound
    if nb > 0
        q = Array(pkg.bound)
        sv = Array(pkg.simvals)
        for i in 1:nb
            n = Int(pkg.nodelist_host[i])
            (n > 0 && m.ibound_host[n] > 0) || continue
            qi = Float64(q[1, i])
            si = Float64(sv[i])
            v = qi - si
            v < 0 || continue
            u = Int(pkg.conn.nodeuser[n]) - Int(pkg.uoffset)
            _csv_row(io, _csv_g0(step.totim), step.kper, step.kstp, i, u, _csv_g0(qi), _csv_g0(si), _csv_g0(v))
        end
    end
    flush(io)
    return nothing
end

"""
    csv_time_step!(c::CsvOutput, m::GwfModel, budgets, step::TimeStep)

The CSV output at the end of a time step: the budget CSV rows of the models (`budgets`,
the step's `ModelBudget` of each model view) and the WEL AUTO_FLOW_REDUCE_CSV rows.
"""
function csv_time_step!(c::CsvOutput, m::GwfModel, budgets::AbstractVector{ModelBudget}, step::TimeStep)
    for (k, io) in enumerate(c.budget)
        isnothing(io) && continue
        write_budget_csv(io, budgets[k], step.totim; header = !c.budget_header[k])
        c.budget_header[k] = true
    end
    for (_, ip, io) in c.afr
        write_afr_csv(io, m.packages[ip], m, step)
    end
    return nothing
end
