# Multi-model solutions at run time (T6.6): the flows of the GWF-GWF exchanges, the budget
# of each model (with the exchange terms MF6 adds) and the exchange records of each
# model's budget file. The merged model and its setup are in src/gwf/exchange.jl.
#
# Ports exg-gwfgwf.f90 `gwf_gwf_cq` (`gwf_gwf_calc_simvals`: the flow of each exchange
# connection, into the model-1 cell, zero when a cell is inactive), `gwf_gwf_bd`
# (`rate_accumulator` of the flows: FLOW-JA-FACE in model 1, the same rates swapped in
# model 2), `gwf_gwf_chd_bd` (FLOW-JA-FACE-CHD: the flows into constant-head cells, which
# the CHD package of a model does not see) and `gwf_gwf_bdsav_model` (the IMETH 6 list of
# each model, from its side: `id1` its user node, `id2` the other model's, `q` the flow
# into its cell, the exchange's auxiliary values).

"""
    ExchangeFlows

The exchange flows of a multi-model solution: device positions `pos` of every exchange
connection in the merged FLOW-JA-FACE (all exchanges, in order), the gathered values `vals`
(device, `T`), `host` (Float64, MF6's SIMVALS of the last `exchange_flows!`) and per
exchange its `ranges` in them.
"""
struct ExchangeFlows{VI, V}
    pos::VI
    vals::V
    host::Vector{Float64}
    ranges::Vector{UnitRange{Int}}
end

function ExchangeFlows(m::GwfModel{T}) where {T}
    sol = m.solution
    pos = Int32[]
    ranges = UnitRange{Int}[]
    for ex in sol.exchanges
        push!(ranges, (length(pos) + 1):(length(pos) + length(ex.pos)))
        append!(pos, ex.pos)
    end
    return ExchangeFlows(_device_copy(m.backend, pos), KernelAbstractions.zeros(m.backend, T, length(pos)), zeros(length(pos)), ranges)
end

Adapt.adapt_structure(to, x::ExchangeFlows) = ExchangeFlows(adapt(to, x.pos), adapt(to, x.vals), copy(x.host), x.ranges)

"""
    exchange_flows!(x::ExchangeFlows, m::GwfModel) -> x

After `calc_flows!`: gather the flow of every exchange connection from the merged
FLOW-JA-FACE (one kernel, one copy to the host; `gwf_gwf_calc_simvals`).
"""
function exchange_flows!(x::ExchangeFlows, m::GwfModel)
    isempty(x.host) && return x
    obs_gather!(x.vals, m.flowja, x.pos, 0)
    x.host .= Float64.(Array(x.vals))
    return x
end

exchange_simvals(x::ExchangeFlows, ie::Integer) = view(x.host, x.ranges[ie])

# The budget rows of the exchanges in a model's budget, in MF6's order: per exchange (in
# order) FLOW-JA-FACE for the model as model 1, then as model 2, then FLOW-JA-FACE-CHD
# likewise (a periodic exchange gives each row twice). (exchange index, side 1 or 2, chd).
function _exchange_rows(sol::SolutionSetup, k::Integer)
    rows = Tuple{Int, Int, Bool}[]
    for (ie, ex) in enumerate(sol.exchanges)
        ex.m1 == k && push!(rows, (ie, 1, false))
        ex.m2 == k && push!(rows, (ie, 2, false))
        ex.m1 == k && push!(rows, (ie, 1, true))
        ex.m2 == k && push!(rows, (ie, 2, true))
    end
    return rows
end

"""
    BudgetAccumulator(v::ModelView)

The budget accumulator of one model of a multi-model solution (T6.6): STO-SS when the
model has STO, its stress packages, then the exchange rows (`FLOW-JA-FACE` and
`FLOW-JA-FACE-CHD`, package name the exchange's).
"""
function BudgetAccumulator(v::ModelView)
    mm = v.model
    T = eltype(mm)
    spec = _vspec(v)
    names = String[]
    packages = String[]
    if !isnothing(spec.sto)
        push!(names, "STO-SS")
        push!(packages, "STORAGE")
    end
    for ip in _vpackages(v)
        pkg = mm.packages[ip]
        push!(names, strip(budget_text(pkg)))
        push!(packages, uppercase(pkg.spec.name))
    end
    for (ie, _, chd) in _exchange_rows(mm.solution, v.index)
        push!(names, chd ? "FLOW-JA-FACE-CHD" : "FLOW-JA-FACE")
        push!(packages, mm.solution.exchanges[ie].name)
    end
    len = max(Int(v.part.conn.nodes), maximum((length(mm.packages[ip].nodelist_host) for ip in _vpackages(v)); init = 0))
    nt = length(names)
    return BudgetAccumulator(names, packages, zeros(nt), zeros(nt), Reducer(mm.backend, len, accumulation_type(mm.backend, T)))
end

"""
    model_budget!(acc::BudgetAccumulator, v::ModelView, step, x::ExchangeFlows) -> ModelBudget

The budget of one model of a multi-model solution after `calc_flows!` and
`exchange_flows!`: storage over the model's cells and its packages as `model_budget!` of a
single model (device reductions), the exchange rows from the exchange flows on the host
(`gwf_gwf_bd`, `gwf_gwf_chd_bd`).
"""
function model_budget!(acc::BudgetAccumulator, v::ModelView, step::TimeStep, x::ExchangeFlows)
    mm = v.model
    backend = mm.backend
    red = acc.reducer
    rates = Tuple{Float64, Float64}[]
    if !isnothing(_vspec(v).sto)
        rin, rout, _, _ = _mapreduce!(red, backend, _inout_term, _SUM4, Int(v.part.conn.nodes), view(mm.sto.strgss, _vrange(v)))
        push!(rates, (rin, rout))
    end
    for ip in _vpackages(v)
        pkg = mm.packages[ip]
        if pkg.nbound == 0
            push!(rates, (0.0, 0.0))
        elseif boundary_kind(pkg) === :CHD
            rin, rout, _, _ = _mapreduce!(red, backend, _sum2_term, _SUM4, pkg.nbound, pkg.ratein, pkg.rateout)
            push!(rates, (rin, rout))
        else
            rin, rout, _, _ = _mapreduce!(red, backend, _inout_term, _SUM4, pkg.nbound, pkg.simvals)
            push!(rates, (rin, rout))
        end
    end
    sol = mm.solution
    for (ie, side, chd) in _exchange_rows(sol, v.index)
        ex = sol.exchanges[ie]
        q = exchange_simvals(x, ie)
        rin = rout = 0.0
        if chd
            # gwf_gwf_chd_bd: flows into the constant-head cells of this side.
            g = side == 1 ? ex.g1 : ex.g2
            for i in eachindex(q)
                mm.ibound_host[g[i]] < 0 || continue
                qi = side == 1 ? q[i] : -q[i]
                qi > 0.0 ? (rout += qi) : (rin -= qi)
            end
        else
            # rate_accumulator of the flows into model 1; swapped for model 2.
            for qi in q
                qi < 0.0 ? (rout -= qi) : (rin += qi)
            end
            side == 2 && ((rin, rout) = (rout, rin))
        end
        push!(rates, (rin, rout))
    end
    terms = BudgetTerm[]
    for (i, (rin, rout)) in enumerate(rates)
        acc.cum_in[i] += rin * step.delt
        acc.cum_out[i] += rout * step.delt
        push!(terms, BudgetTerm(acc.names[i], acc.packages[i], rin, rout, acc.cum_in[i], acc.cum_out[i]))
    end
    return ModelBudget(step.kstp, step.kper, terms, _vspec(v).name)
end

"""
    write_exchange_budget(o::ModelOutput, v::ModelView, step, x::ExchangeFlows)

The IMETH 6 lists of the exchanges of model `v` with SAVE_FLOWS (exg-gwfgwf.f90
`gwf_gwf_bdsav_model`), after the model's own records, in exchange order (a periodic
exchange: from the model-1 side, then from the model-2 side): text `FLOW-JA-FACE`, model and
package names `<model> <exchange> <other model> <exchange>`, per connection `id1` the
model's user node, `id2` the other's, `q` the flow into the model's cell (zero when a cell
is inactive), and the exchange's auxiliary values.
"""
function write_exchange_budget(o::ModelOutput, v::ModelView, step::TimeStep, x::ExchangeFlows)
    io = o.budget
    isnothing(io) && return nothing
    mm = v.model
    sol = mm.solution
    kw = (; kstp = step.kstp, kper = step.kper, delt = step.delt, pertim = step.pertim, totim = step.totim)
    for (ie, ex) in enumerate(sol.exchanges)
        ex.spec.save_flows || continue
        q = exchange_simvals(x, ie)
        d = ex.spec.data
        p1, p2 = sol.models[ex.m1], sol.models[ex.m2]
        u1 = Int32[p1.conn.nodeuser[n] for n in ex.node1]
        u2 = Int32[p2.conn.nodeuser[n] for n in ex.node2]
        aux = Matrix{Float64}(d.auxvar)
        for side in (1, 2)
            (side == 1 ? ex.m1 : ex.m2) == v.index || continue
            list = side == 1 ? BudgetList(u1, u2, collect(q), aux) : BudgetList(u2, u1, -collect(q), aux)
            me, other = side == 1 ? (p1, p2) : (p2, p1)
            write_budget_list(
                io, o.layout, "    FLOW-JA-FACE", list; kw..., modelnam1 = me.spec.name, paknam1 = ex.name,
                modelnam2 = other.spec.name, paknam2 = ex.name, auxtxt = ex.spec.auxnames,
            )
        end
    end
    return nothing
end

"""
    ExchangeObservations

The observation output of the GWF-GWF exchanges with OBS6 (T6.6; exg-gwfgwf.f90
`gwf_gwf_rp_obs`, `gwf_gwf_save_simvals`): the output files (`paths`) and per observation
its exchange (`exg`) and exchange connections (`idx`: the connection given, or every
connection with the boundary name). The value of an observation is the sum of their
flows (FLOW-JA-FACE, into the model-1 cell; MF6 accumulates over the boundaries).
"""
struct ExchangeObservations
    files::Vector{_ObsFile}
    paths::Vector{String}
    exg::Vector{Int}
    idx::Vector{Vector{Int}}
end

function ExchangeObservations(sol::SolutionSetup, simdir::AbstractString, outdir::AbstractString)
    any(ex -> !isnothing(ex.spec.obs), sol.exchanges) || return nothing
    exg = Int[]
    idx = Vector{Int}[]
    opens = Tuple{ObsOutputSpec, Int, Vector{Int}}[]
    for (ie, ex) in enumerate(sol.exchanges)
        os = ex.spec.obs
        isnothing(os) && continue
        for o in os.outputs
            cols = Int[]
            for e in o.observations
                push!(exg, ie)
                push!(idx, isempty(e.boundname) ? [e.cells[1]] : findall(==(e.boundname), ex.spec.data.boundname))
                push!(cols, length(exg))
            end
            push!(opens, (o, os.digits, cols))
        end
    end
    files = _ObsFile[_obs_open(simdir, outdir, o, digits, cols) for (o, digits, cols) in opens]
    return ExchangeObservations(files, [f.path for f in files], exg, idx)
end

Base.close(o::ExchangeObservations) = foreach(f -> close(f.io), o.files)

"""
    write_exchange_observations(o::ExchangeObservations, x::ExchangeFlows, totim)

Append the exchange observations of the time step that ends at `totim` (the flows of the
last `exchange_flows!`) to their files, as `write_observations`.
"""
function write_exchange_observations(o::ExchangeObservations, x::ExchangeFlows, totim::Float64)
    v = [sum(exchange_simvals(x, o.exg[i])[o.idx[i]]; init = 0.0) for i in eachindex(o.exg)]
    for f in o.files
        if f.binary
            write(f.io, totim)
            foreach(i -> write(f.io, v[i]), f.cols)
        else
            print(f.io, obs_format(totim, 13; lead0 = true))
            foreach(i -> print(f.io, ",", _obs_value_text(v[i], f.fmt)), f.cols)
            println(f.io)
        end
        flush(f.io)
    end
    return v
end
