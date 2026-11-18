# Flows and the volumetric budget of a GWF model (gwf.f90 `gwf_cq`, `gwf_bd`; Budget.f90).
#
# `calc_flows!` (gwf_cq, then the `csr_diagsum` of gwf_bd), on the device in `T`:
# 1. `flowja` off-diagonals from the offset heads, `C_nm·(h'_m − h'_n + Δr_nm)` (npf_cq;
#    positive into cell n, zero when either cell has ibound 0), diagonals zeroed;
# 2. storage: `strgss = rho·(hold − h)` (and for convertible cells the `SsTerms` rate and
#    the STO-SY rate `strgsy`, T6.1) added to the diagonal (sto_cq);
# 3. each stress package in MF6's order: `bnd_cf` with the final heads, then `simvals`
#    (bnd_cq_simrate) added to the diagonal; CHD: `calc_chd_rate` (net rate per cell into
#    the model, its split over connections to active cells, and the diagonal);
# 4. the diagonal plus the sum of the off-diagonals (csr_diagsum): the cell's imbalance.
#
# `model_budget!` (gwf_bd, `rate_accumulator`, `budget%addentry`): per term the sum of the
# positive and of the negated negative rates, as device reductions accumulated in Float64
# where the backend has it (PLAN.md section 3.7, rule 4); the cumulative volumes
# (`rate·delt`) and the percent discrepancy on the host in Float64.

# One work item per entry (T4.5; coalesced writes): row `irow[ii]`, column `ja[ii]`.
@kernel function npf_flowja_kernel!(flowja, @Const(ia), @Const(ja), @Const(jas), @Const(irow), @Const(condsat), @Const(Δr), @Const(h), @Const(ibound))
    ii = @index(Global)
    T = eltype(flowja)
    @inbounds begin
        n = irow[ii]
        if ii == ia[n]
            flowja[ii] = zero(T)
        else
            m = ja[ii]
            _, _, _, q, _, _ = connection_terms(condsat, Δr, jas, ibound, h, n, m, ii)
            flowja[ii] = q
        end
    end
end

"""
    npf_flowja!(flowja, csr::CsrStructure, npf::NpfModel, h, ibound) -> flowja

Connection flows of the offset heads `h` (`npf_cq`, `sgwf_npf_qcalc`): off-diagonal
entry `ii` of row `n` gets `C·(h'_m − h'_n + Δr_nm)` (flow into `n`; zero when either
cell is inactive), the diagonal entries are zeroed. Antisymmetric exactly. `C` is `condsat`,
or for convertible cells (T6.1) the conductance from the saturations of the last
`npf_cf!` and the heads `h` (PERCHED: `C·(bot'_lo − h'_lo)` into the upper cell `lo`).
"""
function npf_flowja!(flowja, csr::CsrStructure, npf::NpfModel, h, ibound)
    npf_flowja_kernel!(get_backend(flowja), _WG)(
        flowja, csr.ia, csr.ja, csr.jas, csr.irow, _cond_arg(npf), npf.Δr, h, ibound; ndrange = csr.nja,
    )
    return flowja
end

@kernel function boundary_flowja_kernel!(flowja, @Const(ia), @Const(simvals), @Const(nodeptr), @Const(noderec))
    n = @index(Global)
    @inbounds begin
        k1 = nodeptr[n]
        k2 = nodeptr[n + 1] - Int32(1)
        if k2 >= k1
            s = flowja[ia[n]]
            for k in k1:k2
                s += simvals[noderec[k]]
            end
            flowja[ia[n]] = s
        end
    end
end

@kernel function diagsum_kernel!(flowja, @Const(ia))
    n = @index(Global)
    @inbounds begin
        id = ia[n]
        s = flowja[id]
        for ii in (id + Int32(1)):(ia[n + 1] - Int32(1))
            s += flowja[ii]
        end
        flowja[id] = s
    end
end

"""
    calc_flows!(m::GwfModel) -> m

All flows of the model after a converged (or final) outer iteration (`gwf_cq` and the
`csr_diagsum` of `gwf_bd`; see the header of src/budget.jl): `m.flowja`, `m.sto.strgss`,
the `simvals` of every stress package (CHD: also `ratein`, `rateout`). Device work only.
"""
function calc_flows!(m::GwfModel)
    backend = m.backend
    csr = m.csr
    npf_flowja!(m.flowja, csr, m.npf, m.h, m.ibound)
    if !isnothing(m.sto)
        calc_storage_rates!(m.sto, m.h, m.hold, m.ibound)
        add_storage_flowja!(m.flowja, csr.idiag, m.sto)
    end
    for pkg in m.packages
        if boundary_kind(pkg) === :CHD
            calc_chd_rate!(pkg, m.flowja, csr.ia, csr.ja, m.ibound)
        elseif pkg.nbound > 0
            formulate!(pkg, m.h, m.ibound, m.area; kind = record_kind(pkg, m.npf.cv; newton = m.newton))
            calc_simvals!(pkg, m.h, m.ibound)
            boundary_flowja_kernel!(backend, _WG)(m.flowja, csr.ia, pkg.simvals, pkg.nodeptr, pkg.noderec; ndrange = csr.nodes)
        end
    end
    diagsum_kernel!(backend, _WG)(m.flowja, csr.ia; ndrange = csr.nodes)
    return m
end

# --- budget terms

"""
    BudgetTerm

One row of a [`ModelBudget`](@ref). Fields:
- `name`: the budget text, e.g. `"STO-SS"`, `"CHD"`, `"RCHA"`;
- `package`: the package name as in the listing file, e.g. `"STORAGE"`, `"CHD_0"`;
- `rate_in`, `rate_out`: the inflow and outflow rates of the time step (non-negative,
  Float64, volume per time, L³/T in the model's units);
- `cum_in`, `cum_out`: the cumulative inflow and outflow volumes since the start (L³).
"""
struct BudgetTerm
    name::String
    package::String
    rate_in::Float64
    rate_out::Float64
    cum_in::Float64
    cum_out::Float64
end

"""
    ModelBudget

The volumetric budget of a model for one time step, as in MF6's listing file: `kstp` and
`kper` (time step and stress period numbers) and the `terms` ([`BudgetTerm`](@ref)) in
MF6's order (STO-SS, STO-SY when ICONVERT is used, then the stress packages). Computed on
the device in every time step, also when no budget is printed or saved; found in
`RunResult.budgets` and `Simulation.budgets`.

[`percent_discrepancy(b)`](@ref percent_discrepancy) and
[`cum_percent_discrepancy(b)`](@ref cum_percent_discrepancy) give MF6's percent
discrepancy of the rates and of the cumulative volumes.

# Example
```julia
b = run_simulation("path/to/simulation").budgets[end]
[(t.name, t.rate_in, t.rate_out) for t in b.terms]
ModflowGPU.percent_discrepancy(b)
```
"""
struct ModelBudget
    kstp::Int
    kper::Int
    terms::Vector{BudgetTerm}
end

total_rate_in(b::ModelBudget) = sum(t -> t.rate_in, b.terms; init = 0.0)
total_rate_out(b::ModelBudget) = sum(t -> t.rate_out, b.terms; init = 0.0)
total_cum_in(b::ModelBudget) = sum(t -> t.cum_in, b.terms; init = 0.0)
total_cum_out(b::ModelBudget) = sum(t -> t.cum_out, b.terms; init = 0.0)
percent_discrepancy(b::ModelBudget) = percent_discrepancy(total_rate_in(b), total_rate_out(b))

"""
    ModflowGPU.cum_percent_discrepancy(b::ModelBudget) -> Float64

MF6's percent discrepancy (see [`percent_discrepancy`](@ref)) of the cumulative volumes
of budget `b` since the start of the simulation.
"""
cum_percent_discrepancy(b::ModelBudget) = percent_discrepancy(total_cum_in(b), total_cum_out(b))

@inline function _inout_term(::Type{Ta}, i, q) where {Ta}
    v = @inbounds Ta(q[i])
    return v > zero(Ta) ? (v, zero(Ta), zero(Ta), zero(Ta)) : (zero(Ta), -v, zero(Ta), zero(Ta))
end

@inline _sum2_term(::Type{Ta}, i, a, b) where {Ta} = @inbounds (Ta(a[i]), Ta(b[i]), zero(Ta), zero(Ta))

"""
    BudgetAccumulator

Host state of a model's budget over the run: the term names and packages (fixed by the
model's packages), the cumulative in/out volumes (Float64) and a device `Reducer` for the
rate sums.

    BudgetAccumulator(m::GwfModel)
"""
struct BudgetAccumulator{R <: Reducer}
    names::Vector{String}
    packages::Vector{String}
    cum_in::Vector{Float64}
    cum_out::Vector{Float64}
    reducer::R
end

# Adapt: the reducer moves; the cumulative volumes are copied (they change during a run).
Adapt.adapt_structure(to, a::BudgetAccumulator) =
    BudgetAccumulator(a.names, a.packages, copy(a.cum_in), copy(a.cum_out), adapt(to, a.reducer))

"""
    budget_text(pkg::BoundaryPackage) -> String

The 16-character budget text of a stress package as MF6 writes it in the budget file
(`text`/`texta` of the package module; RCHA and EVTA for READASARRAYS, where EVTA is
`"           EVTA "` in MF6 6.8).
"""
function budget_text(pkg::BoundaryPackage{K}) where {K}
    K === :EVT && pkg.spec.readasarrays && return "           EVTA "
    K === :RCH && pkg.spec.readasarrays && return "            RCHA"
    return lpad(String(K), 16)
end

_uses_sy(m::GwfModel) = !isnothing(m.sto) && !isnothing(m.sto.conv)

function BudgetAccumulator(m::GwfModel{T}) where {T}
    names = String[]
    packages = String[]
    if !isnothing(m.sto)
        push!(names, "STO-SS")
        push!(packages, "STORAGE")
        if _uses_sy(m)
            push!(names, "STO-SY")
            push!(packages, "STORAGE")
        end
    end
    for pkg in m.packages
        push!(names, strip(budget_text(pkg)))
        push!(packages, uppercase(pkg.spec.name))
    end
    len = max(Int(m.conn.nodes), maximum((length(p.nodelist_host) for p in m.packages); init = 0))
    Ta = accumulation_type(m.backend, T)
    nt = length(names)
    return BudgetAccumulator(names, packages, zeros(nt), zeros(nt), Reducer(m.backend, len, Ta))
end

"""
    model_budget!(acc::BudgetAccumulator, m::GwfModel, step::TimeStep) -> ModelBudget

The budget of the time step after `calc_flows!` (`gwf_bd`: `sto_bd`, `bnd_bd`;
`rate_accumulator`): per term the sums of the positive and of the negated negative rates
(device reductions; for CHD the `ratein`/`rateout` split of `calc_chd_rate`), and the
cumulative volumes updated with `rate·delt` on the host.
"""
function model_budget!(acc::BudgetAccumulator, m::GwfModel, step::TimeStep)
    backend = m.backend
    red = acc.reducer
    rates = Tuple{Float64, Float64}[]
    if !isnothing(m.sto)
        rin, rout, _, _ = _mapreduce!(red, backend, _inout_term, _SUM4, Int(m.conn.nodes), m.sto.strgss)
        push!(rates, (rin, rout))
        if _uses_sy(m)
            rin, rout, _, _ = _mapreduce!(red, backend, _inout_term, _SUM4, Int(m.conn.nodes), m.sto.strgsy)
            push!(rates, (rin, rout))
        end
    end
    for pkg in m.packages
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
    terms = BudgetTerm[]
    for (i, (rin, rout)) in enumerate(rates)
        acc.cum_in[i] += rin * step.delt
        acc.cum_out[i] += rout * step.delt
        push!(terms, BudgetTerm(acc.names[i], acc.packages[i], rin, rout, acc.cum_in[i], acc.cum_out[i]))
    end
    return ModelBudget(step.kstp, step.kper, terms)
end
