# Output control and the output of a model run (gwf-oc.f90, OutputControl.f90,
# PrintSaveManager.f90, TimeStepSelect.f90; gwf.f90 `gwf_ot`): heads (HDS), cell-by-cell
# flows (CBC) and a short listing file with the budget tables. Host only: the heads are
# absolute Float64 values `r + h'` (PLAN.md section 3.7, rule 1).

"""
    oc_selected(oc, action, rtype, step::TimeStep) -> Bool

Whether output control `oc` (`OcSpec` or `nothing`) selects `action` (`:save`, `:print`)
of `rtype` (`"HEAD"`, `"BUDGET"`) in time step `step` (`oc_save`/`oc_print`,
`TimeStepSelect` `is_selected`): the records of the last PERIOD block at or before the
step's period (none before the first block; an empty block selects nothing), with
ALL, FIRST (`kstp == 1`), LAST (end of period), FREQUENCY n (`kstp mod n == 0`) or STEPS.
"""
function oc_selected(oc::Union{Nothing, OcSpec}, action::Symbol, rtype::AbstractString, step::TimeStep)
    isnothing(oc) && return false
    ib = period_block(oc.periods, step.kper)
    ib == 0 && return false
    for rec in oc.periods.data[ib]
        (rec.action === action && rec.rtype == rtype) || continue
        _step_selected(rec, step) && return true
    end
    return false
end

function _step_selected(rec::OcRecord, step::TimeStep)
    s = rec.setting
    s === :all && return true
    s === :first && return step.kstp == 1
    s === :last && return step.endofperiod
    s === :frequency && return !isempty(rec.steps) && rec.steps[1] > 0 && mod(step.kstp, rec.steps[1]) == 0
    s === :steps && return step.kstp in rec.steps
    return false
end

"""
    ModelOutput

The open output files of a model run: `head` and `budget` (`IO` or `nothing` when OC
names no file), `listing` (`IO`), the `layout` (`OutputLayout`) and the paths.

    ModelOutput(m::GwfModel, simdir, outdir)
"""
struct ModelOutput
    head::Union{Nothing, IO}
    budget::Union{Nothing, IO}
    listing::IO
    layout::OutputLayout
    head_path::String
    budget_path::String
    listing_path::String
end

function _output_path(simdir, outdir, name)
    rel = isabspath(name) ? relpath(name, simdir) : name
    p = joinpath(outdir, rel)
    mkpath(dirname(p))
    return p
end

function ModelOutput(m::GwfModel, simdir::AbstractString, outdir::AbstractString)
    oc = m.spec.oc
    hp = (isnothing(oc) || isempty(oc.head_file)) ? "" : _output_path(simdir, outdir, oc.head_file)
    bp = (isnothing(oc) || isempty(oc.budget_file)) ? "" : _output_path(simdir, outdir, oc.budget_file)
    lname = isempty(m.spec.list_file) ? splitext(basename(m.spec.file))[1] * ".lst" : m.spec.list_file
    lp = _output_path(simdir, outdir, lname)
    m.spec.npf.save_specific_discharge && !isempty(bp) &&
        @info "NPF SAVE_SPECIFIC_DISCHARGE: DATA-SPDIS is not written to the budget file yet" m.spec.npf.file
    head = isempty(hp) ? nothing : open(hp, "w")
    budget = isempty(bp) ? nothing : open(bp, "w")
    return ModelOutput(head, budget, open(lp, "w"), OutputLayout(m.spec.dis), hp, bp, lp)
end

function Base.close(o::ModelOutput)
    isnothing(o.head) || close(o.head)
    isnothing(o.budget) || close(o.budget)
    close(o.listing)
    return nothing
end

"""
    write_heads(o::ModelOutput, m::GwfModel, step, heads)

Append the absolute heads `heads` (Float64, per reduced node) to the head file
(`gwf_ot_dv`, `record_array`): HNOFLO in user nodes outside the reduced grid.
"""
function write_heads(o::ModelOutput, m::GwfModel, step::TimeStep, heads::Vector{Float64})
    isnothing(o.head) && return nothing
    write_head_record(o.head, o.layout, heads; kstp = step.kstp, kper = step.kper, pertim = step.pertim, totim = step.totim, nodeuser = m.conn.nodeuser)
    return nothing
end

# Aux values of the records in force (naux × nbound), for the budget list records.
function _record_aux(pkg::BoundaryPackage)
    naux = length(pkg.spec.auxnames)
    (naux == 0 || pkg.iblock == 0) && return zeros(Float64, naux, pkg.nbound)
    return Matrix{Float64}(_block_data(pkg).auxvar[:, 1:pkg.nbound])  # T6.3: with time-series values
end

"""
    write_budget(o::ModelOutput, m::GwfModel, step)

Append the cell-by-cell flows of the time step to the budget file in MF6's order and
format (`gwf_ot_flow`): STO-SS (and STO-SY) as full arrays over the user nodes when STO
saves flows, FLOW-JA-FACE (`nja` values) when NPF saves flows, then one IMETH 6 list per
stress package that saves flows (`q` = `simvals`, the net rate into the model; `id1` the
user node, `id2` the record number; with the package's auxiliary values; records without
a cell left out). A package saves flows with its own
SAVE_FLOWS or the model's. DATA-SPDIS is not written (specific discharge arrives later).
"""
function write_budget(o::ModelOutput, m::GwfModel, step::TimeStep)
    io = o.budget
    isnothing(io) && return nothing
    spec = m.spec
    kw = (; kstp = step.kstp, kper = step.kper, delt = step.delt, pertim = step.pertim, totim = step.totim)
    nu = nodesuser(o.layout)
    if !isnothing(m.sto) && (spec.save_flows || spec.sto.save_flows)
        v = zeros(nu)
        v[m.conn.nodeuser] .= Float64.(Array(m.sto.strgss))
        write_budget_array(io, o.layout, "STO-SS", v; kw...)
        _uses_sy(m) && write_budget_array(io, o.layout, "STO-SY", zeros(nu); kw...)
    end
    if spec.save_flows || spec.npf.save_flows
        write_budget_array(io, "FLOW-JA-FACE", Float64.(Array(m.flowja)); kw..., dims = (Int(m.conn.nja), 1, 1))
    end
    name = uppercase(spec.name)
    for pkg in m.packages
        (spec.save_flows || pkg.spec.save_flows) || continue
        # Records without a cell (node 0) are left out (save_print_model_flows).
        keep = [i for i in 1:pkg.nbound if pkg.nodelist_host[i] > 0]
        sim = Float64.(Array(pkg.simvals))[keep]
        nodeu = Int32[m.conn.nodeuser[pkg.nodelist_host[i]] for i in keep]
        list = BudgetList(nodeu, Int32.(keep), sim, _record_aux(pkg)[:, keep])
        write_budget_list(
            io, o.layout, budget_text(pkg), list; kw..., modelnam1 = name, paknam1 = name, modelnam2 = name,
            paknam2 = uppercase(pkg.spec.name), auxtxt = pkg.spec.auxnames,
        )
    end
    return nothing
end

"""
    write_head_listing(io, m::GwfModel, step, heads)

PRINT HEAD: the heads by layer in the listing file (10 values per line, `%12.4E`;
OC HEAD PRINT_FORMAT is not applied).
"""
function write_head_listing(io::IO, m::GwfModel, step::TimeStep, heads::Vector{Float64})
    layout = OutputLayout(m.spec.dis)
    v = head_output_values(layout, heads; nodeuser = m.conn.nodeuser)
    nlay = layout.kind === :disu ? 1 : layout.nlay
    per = length(v) ÷ nlay
    for k in 1:nlay
        @printf(io, "\n  HEAD IN LAYER %3d AT END OF TIME STEP %3d IN STRESS PERIOD %4d\n", k, step.kstp, step.kper)
        vals = view(v, ((k - 1) * per + 1):(k * per))
        for (i, x) in enumerate(vals)
            @printf(io, "%12.4E", x)
            (i % 10 == 0 || i == per) && print(io, "\n")
        end
    end
    return nothing
end
