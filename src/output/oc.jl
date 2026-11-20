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
names no file), `listing` (`IO`), the `layout` (`OutputLayout`), the paths and `obs`, the
observation files (`ObservationOutput`, T6.7; `nothing` without OBS6 input).

    ModelOutput(m::GwfModel, simdir, outdir)
    ModelOutput(v::ModelView, simdir, outdir)

T6.6: one per model of a multi-model solution (`ModelView`).
"""
struct ModelOutput
    head::Union{Nothing, IO}
    budget::Union{Nothing, IO}
    listing::IO
    layout::OutputLayout
    head_path::String
    budget_path::String
    listing_path::String
    obs::Union{Nothing, ObservationOutput}
end

function _output_path(simdir, outdir, name)
    rel = isabspath(name) ? relpath(name, simdir) : name
    p = joinpath(outdir, rel)
    mkpath(dirname(p))
    return p
end

function ModelOutput(m::Union{GwfModel, ModelView}, simdir::AbstractString, outdir::AbstractString)
    spec = _vspec(m)
    oc = spec.oc
    hp = (isnothing(oc) || isempty(oc.head_file)) ? "" : _output_path(simdir, outdir, oc.head_file)
    bp = (isnothing(oc) || isempty(oc.budget_file)) ? "" : _output_path(simdir, outdir, oc.budget_file)
    lname = isempty(spec.list_file) ? splitext(basename(spec.file))[1] * ".lst" : spec.list_file
    lp = _output_path(simdir, outdir, lname)
    spec.npf.save_specific_discharge && !isempty(bp) &&
        @info "NPF SAVE_SPECIFIC_DISCHARGE: DATA-SPDIS is not written to the budget file yet" spec.npf.file
    # T6.7: the observations first (input errors before any file is opened).
    obs = ObservationOutput(m, simdir, outdir)
    head = isempty(hp) ? nothing : open(hp, "w")
    budget = isempty(bp) ? nothing : open(bp, "w")
    listing = open(lp, "w")
    return ModelOutput(head, budget, listing, OutputLayout(spec.dis), hp, bp, lp, obs)
end

function Base.close(o::ModelOutput)
    isnothing(o.head) || close(o.head)
    isnothing(o.budget) || close(o.budget)
    close(o.listing)
    isnothing(o.obs) || close(o.obs)
    return nothing
end

"""
    write_heads(o::ModelOutput, m::GwfModel, step, heads)

Append the absolute heads `heads` (Float64, per reduced node) to the head file
(`gwf_ot_dv`, `record_array`): HNOFLO in user nodes outside the reduced grid.
"""
function write_heads(o::ModelOutput, m::Union{GwfModel, ModelView}, step::TimeStep, heads::Vector{Float64})
    isnothing(o.head) && return nothing
    write_head_record(o.head, o.layout, heads; kstp = step.kstp, kper = step.kper, pertim = step.pertim, totim = step.totim, nodeuser = _vconn(m).nodeuser)
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

T6.6: for a model of a multi-model solution (`ModelView`), the model's own values, then the
flows of each exchange with SAVE_FLOWS, from the model's side (`write_exchange_budget`, with
the `ExchangeFlows` `exflows` of the step).
"""
function write_budget(o::ModelOutput, m::Union{GwfModel, ModelView}, step::TimeStep; exflows = nothing)
    io = o.budget
    isnothing(io) && return nothing
    spec = _vspec(m)
    mm = _vmodel(m)
    conn = _vconn(m)
    off = _voffset(m)
    kw = (; kstp = step.kstp, kper = step.kper, delt = step.delt, pertim = step.pertim, totim = step.totim)
    nu = nodesuser(o.layout)
    if !isnothing(spec.sto) && (spec.save_flows || spec.sto.save_flows)
        v = zeros(nu)
        v[conn.nodeuser] .= _model_values(m, mm.sto.strgss)
        write_budget_array(io, o.layout, "STO-SS", v; kw...)
        if _uses_sy(mm)
            v = zeros(nu)
            v[conn.nodeuser] .= _model_values(m, mm.sto.strgsy)
            write_budget_array(io, o.layout, "STO-SY", v; kw...)
        end
    end
    if spec.save_flows || spec.npf.save_flows
        write_budget_array(io, "FLOW-JA-FACE", model_flowja(m); kw..., dims = (Int(conn.nja), 1, 1))
    end
    name = uppercase(spec.name)
    for ip in _vpackages(m)
        pkg = mm.packages[ip]
        (spec.save_flows || pkg.spec.save_flows) || continue
        # Records without a cell (node 0) are left out (save_print_model_flows).
        keep = [i for i in 1:pkg.nbound if pkg.nodelist_host[i] > 0]
        sim = Float64.(Array(pkg.simvals))[keep]
        nodeu = Int32[conn.nodeuser[pkg.nodelist_host[i] - off] for i in keep]
        list = BudgetList(nodeu, Int32.(keep), sim, _record_aux(pkg)[:, keep])
        write_budget_list(
            io, o.layout, budget_text(pkg), list; kw..., modelnam1 = name, paknam1 = name, modelnam2 = name,
            paknam2 = uppercase(pkg.spec.name), auxtxt = pkg.spec.auxnames,
        )
    end
    m isa ModelView && write_exchange_budget(o, m, step, exflows)
    return nothing
end

# The values of a model's nodes of a merged-model array (host, Float64).
_model_values(m::GwfModel, a) = Float64.(Array(a))
_model_values(v::ModelView, a) = Float64.(Array(a)[_vrange(v)])

"""
    model_flowja(m::GwfModel) -> Vector{Float64}
    model_flowja(v::ModelView) -> Vector{Float64}

FLOW-JA-FACE of a model as MF6 writes it (`nja` of the model, host Float64). For a model of
a multi-model solution (T6.6) the model's entries of the merged flows; the diagonal of a
constant-head cell leaves out its exchange flows, which MF6 does not add there
(exg-gwfgwf.f90 `gwf_gwf_add_to_flowja`: active cells only).
"""
model_flowja(m::GwfModel) = Float64.(Array(m.flowja))
function model_flowja(v::ModelView)
    mm = v.model
    f = Float64.(Array(mm.flowja))
    part = v.part
    c = part.conn
    out = f[part.jamap]
    ia, ex = mm.conn.ia, mm.solution.exgentry
    for n in 1:c.nodes
        g = n + part.offset
        mm.ibound_host[g] < 0 || continue
        for ii in (ia[g] + 1):(ia[g + 1] - 1)
            ex[ii] && (out[c.ia[n]] -= f[ii])
        end
    end
    return out
end

"""
    write_conversions(io, m::GwfModel, kstp, kper)

The cell conversions of `m.conversions` (T6.9) in the listing as MF6 writes them
(gwf-npf.f90 `sgwf_npf_wdmsg`): per `wetdry!` call a header `CELL CONVERSIONS FOR
ITER.=<kiter>  STEP=<kstp>  PERIOD=<kper>   (NODE or LRC)`, then five conversions per line
(`(1X,3X,5(A4, A20))`: `WET` or `DRY`, the cell as `(layer,row,col)` on DIS, `(layer,cell2d)`
on DISV, `(node)` on DISU, `nodeu_to_string`). The initial call (`kiter` 0) is written
with step and period 0.
"""
function write_conversions(io::IO, m::GwfModel, kstp::Integer, kper::Integer)
    isempty(m.conversions) && return nothing
    grid = grid_dims(m.spec.dis)
    cell(n) = "(" * join(cellid(grid, m.conn.nodeuser[n]), ",") * ")"
    for (kiter, wet, dry) in m.conversions
        @printf(io, "\n CELL CONVERSIONS FOR ITER.=%d  STEP=%d  PERIOD=%d   (NODE or LRC)\n", kiter, kiter == 0 ? 0 : kstp, kiter == 0 ? 0 : kper)
        items = vcat([("WET", n) for n in wet], [("DRY", n) for n in dry])
        for chunk in Iterators.partition(items, 5)
            print(io, "    ")
            for (a, n) in chunk
                print(io, lpad(a, 4), lpad(cell(n), 20))
            end
            println(io)
        end
    end
    return nothing
end
write_conversions(::IO, ::ModelView, ::Integer, ::Integer) = nothing

"""
    write_head_listing(io, m::GwfModel, step, heads)

PRINT HEAD: the heads by layer in the listing file (10 values per line, `%12.4E`;
OC HEAD PRINT_FORMAT is not applied).
"""
function write_head_listing(io::IO, m::Union{GwfModel, ModelView}, step::TimeStep, heads::Vector{Float64})
    layout = OutputLayout(_vspec(m).dis)
    v = head_output_values(layout, heads; nodeuser = _vconn(m).nodeuser)
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
