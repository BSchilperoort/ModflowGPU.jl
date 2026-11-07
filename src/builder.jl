# In-memory model construction (PLAN.md sections 3.2 and 3.6): a `SimulationSpec` from
# Julia arrays, without files. `load_simulation` (src/input/simulation.jl) and this builder
# produce the same structs; for the same model they are identical (`spec_identical`), down
# to the Float64 bit patterns and the record order (test group builder).
#
# Grid arrays are given in MF6's memory shape, as ModflowCoupling holds them: `(ncol,
# nrow)` for TOP and READASARRAYS arrays, `(ncol, nrow, nlay)` for the cell arrays (`A[j,
# i, k]`, column-major, so `vec(A)` is in user node order). Node arrays also accept a
# scalar (every cell), a vector of `nlay` values (one per layer) or a vector by user node.
# Boundary records are cell lists plus value matrices `(nrecords, ncolumns)` with the
# columns of the DFN PERIOD record (e.g. RIV: stage, cond, rbot).
#
# The builder validates what the loader would reject (shapes, cells outside the grid or
# in inactive cells, non-finite values, names, period numbers) with an `ArgumentError`
# that names the package and the problem; the DIS geometry goes through `check_dis_grid`
# (MF6's messages, `MF6InputError`).
#
# File names: specs built here carry the names `write_simulation` (src/mf6_writer.jl)
# writes them under, `<model>.<ext>` in lower case by default (`<model>.<package name>`
# for stress packages), so that spec → write → load gives the same spec again.

# ---------------------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------------------

_bad(ctx, msg) = throw(ArgumentError("$ctx: $msg"))

function _check_finite(a, ctx, name)
    for (i, x) in enumerate(a)
        isfinite(x) || _bad(ctx, "$name[$i] = $x is not finite")
    end
    return a
end

function _to_int32(x, ctx, name)
    (x isa Integer || (x isa Real && isinteger(x))) || _bad(ctx, "$name: $x is not an integer")
    return Int32(x)
end

# A node array (by user node) from a scalar, a per-layer vector, a vector by user node or
# an array in the grid's shape.
function _node_array(x, dis::AbstractDisSpec, ctx, name; int::Bool = false)
    g = grid_dims(dis)
    nodes = prod(g.mshape)
    nlay = length(g.mshape) == 1 ? 0 : g.mshape[1]
    conv(v) = int ? _to_int32(v, ctx, name) : Float64(v)
    if x isa Real
        out = fill(conv(x), nodes)
    elseif x isa AbstractArray
        shape = reverse(Tuple(g.mshape))  # (ncol, nrow, nlay) / (ncpl, nlay) / (nodes,)
        if size(x) == shape || (x isa AbstractVector && length(x) == nodes)
            out = [conv(v) for v in vec(x)]
        elseif x isa AbstractVector && length(x) == nlay
            ncpl = nodes ÷ nlay
            out = [conv(x[k]) for k in 1:nlay for _ in 1:ncpl]
        else
            per_layer = nlay > 0 ? ", $nlay (one value per layer)" : ""
            _bad(ctx, "$name has size $(size(x)); expected a scalar, $shape, $nodes (by user node)$per_layer")
        end
    else
        _bad(ctx, "$name must be a number or an array, got $(typeof(x))")
    end
    int || _check_finite(out, ctx, name)
    return out
end

_opt_node_array(x, dis, ctx, name) = isnothing(x) ? nothing : _node_array(x, dis, ctx, name)

# A cell-column array (`ncpl` values, READASARRAYS) from a scalar, `(ncol, nrow)` or a
# vector of `ncpl`.
function _cpl_array(x, dis::AbstractDisSpec, ctx, name; int::Bool = false)
    g = grid_dims(dis)
    length(g.mshape) == 1 && _bad(ctx, "READASARRAYS input needs a DIS or DISV grid")
    ncpl = prod(g.mshape[2:end])
    conv(v) = int ? _to_int32(v, ctx, name) : Float64(v)
    if x isa Real
        out = fill(conv(x), ncpl)
    elseif x isa AbstractArray && (length(x) == ncpl && (x isa AbstractVector || size(x) == reverse(Tuple(g.mshape[2:end]))))
        out = [conv(v) for v in vec(x)]
    else
        _bad(ctx, "$name must be a scalar, a $(Tuple(reverse(g.mshape[2:end]))) array or a vector of $ncpl values, got $(x isa AbstractArray ? size(x) : typeof(x))")
    end
    int || _check_finite(out, ctx, name)
    return out
end

_active_nodes(dis::AbstractDisSpec) = [d > 0 for d in idomain_user(dis)]

function _check_positive_active(a, dis, ctx, name)
    isnothing(a) && return nothing
    g = grid_dims(dis)
    for (n, act) in enumerate(_active_nodes(dis))
        (act && !(a[n] > 0)) && _bad(ctx, "$name in cell $(cellid(g, n)) is $(a[n]); it must be greater than zero")
    end
    return nothing
end

function _check_nonnegative_active(a, dis, ctx, name)
    isnothing(a) && return nothing
    g = grid_dims(dis)
    for (n, act) in enumerate(_active_nodes(dis))
        (act && a[n] < 0) && _bad(ctx, "$name in cell $(cellid(g, n)) is $(a[n]); it must not be negative")
    end
    return nothing
end

_check_choice(ctx, name, v, choices) =
    v in choices || _bad(ctx, "$name = $(repr(v)) is not one of $(join(repr.(choices), ", "))")

function _check_name(ctx, what, name, maxlen)
    isempty(name) && _bad(ctx, "empty $what")
    any(isspace, name) && _bad(ctx, "$what \"$name\" contains white space")
    length(name) <= maxlen || _bad(ctx, "$what \"$name\" is longer than $maxlen characters")
    return name
end

# `StressPeriods` from `kper => data` pairs (or a dict), sorted by period.
function _stress_periods(::Type{D}, periods, ctx, f) where {D}
    ps = periods isa AbstractDict ? collect(pairs(periods)) : collect(periods)
    all(p -> p isa Pair, ps) || _bad(ctx, "periods must be `kper => data` pairs or a dict")
    sort!(ps; by = first)
    iper = Int[Int(first(p)) for p in ps]
    for (b, k) in enumerate(iper)
        k >= 1 || _bad(ctx, "stress period $k: periods are numbered from 1")
        (b > 1 && iper[b - 1] == k) && _bad(ctx, "stress period $k is given twice")
    end
    return StressPeriods(iper, D[f(k, last(p)) for (k, p) in zip(iper, ps)])
end

# Rebuild a keyword-constructed struct with some fields replaced.
_with(x::T; kw...) where {T} = T(; (f => getfield(x, f) for f in fieldnames(T))..., kw...)

# ---------------------------------------------------------------------------------------
# Discretization, NPF, IC, STO, OC
# ---------------------------------------------------------------------------------------

"""
    build_dis(; delr, delc, top, botm, idomain = nothing, xorigin = 0.0, yorigin = 0.0,
              angrot = 0.0, length_units = :undefined, nogrb = false, grb_file = "",
              file = "") -> DisSpec

A structured grid. `botm` is `(ncol, nrow, nlay)` (a matrix for one layer); `delr` is a
scalar or a vector of `ncol`, `delc` a scalar or a vector of `nrow`, `top` a scalar or
`(ncol, nrow)`; `idomain` is `nothing` (all active), `(ncol, nrow)` (the same in every
layer, as ModflowCoupling's land mask) or `(ncol, nrow, nlay)`. The geometry is checked
as MF6 does (`check_dis_grid`: positive thickness of active cells, at least one active
cell).
"""
function build_dis(;
        delr, delc, top, botm, idomain = nothing, xorigin::Real = 0.0, yorigin::Real = 0.0, angrot::Real = 0.0,
        length_units::Symbol = :undefined, nogrb::Bool = false, grb_file::AbstractString = "", file::AbstractString = "",
    )
    ctx = "build_dis"
    botm isa AbstractArray && ndims(botm) in (2, 3) || _bad(ctx, "botm must be a (ncol, nrow, nlay) array")
    ncol, nrow, nlay = size(botm, 1), size(botm, 2), size(botm, 3)
    vec_of(x, n, name) = x isa Real ? fill(Float64(x), n) :
        (x isa AbstractVector && length(x) == n ? Float64.(x) : _bad(ctx, "$name must be a scalar or a vector of $n values"))
    delr_ = _check_finite(vec_of(delr, ncol, "delr"), ctx, "delr")
    delc_ = _check_finite(vec_of(delc, nrow, "delc"), ctx, "delc")
    for (name, v) in (("delr", delr_), ("delc", delc_)), (i, x) in enumerate(v)
        x > 0 || _bad(ctx, "$name[$i] = $x; cell widths must be greater than zero")
    end
    top_ = top isa Real ? fill(Float64(top), ncol, nrow) :
        (top isa AbstractMatrix && size(top) == (ncol, nrow) ? Matrix{Float64}(top) : _bad(ctx, "top must be a scalar or a ($ncol, $nrow) matrix"))
    botm_ = Array{Float64, 3}(reshape(botm, ncol, nrow, nlay))
    idom = if isnothing(idomain)
        ones(Int32, ncol, nrow, nlay)
    elseif idomain isa AbstractArray && size(idomain) == (ncol, nrow)
        Int32[_to_int32(idomain[j, i], ctx, "idomain") for j in 1:ncol, i in 1:nrow, _ in 1:nlay]
    elseif idomain isa AbstractArray && size(idomain) == (ncol, nrow, nlay)
        Int32[_to_int32(x, ctx, "idomain") for x in idomain]
    else
        _bad(ctx, "idomain must be nothing, a ($ncol, $nrow) or a ($ncol, $nrow, $nlay) array")
    end
    _check_finite(top_, ctx, "top")
    _check_finite(botm_, ctx, "botm")
    _check_choice(ctx, "length_units", length_units, (:undefined, :feet, :meters, :centimeters))
    dis = DisSpec(;
        nlay = Int32(nlay), nrow = Int32(nrow), ncol = Int32(ncol), delr = delr_, delc = delc_, top = top_,
        botm = botm_, idomain = idom, xorigin = Float64(xorigin), yorigin = Float64(yorigin), angrot = Float64(angrot),
        length_units, nogrb, grb_file = String(grb_file), file = String(file),
    )
    check_dis_grid(dis)
    return dis
end

"""
    build_npf(dis; k, icelltype = 0, k22 = nothing, k33 = nothing, angle1 = nothing,
              angle2 = nothing, angle3 = nothing, wetdry = nothing, options...) -> NpfSpec

Node property flow on grid `dis`. Each array is a scalar, one value per layer, a vector by
user node or a grid-shaped array (`(ncol, nrow, nlay)`); `nothing` means not given (MF6
then uses K22 = K33 = K and zero angles). K, K22 and K33 must be positive in active
cells. `options` are the other `NpfSpec` fields (`cell_averaging`, `save_flows`, ...).
"""
function build_npf(
        dis::AbstractDisSpec; k, icelltype = 0, k22 = nothing, k33 = nothing, angle1 = nothing, angle2 = nothing,
        angle3 = nothing, wetdry = nothing, options...,
    )
    ctx = "build_npf"
    for o in keys(options)
        o in fieldnames(NpfSpec) || _bad(ctx, "unknown option $o")
    end
    haskey(options, :cell_averaging) && _check_choice(ctx, "cell_averaging", options[:cell_averaging], (:harmonic, :logarithmic, :amt_lmk, :amt_hmk))
    npf = NpfSpec(;
        icelltype = _node_array(icelltype, dis, ctx, "icelltype"; int = true), k = _node_array(k, dis, ctx, "k"),
        k22 = _opt_node_array(k22, dis, ctx, "k22"), k33 = _opt_node_array(k33, dis, ctx, "k33"),
        angle1 = _opt_node_array(angle1, dis, ctx, "angle1"), angle2 = _opt_node_array(angle2, dis, ctx, "angle2"),
        angle3 = _opt_node_array(angle3, dis, ctx, "angle3"), wetdry = _opt_node_array(wetdry, dis, ctx, "wetdry"),
        options...,
    )
    (npf.k22overk && isnothing(npf.k22)) && _bad(ctx, "K22OVERK needs k22")
    (npf.k33overk && isnothing(npf.k33)) && _bad(ctx, "K33OVERK needs k33")
    for (name, a) in (("k", npf.k), ("k22", npf.k22), ("k33", npf.k33))
        _check_positive_active(a, dis, ctx, name)
    end
    return npf
end

"""
    build_ic(dis; strt, file = "") -> IcSpec

Initial heads on grid `dis` (a scalar, one value per layer, by user node or grid-shaped).
"""
build_ic(dis::AbstractDisSpec; strt, file::AbstractString = "") =
    IcSpec(; strt = _node_array(strt, dis, "build_ic", "strt"), file = String(file))

"""
    build_sto(dis; ss, iconvert = 0, sy = nothing, periods = (), options...) -> StoSpec

Storage on grid `dis`. `periods` are `kper => :steady_state` / `kper => :transient`
pairs (MF6's PERIOD blocks; before the first one a model with STO is transient).
SS and SY must not be negative in active cells; convertible cells need SY. `options`
are the other `StoSpec` fields (`storagecoefficient`, `ss_confined_only`, `save_flows`,
`file`).
"""
function build_sto(dis::AbstractDisSpec; ss, iconvert = 0, sy = nothing, periods = (), options...)
    ctx = "build_sto"
    for o in keys(options)
        o in fieldnames(StoSpec) || _bad(ctx, "unknown option $o")
    end
    sp = _stress_periods(Symbol, periods, ctx, (k, s) -> (_check_choice(ctx, "period $k", s, (:steady_state, :transient)); s))
    sto = StoSpec(;
        iconvert = _node_array(iconvert, dis, ctx, "iconvert"; int = true), ss = _node_array(ss, dis, ctx, "ss"),
        sy = _opt_node_array(sy, dis, ctx, "sy"), periods = sp, options...,
    )
    act = _active_nodes(dis)
    (isnothing(sto.sy) && any(n -> act[n] && sto.iconvert[n] != 0, eachindex(act))) && _bad(ctx, "convertible cells (iconvert /= 0) need sy")
    _check_nonnegative_active(sto.ss, dis, ctx, "ss")
    _check_nonnegative_active(sto.sy, dis, ctx, "sy")
    return sto
end

"""
    oc_record(action, rtype, setting = :all, steps = Int[]) -> OcRecord
    oc_record(text) -> OcRecord

One OC PERIOD record: `oc_record(:save, "HEAD")`, `oc_record(:print, "BUDGET", :steps,
[1, 5])` or `oc_record("SAVE HEAD FREQUENCY 2")`. `action` is `:save` or `:print`,
`rtype` `"HEAD"` or `"BUDGET"`, `setting` `:all`, `:first`, `:last`, `:frequency` (one
step number) or `:steps`.
"""
function oc_record(action::Symbol, rtype::AbstractString, setting::Symbol = :all, steps = Int[])
    ctx = "oc_record"
    _check_choice(ctx, "action", action, (:save, :print))
    _check_choice(ctx, "rtype", uppercase(rtype), ("HEAD", "BUDGET"))
    _check_choice(ctx, "setting", setting, (:all, :first, :last, :frequency, :steps))
    st = Int[Int(s) for s in steps]
    setting === :frequency && length(st) != 1 && _bad(ctx, "FREQUENCY takes one step interval")
    setting === :steps && (isempty(st) || any(<=(0), st)) && _bad(ctx, "STEPS takes positive step numbers")
    setting in (:all, :first, :last) && !isempty(st) && _bad(ctx, "$(uppercase(String(setting))) takes no step numbers")
    return OcRecord(action, uppercase(rtype), setting, st)
end

function oc_record(text::AbstractString)
    r = _oc_record(InputLine(text, "oc_record", 0))
    return oc_record(r.action, r.rtype, r.setting, r.steps)
end

_oc_records(recs) = OcRecord[r isa OcRecord ? oc_record(r.action, r.rtype, r.setting, r.steps) : oc_record(r) for r in recs]

"""
    build_oc(; head_file = "", budget_file = "", budgetcsv_file = "",
             head_print_format = nothing, periods = (), file = "") -> OcSpec

Output control. `periods` are `kper => records` pairs, the records `OcRecord`s or their
text (`"SAVE HEAD ALL"`, see `oc_record`); an empty list turns output off from that
period on.
"""
function build_oc(;
        head_file::AbstractString = "", budget_file::AbstractString = "", budgetcsv_file::AbstractString = "",
        head_print_format::Union{Nothing, PrintFormat} = nothing, periods = (), file::AbstractString = "",
    )
    sp = _stress_periods(Vector{OcRecord}, periods, "build_oc", (_, r) -> _oc_records(r))
    return OcSpec(; head_file = String(head_file), budget_file = String(budget_file), budgetcsv_file = String(budgetcsv_file), head_print_format, periods = sp, file = String(file))
end

# ---------------------------------------------------------------------------------------
# Stress packages
# ---------------------------------------------------------------------------------------

_dfn_list_name(kind::Symbol) = "gwf-" * lowercase(String(kind))

# User nodes of a cell list: MF6 cellids (tuples `(k, i, j)`), array indices into the grid
# shape (`CartesianIndex(j, i, k)`, or `CartesianIndex(j, i)` with `layer`) or user nodes.
function _cell_nodes(cells, layer, g::GridDims, ctx)
    shape = reverse(Tuple(g.mshape))
    n = length(cells)
    layers = layer isa Integer ? fill(Int(layer), n) :
        (length(layer) == n ? Int.(layer) : _bad(ctx, "layer must be one integer or one per cell ($n)"))
    nodes = Vector{Int32}(undef, n)
    for (r, c) in enumerate(cells)
        if c isa Integer
            1 <= c <= prod(shape) || _bad(ctx, "record $r: user node $c is outside the grid (1..$(prod(shape)))")
            nodes[r] = c
        elseif c isa Tuple
            (length(c) == length(shape) && valid_cellid(g, c)) || _bad(ctx, "record $r: cellid $c is outside the grid $(Tuple(g.mshape))")
            nodes[r] = user_node(g, c)
        elseif c isa CartesianIndex
            ci = length(c) == length(shape) - 1 ? CartesianIndex(Tuple(c)..., layers[r]) : c
            (length(ci) == length(shape) && checkbounds(Bool, CartesianIndices(shape), ci)) ||
                _bad(ctx, "record $r: cell $ci is outside the grid $shape")
            nodes[r] = LinearIndices(shape)[ci]
        else
            _bad(ctx, "record $r: a cell must be a cellid tuple, a CartesianIndex or a user node, got $(typeof(c))")
        end
    end
    return nodes
end

function _record_matrix(x, n::Int, ncol::Int, ctx, name)
    ncol == 0 && (isnothing(x) || isempty(x)) && return zeros(0, n)
    isnothing(x) && _bad(ctx, "$name not given ($ncol column(s) expected)")
    m = x isa AbstractVector && ncol == 1 ? reshape(x, :, 1) : x
    (m isa AbstractMatrix && size(m) == (n, ncol)) ||
        _bad(ctx, "$name has size $(size(x)); expected ($n, $ncol) (one row per record)")
    out = Matrix{Float64}(permutedims(m))
    return _check_finite(out, ctx, name)
end

# One period's list from `(; cells, values, aux, boundnames, layer)`.
function _period_list(rec, s::ListSchema, idom, ctx)
    rec isa PeriodList && return rec
    rec isa NamedTuple || _bad(ctx, "period data must be a NamedTuple (cells, values[, aux, boundnames, layer]) or a PeriodList")
    for k in keys(rec)
        k in (:cells, :values, :aux, :boundnames, :layer) || _bad(ctx, "unknown period field $k")
    end
    cells = rec.cells
    n = length(cells)
    n <= s.maxbound || _bad(ctx, "$n records exceed MAXBOUND = $(s.maxbound)")
    nodes = _cell_nodes(cells, get(rec, :layer, 1), s.grid, ctx)
    for (r, nd) in enumerate(nodes)
        idom[nd] > 0 || _bad(ctx, "record $r: cell $(cellid(s.grid, nd)) is outside the active grid domain (IDOMAIN <= 0)")
    end
    bound = _record_matrix(get(rec, :values, nothing), n, s.nbound_rows, ctx, "values")
    auxvar = _record_matrix(get(rec, :aux, nothing), n, length(s.auxnames), ctx, "aux")
    bn = get(rec, :boundnames, nothing)
    names = String[]
    if s.boundnames
        names = isnothing(bn) ? fill("", n) :
            (length(bn) == n ? [uppercase(String(b)) for b in bn] : _bad(ctx, "boundnames: $(length(bn)) names for $n records"))
        for b in names
            isempty(b) || _check_name(ctx, "boundname", b, LENBOUNDNAME)
        end
    else
        isnothing(bn) || _bad(ctx, "boundnames given without boundnames = true")
    end
    return PeriodList(nodes, bound, auxvar, names, TsRef[])
end

function _boundary_options(ctx, kind, auxnames, auxmultname)
    kind in BOUNDARY_KINDS || _bad(ctx, "unknown stress package kind $kind (expected one of $(join(BOUNDARY_KINDS, ", ")))")
    aux = [uppercase(String(a)) for a in auxnames]
    for a in aux
        _check_name(ctx, "auxiliary name", a, LENAUXNAME)
    end
    allunique(aux) || _bad(ctx, "auxiliary names are not unique: $(join(aux, ", "))")
    mult = uppercase(String(auxmultname))
    col = isempty(mult) ? 0 : something(findfirst(==(mult), aux), 0)
    (!isempty(mult) && col == 0) && _bad(ctx, "auxmultname $mult is not an auxiliary variable ($(join(aux, ", ")))")
    return aux, mult, Int32(col)
end

"""
    build_boundary(kind, dis; periods, name = "", auxnames = String[], auxmultname = "",
                   auxdepthname = "", boundnames = false, maxbound = 0, nseg = 1,
                   surf_rate_specified = false, fixed_cell = false, print_input = false,
                   print_flows = false, save_flows = false, file = "") -> BoundarySpec{kind}

A list stress package (`kind` one of `BOUNDARY_KINDS`) on grid `dis`. `periods` are
`kper => (; cells, values, aux, boundnames, layer)` pairs, the records of each PERIOD
block (they apply until the next block; an empty `cells` list removes all stresses):

- `cells`: MF6 cellids `(k, i, j)`, array indices `CartesianIndex(j, i, k)` (the
  `(ncol, nrow, nlay)` shape), `CartesianIndex(j, i)` with `layer` (one integer or one per
  cell; default 1), or user node numbers. Cells must be active.
- `values`: `(nrecords, ncolumns)`, the double columns of the DFN record in order (CHD
  head; WEL q; RCH recharge; GHB bhead cond; RIV stage cond rbot; DRN elev cond; EVT
  surface rate depth, then `pxdp`, `petm` (`nseg - 1` each) and `petm0` with
  `surf_rate_specified`); a vector when there is one column.
- `aux`: `(nrecords, length(auxnames))`; `boundnames`: one name per record (with
  `boundnames = true`).

`maxbound = 0` uses the largest number of records of a period. An empty `name` gets
MF6's default (`"WEL-1"`, ...) in `build_model`.
"""
function build_boundary(
        kind::Symbol, dis::AbstractDisSpec; periods, name::AbstractString = "", auxnames = String[],
        auxmultname::AbstractString = "", auxdepthname::AbstractString = "", boundnames::Bool = false,
        maxbound::Integer = 0, nseg::Integer = 1, surf_rate_specified::Bool = false, fixed_cell::Bool = false,
        print_input::Bool = false, print_flows::Bool = false, save_flows::Bool = false, file::AbstractString = "",
    )
    ctx = "build_boundary($(repr(kind))$(isempty(name) ? "" : ", \"$name\""))"
    aux, mult, col = _boundary_options(ctx, kind, auxnames, auxmultname)
    depth = uppercase(String(auxdepthname))
    (!isempty(depth) && !(depth in aux)) && _bad(ctx, "auxdepthname $depth is not an auxiliary variable")
    (!isempty(depth) && kind !== :DRN) && _bad(ctx, "auxdepthname is a DRN option")
    nseg >= 1 || _bad(ctx, "nseg must be at least 1")
    (kind !== :EVT && (nseg != 1 || surf_rate_specified)) && _bad(ctx, "nseg and surf_rate_specified are EVT options")
    (fixed_cell && !(kind in (:RCH, :EVT))) && _bad(ctx, "fixed_cell is an RCH/EVT option")
    ps = periods isa AbstractDict ? collect(pairs(periods)) : collect(periods)
    nmax = maximum((length(last(p) isa PeriodList ? last(p).nodeu : last(p).cells) for p in ps); init = 0)
    mb = maxbound == 0 ? max(nmax, 1) : Int(maxbound)
    o = PackageOptions(; auxnames = aux, boundnames, surf_rate_specified, maxbound = mb, nseg = Int(nseg))
    g = grid_dims(dis)
    schema = list_schema(dfn_definition(_dfn_list_name(kind)), g, o)
    idom = idomain_user(dis)
    sp = _stress_periods(PeriodList, ps, ctx, (k, r) -> _period_list(r, schema, idom, "$ctx, period $k"))
    return BoundarySpec{kind}(;
        name = uppercase(String(name)), auxnames = aux, auxmultname = mult, auxmultcol = col, auxdepthname = depth,
        boundnames, fixed_cell, nseg = Int32(nseg), surf_rate_specified, maxbound = Int32(mb), print_input, print_flows,
        save_flows, schema, periods = sp, file = String(file),
    )
end

"""
    build_array_boundary(kind, dis; periods, name = "", auxnames = String[],
                         auxmultname = "", fixed_cell = false, print_input = false,
                         print_flows = false, save_flows = false, file = "")
        -> BoundarySpec{kind}

A READASARRAYS stress package (RCHA: `kind = :RCH`; EVTA: `kind = :EVT`) on a DIS or
DISV grid. `periods` are `kper => (; name = array, ...)` pairs: the arrays given in
that PERIOD block (`recharge`, `irch`; `surface`, `rate`, `depth`, `ievt`; auxiliary
variables by name), each a scalar, `(ncol, nrow)` or a vector of `ncpl`. Arrays not
given keep their previous values (zero, layer 1 at first), as in MF6.
"""
function build_array_boundary(
        kind::Symbol, dis::AbstractDisSpec; periods, name::AbstractString = "", auxnames = String[],
        auxmultname::AbstractString = "", fixed_cell::Bool = false, print_input::Bool = false,
        print_flows::Bool = false, save_flows::Bool = false, file::AbstractString = "",
    )
    ctx = "build_array_boundary($(repr(kind))$(isempty(name) ? "" : ", \"$name\""))"
    kind in (:RCH, :EVT) || _bad(ctx, "READASARRAYS input exists for RCH and EVT only")
    aux, mult, col = _boundary_options(ctx, kind, auxnames, auxmultname)
    g = grid_dims(dis)
    length(g.mshape) == 1 && _bad(ctx, "READASARRAYS input needs a DIS or DISV grid")
    nlay = g.mshape[1]
    ncpl = prod(g.mshape[2:end])
    pb = dfn_block(dfn_definition(_dfn_list_name(kind) * "a"), "PERIOD")
    layername = kind === :RCH ? "irch" : "ievt"
    # read_array_periods: the state carried from block to block.
    layer = fill(Int32(1), ncpl)
    values = Dict{String, Vector{Float64}}(v.name => zeros(ncpl) for v in pb.variables if v.type === :double && v.name != "aux")
    auxvar = zeros(length(aux), ncpl)
    function period(k, arrays)
        pctx = "$ctx, period $k"
        arrays isa Union{NamedTuple, AbstractDict} || _bad(pctx, "period data must be a NamedTuple or a dict of arrays")
        values = copy(values)
        auxvar = copy(auxvar)
        given = String[]
        for (key, x) in pairs(arrays)
            kname = String(key)
            iaux = findfirst(==(uppercase(kname)), aux)
            if !isnothing(iaux)
                auxvar[iaux, :] .= _cpl_array(x, dis, pctx, kname)
                push!(given, aux[iaux])
            elseif lowercase(kname) == layername
                layer = _cpl_array(x, dis, pctx, kname; int = true)
                for (j, l) in enumerate(layer)
                    1 <= l <= nlay || _bad(pctx, "$layername[$j] = $l is not a layer (1..$nlay)")
                end
                push!(given, layername)
            elseif haskey(values, lowercase(kname))
                values[lowercase(kname)] = _cpl_array(x, dis, pctx, kname)
                push!(given, lowercase(kname))
            else
                _bad(pctx, "\"$kname\" is not a PERIOD array ($(join([layername; sort!(collect(keys(values)))], ", "))) nor an auxiliary variable")
            end
        end
        allunique(given) || _bad(pctx, "an array is given twice")
        return ArrayPeriod(layer, values, auxvar, Dict{String, Symbol}(), given)
    end
    sp = _stress_periods(ArrayPeriod, periods, ctx, period)
    return BoundarySpec{kind}(;
        name = uppercase(String(name)), readasarrays = true, auxnames = aux, auxmultname = mult, auxmultcol = col,
        fixed_cell, print_input, print_flows, save_flows, schema = nothing, periods = sp, file = String(file),
    )
end

# ---------------------------------------------------------------------------------------
# Model and simulation
# ---------------------------------------------------------------------------------------

_ftype(::DisSpec) = "DIS6"
_ftype(::DisvSpec) = "DISV6"
_ftype(::DisuSpec) = "DISU6"

_default_file(file, base, ext) = isempty(file) ? "$base.$ext" : file

function _check_model_arrays(ctx, dis, npf, ic, sto)
    n = nodes_user(dis)
    for (name, a) in (
            ("npf.icelltype", npf.icelltype), ("npf.k", npf.k), ("npf.k22", npf.k22), ("npf.k33", npf.k33),
            ("npf.angle1", npf.angle1), ("npf.angle2", npf.angle2), ("npf.angle3", npf.angle3), ("npf.wetdry", npf.wetdry),
            ("ic.strt", ic.strt), ("sto.iconvert", isnothing(sto) ? nothing : sto.iconvert),
            ("sto.ss", isnothing(sto) ? nothing : sto.ss), ("sto.sy", isnothing(sto) ? nothing : sto.sy),
        )
        isnothing(a) || length(a) == n || _bad(ctx, "$name has $(length(a)) values; the grid has $n cells")
    end
    return nothing
end

function _check_boundary_grid(ctx, b::BoundarySpec, dis)
    g = grid_dims(dis)
    idom = idomain_user(dis)
    if b.readasarrays
        ncpl = prod(g.mshape[2:end])
        for p in b.periods.data
            length(p.layer) == ncpl || _bad(ctx, "package $(b.name): $(length(p.layer)) cell columns; the grid has $ncpl")
        end
    else
        b.schema.grid.mshape == g.mshape || _bad(ctx, "package $(b.name) was built for grid $(b.schema.grid.mshape), not $(g.mshape)")
        for p in b.periods.data, nd in p.nodeu
            idom[nd] > 0 || _bad(ctx, "package $(b.name): cell $(cellid(g, nd)) is outside the active grid domain")
        end
    end
    return nothing
end

"""
    build_model(name; dis, npf, ic, sto = nothing, oc = nothing, boundaries = [],
                list_file = "", print_input = false, print_flows = false,
                save_flows = false, newton = false, newton_under_relaxation = false,
                file = "") -> GwfSpec

A GWF model from package specs (`build_dis`, `build_npf`, `build_ic`, `build_sto`,
`build_oc`, `build_boundary`, `build_array_boundary`). `boundaries` are given in name-file
order; the spec holds them in MF6's order (WEL, DRN, RIV, GHB, RCH, EVT, CHD, then this
order). Unnamed stress packages get MF6's default names (`"CHD-1"`, ...). Files without a
name get `<name>.<ext>` (lower case; `<name>.<package name>` for stress packages), the
names `write_simulation` uses.
"""
function build_model(
        name::AbstractString; dis::AbstractDisSpec, npf::NpfSpec, ic::IcSpec, sto::Union{Nothing, StoSpec} = nothing,
        oc::Union{Nothing, OcSpec} = nothing, boundaries = BoundarySpec[], list_file::AbstractString = "",
        print_input::Bool = false, print_flows::Bool = false, save_flows::Bool = false, newton::Bool = false,
        newton_under_relaxation::Bool = false, file::AbstractString = "",
    )
    mname = uppercase(String(name))
    ctx = "build_model(\"$mname\")"
    _check_name(ctx, "model name", mname, LENMODELNAME)
    _check_model_arrays(ctx, dis, npf, ic, sto)
    base = lowercase(mname)
    dis = _with(dis; file = _default_file(dis.file, base, "dis"))
    npf = _with(npf; file = _default_file(npf.file, base, "npf"))
    ic = _with(ic; file = _default_file(ic.file, base, "ic"))
    isnothing(sto) || (sto = _with(sto; file = _default_file(sto.file, base, "sto")))
    isnothing(oc) || (oc = _with(oc; file = _default_file(oc.file, base, "oc")))
    entries = PackageEntry[PackageEntry(_ftype(dis), dis.file, "DIS"), PackageEntry("NPF6", npf.file, "NPF"), PackageEntry("IC6", ic.file, "IC")]
    isnothing(sto) || push!(entries, PackageEntry("STO6", sto.file, "STO"))
    counts = Dict{Symbol, Int}()
    named = BoundarySpec[]
    for b in boundaries
        b isa BoundarySpec || _bad(ctx, "boundaries must be BoundarySpecs (build_boundary), got $(typeof(b))")
        kind = boundary_kind(b)
        counts[kind] = get(counts, kind, 0) + 1
        bname = isempty(b.name) ? "$kind-$(counts[kind])" : uppercase(b.name)
        _check_name(ctx, "package name", bname, LENPACKAGENAME)
        any(c -> c.name == bname, named) && _bad(ctx, "package name $bname is used twice")
        _check_boundary_grid(ctx, b, dis)
        b = _with(b; name = bname, file = _default_file(b.file, base, lowercase(bname)))
        push!(named, b)
        push!(entries, PackageEntry("$(kind)6", b.file, bname))
    end
    isnothing(oc) || push!(entries, PackageEntry("OC6", oc.file, "OC"))
    allunique(e.file for e in entries) || _bad(ctx, "two packages share a file name: $(join((e.file for e in entries), ", "))")
    order = Symbol.(rstrip.(_GWF_MULTI_FTYPES, '6'))
    sorted = BoundarySpec[b for k in order for b in named if boundary_kind(b) === k]
    return GwfSpec(;
        name = mname, list_file = String(list_file), print_input, print_flows, save_flows, newton, newton_under_relaxation,
        packages = entries, dis, npf, ic, sto, oc, boundaries = sorted, file = _default_file(String(file), base, "nam"),
    )
end

"""
    build_tdis(; perlen, nstp = 1, tsmult = 1.0, time_units = :undefined,
               start_date_time = "", file = "") -> TdisSpec

Time discretization: `perlen` per stress period; `nstp` and `tsmult` a scalar (every
period) or one value per period. Checked as MF6's `check_tdis_timing`.
"""
function build_tdis(; perlen, nstp = 1, tsmult = 1.0, time_units::Symbol = :undefined, start_date_time::AbstractString = "", file::AbstractString = "")
    ctx = "build_tdis"
    p = perlen isa Real ? [Float64(perlen)] : Float64.(collect(perlen))
    isempty(p) && _bad(ctx, "no stress periods")
    n = length(p)
    per(x, name) = x isa Real ? fill(x, n) : (length(x) == n ? collect(x) : _bad(ctx, "$name must be a scalar or $n values"))
    ns = Int32[_to_int32(x, ctx, "nstp") for x in per(nstp, "nstp")]
    tm = Float64.(per(tsmult, "tsmult"))
    _check_finite(p, ctx, "perlen")
    _check_finite(tm, ctx, "tsmult")
    _check_choice(ctx, "time_units", time_units, _TIME_UNIT_SYMBOLS)
    _check_tdis_timing(ctx, p, ns, tm)
    return TdisSpec(; perlen = p, nstp = ns, tsmult = tm, time_units, start_date_time = String(start_date_time), file = String(file))
end

# ImsSpec fields that are not IMS keywords, and the keyword of the others.
const _IMS_NOT_KEYWORDS = (:models, :given, :file)
_ims_keyword(f::Symbol) = f === :rclose_option ? "INNER_RCLOSE" : uppercase(String(f))

const _IMS_CHOICES = Dict(
    :print_option => (:none, :summary, :all), :complexity => (:simple, :moderate, :complex),
    :no_ptc => (:none, :all, :first), :under_relaxation => (:none, :simple, :cooley, :dbd),
    :rclose_option => (:infinity_norm, :strict, :l2norm, :relative, :l2norm_relative),
    :linear_acceleration => (:cg, :bicgstab), :scaling_method => (:none, :diagonal, :l2norm),
    :reordering_method => (:none, :rcm, :md),
)

"""
    build_ims(; complexity = nothing, models = String[], file = "", settings...) -> ImsSpec

An IMS solution: MF6's defaults of `complexity` (`:simple` when not given) with the
`ImsSpec` fields in `settings` (`outer_dvclose = 1e-6`, `linear_acceleration =
:bicgstab`, ...). `given` lists the keywords passed (COMPLEXITY included when given), in
the order `write_simulation` writes them. `models` is filled in by `build_simulation`
when empty.
"""
function build_ims(; complexity::Union{Nothing, Symbol} = nothing, models = String[], file::AbstractString = "", settings...)
    ctx = "build_ims"
    passed = Set{Symbol}(keys(settings))
    isnothing(complexity) || push!(passed, :complexity)
    for k in keys(settings)
        (k in fieldnames(ImsSpec) && !(k in _IMS_NOT_KEYWORDS)) || _bad(ctx, "unknown IMS setting $k")
    end
    cx = something(complexity, :simple)
    _check_choice(ctx, "complexity", cx, _IMS_CHOICES[:complexity])
    d = ims_defaults(cx)
    vals = Dict{Symbol, Any}(f => getfield(d, f) for f in fieldnames(ImsSpec))
    merge!(vals, Dict(settings))
    for (k, choices) in _IMS_CHOICES
        _check_choice(ctx, String(k), vals[k], choices)
    end
    vals[:under_relaxation_theta] = max(vals[:under_relaxation_theta], 1.0e-3)  # sln_read_options
    vals[:outer_maximum] > 0 || _bad(ctx, "outer_maximum must be > 0")
    (vals[:under_relaxation] === :simple && vals[:under_relaxation_gamma] == 0) && _bad(ctx, "under_relaxation_gamma must be > 0 with SIMPLE under-relaxation")
    given = String[]
    for f in fieldnames(ImsSpec)
        f in passed || continue
        kw = _ims_keyword(f)
        kw in given || push!(given, kw)
    end
    vals[:given] = given
    vals[:models] = [uppercase(String(m)) for m in models]
    vals[:file] = String(file)
    return ImsSpec(; vals...)
end

"""
    build_simulation(models; tdis, ims = build_ims(), mxiter = 1, simdir = "",
                     continue_on_nonconvergence = false, nocheck = false,
                     memory_print_option = "", profile_option = "", maxerrors = 0,
                     print_input = false) -> SimulationSpec

A simulation of one or more GWF models (`build_model`) solved by one IMS solution in one
solution group. The IMS solution gets the model names when its `models` is empty; TDIS
and IMS files without a name get `<first model>.tdis` / `.ims`. Period data must not
refer to stress periods after the last one.
"""
function build_simulation(
        models; tdis::TdisSpec, ims::ImsSpec = build_ims(), mxiter::Integer = 1, simdir::AbstractString = "",
        continue_on_nonconvergence::Bool = false, nocheck::Bool = false, memory_print_option::AbstractString = "",
        profile_option::AbstractString = "", maxerrors::Integer = 0, print_input::Bool = false,
    )
    ctx = "build_simulation"
    ms = models isa GwfSpec ? GwfSpec[models] : GwfSpec[m for m in models]
    isempty(ms) && _bad(ctx, "no models")
    names = [m.name for m in ms]
    allunique(names) || _bad(ctx, "model names are not unique: $(join(names, ", "))")
    files = [m.file for m in ms]
    allunique(files) || _bad(ctx, "model name files are not unique: $(join(files, ", "))")
    np = nper(tdis)
    for m in ms
        sps = Any[("STO", m.sto), ("OC", m.oc)]
        append!(sps, ((b.name, b) for b in m.boundaries))
        for (pname, p) in sps
            isnothing(p) && continue
            isempty(p.periods.iper) || last(p.periods.iper) <= np ||
                _bad(ctx, "model $(m.name), package $pname: PERIOD $(last(p.periods.iper)) after the last stress period ($np)")
        end
    end
    base = lowercase(ms[1].name)
    ims_models = isempty(ims.models) ? names : ims.models
    sort(ims_models) == sort(names) || _bad(ctx, "the IMS solution's models ($(join(ims_models, ", "))) are not the simulation's ($(join(names, ", ")))")
    ims = _with(ims; models = ims_models, file = _default_file(ims.file, base, "ims"))
    tdis = _with(tdis; file = _default_file(tdis.file, base, "tdis"))
    mxiter >= 1 || _bad(ctx, "mxiter must be at least 1")
    return SimulationSpec(;
        simdir = String(simdir), continue_on_nonconvergence, nocheck, memory_print_option = String(memory_print_option),
        profile_option = String(profile_option), maxerrors = Int32(maxerrors), print_input, tdis, models = ms,
        solution_groups = [SolutionGroupSpec(; mxiter = Int32(mxiter), solutions = [ims])],
    )
end

# ---------------------------------------------------------------------------------------
# Structural equality
# ---------------------------------------------------------------------------------------

"""
    spec_difference(a, b; ignore = ()) -> Union{Nothing, String}

The first difference between two specs (or any values made of structs, arrays, dicts,
tuples, strings and numbers), as a field path and the two values, or `nothing` when they
are identical: the same types, the same array shapes and order, and floating-point values
with the same bit pattern (`-0.0` differs from `0.0`; NaN equals NaN only bitwise).
Fields named in `ignore` (e.g. `(:simdir,)`) are skipped at any depth.
"""
spec_difference(a, b; ignore = ()) = _spec_diff(a, b, "spec", Tuple(ignore))

"""
    spec_identical(a, b; ignore = ()) -> Bool

Whether `spec_difference(a, b; ignore)` finds no difference.
"""
spec_identical(a, b; ignore = ()) = isnothing(spec_difference(a, b; ignore))

_short(x) = (s = repr(x); length(s) > 80 ? s[1:77] * "..." : s)

function _spec_diff(a, b, path, ignore)
    typeof(a) === typeof(b) || return "$path: type $(typeof(a)) vs $(typeof(b))"
    if a isa AbstractFloat
        return reinterpret(Unsigned, a) == reinterpret(Unsigned, b) ? nothing : "$path: $(repr(a)) vs $(repr(b)) (bits differ)"
    elseif a isa Union{Number, Symbol, Nothing, AbstractString, Char}
        return isequal(a, b) ? nothing : "$path: $(_short(a)) vs $(_short(b))"
    elseif a isa AbstractArray
        size(a) == size(b) || return "$path: size $(size(a)) vs $(size(b))"
        for i in eachindex(a, b)
            d = _spec_diff(a[i], b[i], "$path[$i]", ignore)
            isnothing(d) || return d
        end
        return nothing
    elseif a isa AbstractDict
        ka, kb = sort!(collect(keys(a))), sort!(collect(keys(b)))
        ka == kb || return "$path: keys $(_short(ka)) vs $(_short(kb))"
        for k in ka
            d = _spec_diff(a[k], b[k], "$path[$(repr(k))]", ignore)
            isnothing(d) || return d
        end
        return nothing
    elseif a isa Tuple
        for i in eachindex(a)
            d = _spec_diff(a[i], b[i], "$path[$i]", ignore)
            isnothing(d) || return d
        end
        return nothing
    elseif isstructtype(typeof(a))
        for f in fieldnames(typeof(a))
            f in ignore && continue
            d = _spec_diff(getfield(a, f), getfield(b, f), "$path.$f", ignore)
            isnothing(d) || return d
        end
        return nothing
    end
    return a === b ? nothing : "$path: $(_short(a)) vs $(_short(b))"
end
