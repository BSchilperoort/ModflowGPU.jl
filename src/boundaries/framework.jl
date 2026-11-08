# Boundary packages without atomics (PLAN.md section 3.4; BoundaryPackage.f90).
#
# Each list package keeps MF6's layout: `nodelist[nbound]` (reduced nodes), `bound[ncol,
# nbound]`, `hcof[nbound]`, `rhs[nbound]`, `simvals[nbound]`. Per outer iteration:
#
# 1. `formulate!` runs one work item per record (`bnd_cf` and the package `*_cf`) and
#    computes `hcof`/`rhs` from the current offset head;
# 2. `add_to_system!` runs one work item per node over a node→record CSR map built on the
#    host when period data are applied (`bnd_fc`), and adds the records of that node in
#    record order to the diagonal and the right-hand side. No atomics; the summation order
#    is MF6's, so CPU and GPU results agree to the last bit;
# 3. `calc_simvals!` computes the flow into the model per record (`bnd_cq_simrate`).
#
# Heads are offsets from the per-cell references `r` (section 3.7, rule 1). Head-like
# columns of `bound` (CHD head, GHB bhead, RIV stage and rbot, DRN elev, EVT surface) are converted to offsets on the host in Float64
# when period data are applied (`to_offsets`), so `rhs` holds `rhs_mf6 − hcof·r[n]` and
# `hcof·h' − rhs` equals MF6's `hcof·h − rhs`. The MULTIPLIER auxiliary variable
# (`AUXMULTNAME`) is applied on the host in Float64 at the same time, before the offset
# (MF6 multiplies on use, e.g. `head_mult`, `cond_mult`; the values are the same because
# time series are not supported yet: T6.3 must re-apply both when a series updates a
# value).

"""
    BoundaryKind{K}

Singleton for the package kind `K` (`:CHD`, `:WEL`, `:RCH`, `:GHB`, ...), passed to the
kernels to select the package formulation.
"""
struct BoundaryKind{K} end

"""
    bound_columns(::BoundaryKind) -> Tuple of String

The columns of `bound` of a package kind: DFN names of the PERIOD block list columns
(the READASARRAYS array for RCHA).
"""
function bound_columns end

"""
    bound_layout(kind::BoundaryKind, spec::BoundarySpec) -> Vector

Where each column of `bound` comes from, for the package `spec`: a `(name, j)` tuple for
element `j` of the PERIOD list column (or READASARRAYS array) `name`, `(:aux, iaux)` for
auxiliary variable `iaux`, or a `Float64` constant. The default is one `(name, 1)` per
`bound_columns(kind)`; DRN (AUXDEPTHNAME) and EVT (NSEG, SURF_RATE_SPECIFIED) override it.
"""
bound_layout(kind::BoundaryKind, ::BoundarySpec) = Any[(name, 1) for name in bound_columns(kind)]

"""
    check_records(kind::BoundaryKind, pkg, bound, n)

Check the first `n` host records of `pkg` after period data are read: `bound[c, i]` are
the values with the MULTIPLIER applied, before offsets (Float64, a `BoundSource`; the
package `*_ck` routine). Throws `MF6InputError`. The default checks nothing.
"""
check_records(::BoundaryKind, pkg, bound, n) = nothing

"""
    head_columns(::BoundaryKind) -> Tuple of Int

The columns of `bound` that hold heads or elevations; they are stored as offsets from the
cell reference (PLAN.md section 3.4).
"""
head_columns(::BoundaryKind) = ()

"""
    mult_column(::BoundaryKind) -> Int

The column of `bound` that the MULTIPLIER auxiliary variable (`AUXMULTNAME`) scales
(MF6's `iscloc`).
"""
mult_column(::BoundaryKind) = 1

"""
    check_boundary_spec(spec::BoundarySpec)

Throw `MF6UnsupportedError` for options of a stress package that the engine does not
support yet (WEL `AUTO_FLOW_REDUCE`, MVR), `MF6InputError` for invalid EVT dimensions
(NSEG < 1, NSEG > 1 or SURF_RATE_SPECIFIED with READASARRAYS) and `ArgumentError` for
kinds without a formulation.
"""
function check_boundary_spec(spec::BoundarySpec{K}) where {K}
    K in (:CHD, :WEL, :RCH, :GHB, :RIV, :DRN, :EVT) ||
        throw(ArgumentError("$(spec.name): boundary package $K has no engine formulation yet"))
    spec.mover && throw(MF6UnsupportedError(spec.file, "unsupported: MOVER option of $(spec.name)"))
    K === :WEL && spec.auto_flow_reduce != 0 &&
        throw(MF6UnsupportedError(spec.file, "unsupported: AUTO_FLOW_REDUCE of $(spec.name) (needs convertible cells, T6.1)"))
    K === :EVT && _check_evt_spec(spec)
    return nothing
end

"""
    BoundaryPackage{K, T}(backend, spec::BoundarySpec{K}, conn::Connectivity, dis::AbstractDisSpec)
    BoundaryPackage(backend, T, spec, conn, dis)

A stress package of kind `K` on the device in precision `T` (PLAN.md section 3.4). Arrays
are allocated for `maxbound` records (NCPL for READASARRAYS) and filled by
`apply_period!`.

Device arrays: `nodelist` (reduced node per record, 0 when the record has no cell),
`bound` `(ncol, maxbound)` (multiplier applied, head-like columns as offsets), `hcof`,
`rhs`, `simvals`; `nodeptr` `(nodes + 1)` and `noderec` `(maxbound)`, the node→record CSR
map (records of node `n` are `noderec[nodeptr[n]:nodeptr[n+1]-1]`, in record order). CHD
also has `ratein` and `rateout` (`calc_chd_rate`; empty for other kinds).

Host fields: `spec`, `nbound`, `iblock` (index of the PERIOD block in force, 0 before the
first one), `nodelist_host`, `bound_user` (the values as given: absolute, before the
multiplier; what the stepping API reads and writes, see `bound_user!` and `set_bound!`), `boundname`,
`bound_version` (incremented whenever the values change: new period data or a stepping
API write), `user_stale` (`bound_user` not yet filled from the PERIOD block in force: read
it with `bound_user!`), `staging` (`(ncol, maxbound)` in `T`: the values copied to `bound`, registered
with `pin_memory!`, whose result is kept in `pinned`), `nodeptr_host` / `noderec_host` (the
node→record map on the host), `rrec` / `rrec_for` (the reference `r[nodelist[i]]` per
record, 0 without a cell, for heads of the packages with head-like columns, and the `r` it
was taken from; `nothing` after the cells change). `bound_values(pkg, r)` gives the
Float64 values of `bound`.
"""
mutable struct BoundaryPackage{K, T <: AbstractFloat, VI <: AbstractVector{Int32}, VT <: AbstractVector{T}, MT <: AbstractMatrix{T}}
    const spec::BoundarySpec{K}
    const conn::Connectivity
    const grid::GridDims
    const nodelist::VI
    const bound::MT
    const hcof::VT
    const rhs::VT
    const simvals::VT
    const ratein::VT
    const rateout::VT
    const nodeptr::VI
    const noderec::VI
    nbound::Int
    iblock::Int
    const nodelist_host::Vector{Int32}
    const bound_user::Matrix{Float64}
    const boundname::Vector{String}
    bound_version::Int
    user_stale::Bool
    const staging::Matrix{T}
    const pinned::Any
    const nodeptr_host::Vector{Int32}
    const noderec_host::Vector{Int32}
    const rrec::Vector{Float64}
    rrec_for::Union{Nothing, Vector{Float64}}
end

# The staging buffer of `bound` on the host, and the result of `pin_memory!` (kept so the
# registration lives as long as the package).
function _staging_buffer(backend::Backend, ::Type{T}, ncol, maxbound) where {T}
    staging = zeros(T, ncol, maxbound)
    pinned = isempty(staging) ? nothing : pin_memory!(backend, staging)
    return staging, pinned
end

function BoundaryPackage{K, T}(backend::Backend, spec::BoundarySpec{K}, conn::Connectivity, dis::AbstractDisSpec) where {K, T <: AbstractFloat}
    check_boundary_spec(spec)
    grid = grid_dims(dis)
    maxbound = spec.readasarrays ? prod(grid.mshape[2:end]) : Int(spec.maxbound)
    ncol = length(bound_layout(BoundaryKind{K}(), spec))
    zi(n) = KernelAbstractions.zeros(backend, Int32, n)
    zt(dims...) = KernelAbstractions.zeros(backend, T, dims...)
    nchd = K === :CHD ? maxbound : 0
    staging, pinned = _staging_buffer(backend, T, ncol, maxbound)
    return BoundaryPackage(
        spec, conn, grid, zi(maxbound), zt(ncol, maxbound), zt(maxbound), zt(maxbound), zt(maxbound),
        zt(nchd), zt(nchd), KernelAbstractions.ones(backend, Int32, conn.nodes + 1), zi(maxbound), 0, 0,
        zeros(Int32, maxbound), zeros(Float64, ncol, maxbound), String[], 0, false, staging, pinned,
        zeros(Int32, conn.nodes + 1), zeros(Int32, maxbound),
        zeros(isempty(head_columns(BoundaryKind{K}())) ? 0 : maxbound), nothing,
    )
end

BoundaryPackage(backend::Backend, ::Type{T}, spec::BoundarySpec{K}, conn::Connectivity, dis::AbstractDisSpec) where {K, T} =
    BoundaryPackage{K, T}(backend, spec, conn, dis)

# Adapt (e.g. `adapt(backend, pkg)`): the device arrays move; the host period state is
# copied, so the two packages can take different periods (with a new staging buffer,
# pinned for the target backend).
function Adapt.adapt_structure(to, p::BoundaryPackage)
    bound = adapt(to, p.bound)
    staging, pinned = _staging_buffer(get_backend(bound), eltype(bound), size(p.staging)...)
    copyto!(staging, p.staging)
    return BoundaryPackage(
        p.spec, p.conn, p.grid, adapt(to, p.nodelist), bound, adapt(to, p.hcof), adapt(to, p.rhs),
        adapt(to, p.simvals), adapt(to, p.ratein), adapt(to, p.rateout), adapt(to, p.nodeptr), adapt(to, p.noderec),
        p.nbound, p.iblock, copy(p.nodelist_host), copy(p.bound_user), copy(p.boundname), p.bound_version, p.user_stale,
        staging, pinned, copy(p.nodeptr_host), copy(p.noderec_host), copy(p.rrec), p.rrec_for,
    )
end

boundary_kind(::BoundaryPackage{K}) where {K} = K
Base.eltype(::BoundaryPackage{K, T}) where {K, T} = T
Base.show(io::IO, p::BoundaryPackage{K, T}) where {K, T} =
    print(io, "BoundaryPackage{:", K, ", ", T, "}(\"", p.spec.name, "\", ", p.nbound, " of ", length(p.nodelist_host), " records)")

# ---------------------------------------------------------------------------------------
# Period data (host)
# ---------------------------------------------------------------------------------------

"""
    highest_active(conn::Connectivity, n, ibound) -> Int

The highest active cell at or below reduced node `n` (`highest_active` in
DiscretizationBase.f90): follow the vertical connections (`ihc == 0`) downward from `n`
while `ibound` is 0. Returns the bottom cell of the column when all are inactive.
"""
function highest_active(conn::Connectivity, n::Integer, ibound::AbstractVector{<:Integer})
    n = Int(n)
    while true
        next = 0
        for ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
            m = conn.ja[ii]
            if conn.ihc[conn.jas[ii]] == 0 && m > n
                next = m
                break
            end
        end
        next == 0 && return n  # bottom cell
        n = next
        ibound[n] != 0 && return n
    end
    return
end

# The reduced node of user node `nu` (0 when the cell is not in the reduced grid).
_reduced(conn::Connectivity, nu::Integer) = (nu < 1 || nu > conn.nodesuser) ? Int32(0) : max(conn.nodereduced[nu], Int32(0))

# Period data (T4.6). A new PERIOD block is applied in at most three host passes:
#
# 1. the cells (`_period_nodes!`: cellid → reduced node, then the node→record map,
#    `_upload_structure!`), only when the block's cells differ from those in force;
# 2. the checks (`check_records`) on the values with the multiplier applied;
# 3. one fused pass that reads the values straight from the PERIOD block, applies the
#    multiplier and the offsets in Float64, casts to `T` into the (pinned) staging buffer,
#    and one `copyto!` of it to `bound` (`_stage_bound!`).
#
# `bound_user` (the values as given, for the stepping API) is filled from the block only
# when it is asked for (`bound_user!`), so period data without the stepping API do not
# copy the values twice.

"""
    BoundSource <: AbstractMatrix{Float64}

Lazy host view of the `bound` values of a package's records `(ncol, n)` in Float64,
before the offsets: column `c` is row `rows[c]` of `mats[c]` (a list PERIOD block's
`bound` or `auxvar`, an array period's values, or `bound_user`), or the constant
`consts[c]` when `rows[c] == 0`. With `aux`, the MULTIPLIER (`aux[imult, i]`) is applied to
column `mcol` (`bnd_rp`, `cond_mult`). What the `check_records` hooks see.
"""
struct BoundSource{N, A <: Union{Nothing, Matrix{Float64}}} <: AbstractMatrix{Float64}
    mats::NTuple{N, Matrix{Float64}}
    rows::NTuple{N, Int}
    consts::NTuple{N, Float64}
    n::Int
    aux::A
    imult::Int
    mcol::Int
end
BoundSource(mats::AbstractVector, rows::AbstractVector, consts::AbstractVector, n, aux, imult, mcol) =
    BoundSource(Tuple(mats), Tuple(rows), Tuple(consts), n, aux, imult, mcol)
Base.size(b::BoundSource{N}) where {N} = (N, b.n)
Base.@propagate_inbounds function Base.getindex(b::BoundSource, c::Int, i::Int)
    k = b.rows[c]
    v = k == 0 ? b.consts[c] : b.mats[c][k, i]
    (b.aux !== nothing && c == b.mcol) && (v *= b.aux[b.imult, i])
    return v
end

# The MULTIPLIER aux values of the PERIOD block in force (`(naux, n)`), or `nothing`.
function _mult_aux(pkg::BoundaryPackage)
    imult = Int(pkg.spec.auxmultcol)
    (imult > 0 && pkg.iblock > 0) || return nothing
    return pkg.spec.periods.data[pkg.iblock].auxvar::Matrix{Float64}
end

# A `BoundSource` for the values of `pkg` (`mult`: with the multiplier): from the PERIOD
# block in force while `bound_user` is stale, else from `bound_user`.
function _bound_source(pkg::BoundaryPackage{K}; mult::Bool = true) where {K}
    ncol = size(pkg.bound_user, 1)
    aux = mult ? _mult_aux(pkg) : nothing
    imult = Int(pkg.spec.auxmultcol)
    mcol = mult_column(BoundaryKind{K}())
    if !pkg.user_stale
        return BoundSource(fill(pkg.bound_user, ncol), collect(1:ncol), zeros(ncol), size(pkg.bound_user, 2), aux, imult, mcol)
    end
    spec = pkg.spec
    data = spec.periods.data[pkg.iblock]
    mats = Vector{Matrix{Float64}}(undef, ncol)
    rows = zeros(Int, ncol)
    consts = zeros(ncol)
    for (c, src) in enumerate(bound_layout(BoundaryKind{K}(), spec))
        if src isa Float64
            mats[c] = pkg.bound_user   # unused
            consts[c] = src
        elseif src[1] === :aux
            mats[c] = data.auxvar
            rows[c] = src[2]
        elseif data isa PeriodList
            col = list_column(spec.schema, src[1])
            isnothing(col) && throw(ArgumentError("$(spec.name): no column $(src[1]) in the list schema"))
            mats[c] = data.bound
            rows[c] = col.first + src[2] - 1
        else
            v = (data::ArrayPeriod).values[src[1]]
            mats[c] = reshape(v, 1, length(v))   # shares the memory
            rows[c] = 1
        end
    end
    return BoundSource(mats, rows, consts, pkg.nbound, aux, imult, mcol)
end

"""
    bound_user!(pkg::BoundaryPackage) -> Matrix{Float64}

`pkg.bound_user` (the values as given: absolute, before the multiplier, `(ncol,
maxbound)`), filled from the PERIOD block in force first if a new block was applied since
(T4.6: `apply_period!` reads the values from the block directly). Get it through this
function before reading or changing `bound_user`, then call `set_bound!`.
"""
function bound_user!(pkg::BoundaryPackage)
    pkg.user_stale || return pkg.bound_user
    _fill_user!(pkg.bound_user, _bound_source(pkg; mult = false))
    pkg.user_stale = false
    return pkg.bound_user
end

function _fill_user!(user, b::BoundSource)
    @inbounds for i in 1:b.n, c in 1:size(b, 1)
        user[c, i] = b[c, i]
    end
    return user
end

# Time-series input is rejected until T6.3.
function _check_period_ts(spec::BoundarySpec, data)
    if data isa ArrayPeriod
        isempty(data.tas) ||
            throw(MF6UnsupportedError(spec.file, "unsupported: TIMEARRAYSERIES in $(spec.name) (time series arrive with T6.3)"))
    else
        isempty(data.ts) ||
            throw(MF6UnsupportedError(spec.file, "unsupported: time series $(first(data.ts).name) in $(spec.name) (time series arrive with T6.3)"))
    end
    return nothing
end

# Whether two PERIOD blocks of a package give the same cells (user nodes; array input: the
# same layer per cell column), so the reduced nodes and the node→record map carry over.
_same_cells(a::PeriodList, b::PeriodList) = a.nodeu === b.nodeu || a.nodeu == b.nodeu
_same_cells(a::ArrayPeriod, b::ArrayPeriod) = a.layer === b.layer || a.layer == b.layer
_same_cells(a, b) = false

# Fill `pkg.nodelist_host` from the cells of `data`; returns the number of records.
function _period_nodes!(pkg::BoundaryPackage, data::ArrayPeriod)
    nodeu = array_nodes(data, pkg.grid)
    n = length(nodeu)
    n <= length(pkg.nodelist_host) || throw(ArgumentError("$(pkg.spec.name): $n cells for $(length(pkg.nodelist_host)) records"))
    for i in 1:n
        pkg.nodelist_host[i] = _reduced(pkg.conn, nodeu[i])
    end
    return n
end

function _period_nodes!(pkg::BoundaryPackage, p::PeriodList)
    spec = pkg.spec
    n = length(p.nodeu)
    n <= length(pkg.nodelist_host) ||
        throw(MF6InputError(spec.file, 0, "$(spec.name): $n records in a PERIOD block, but MAXBOUND is $(length(pkg.nodelist_host))"))
    for i in 1:n
        nd = _reduced(pkg.conn, p.nodeu[i])
        # cellid_to_nlist (BoundaryPackageExt.f90): a list record in a cell outside the
        # reduced grid (IDOMAIN <= 0) is an error.
        nd > 0 || throw(MF6InputError(spec.file, 0, "$(spec.name): Cell is outside active grid domain (user node $(p.nodeu[i]))."))
        pkg.nodelist_host[i] = nd
    end
    return n
end

# Record i of the staging pass, column c: the multiplier (Float64, in `b`), then the offset
# from the reference of the record's cell (`to_offsets`: `v − r[n]`; records without a cell
# keep their value). The caller casts to T once.
Base.@propagate_inbounds function _staged_value(b::BoundSource, hcols, nodes, r, c, i)
    v = b[c, i]
    if c in hcols
        nd = nodes[i]
        nd > 0 && (v = v - r[nd])
    end
    return v
end

"""
    bound_values(pkg::BoundaryPackage, r) -> Matrix{Float64}

The `bound` of the records in force `(ncol, nbound)` in Float64 before the cast to `T`: the
MULTIPLIER aux variable applied and the head-like columns as offsets from `r`. What
`apply_period!` and `set_bound!` cast and copy to the device (allocates; for inspection
and tests).
"""
function bound_values(pkg::BoundaryPackage{K}, r::AbstractVector{Float64}) where {K}
    b = _bound_source(pkg)
    hcols = head_columns(BoundaryKind{K}())
    return Float64[_staged_value(b, hcols, pkg.nodelist_host, r, c, i) for c in axes(pkg.bound_user, 1), i in 1:pkg.nbound]
end

# Records i in `range`: staging[:, i] = T(multiplier and offsets of record i), with the
# record references `rrec` (`r[nodes[i]]`, 0 without a cell: `v − 0.0 == v`, so this is
# `_staged_value`).
function _stage_range!(staging::AbstractMatrix{T}, b::BoundSource{N}, hcols, rrec, range) where {T, N}
    @inbounds for i in range
        ref = isempty(hcols) ? 0.0 : rrec[i]
        for c in 1:N
            v = b[c, i]
            c in hcols && (v -= ref)
            staging[c, i] = T(v)
        end
    end
    return nothing
end

# The matrix whose first `ncol·n` values are `bound` as it is, when no pass is needed:
# `T == Float64`, no multiplier, no head-like columns and one source matrix with exactly
# the columns of `bound` (e.g. RCH and WEL lists in `:double`); otherwise `nothing`.
function _direct_source(::Type{T}, b::BoundSource{N}, hcols) where {T, N}
    (T === Float64 && b.aux === nothing && isempty(hcols)) || return nothing
    m = b.mats[1]
    size(m, 1) == N || return nothing
    for c in 1:N
        (b.mats[c] === m && b.rows[c] == c) || return nothing
    end
    return m
end

# The record references `pkg.rrec[1:nbound]` for `r` (recomputed when the cells or `r`
# changed).
function _record_refs!(pkg::BoundaryPackage{K}, r::AbstractVector{Float64}) where {K}
    isempty(head_columns(BoundaryKind{K}())) && return pkg.rrec
    pkg.rrec_for === r && return pkg.rrec
    for i in 1:pkg.nbound
        nd = pkg.nodelist_host[i]
        pkg.rrec[i] = nd > 0 ? r[nd] : 0.0
    end
    pkg.rrec_for = r
    return pkg.rrec
end

# Records per task of the threaded staging pass.
const STAGING_CHUNK = 1 << 16

# The fused host pass for the records in force (Float64 multiplier and offsets, one cast to
# T, into the preallocated staging buffer; threaded when Julia has threads), then one
# `copyto!` of the `ncol·nbound` values to the device. Elementwise, so the result does not
# depend on the number of threads.
function _stage_bound!(pkg::BoundaryPackage{K, T}, r::AbstractVector{Float64}) where {K, T}
    n = pkg.nbound
    n == 0 && return pkg
    # A copy from pinned memory may still be in flight: wait before overwriting the buffer.
    isnothing(pkg.pinned) || KernelAbstractions.synchronize(get_backend(pkg.bound))
    b = _bound_source(pkg)
    hcols = head_columns(BoundaryKind{K}())
    m = _direct_source(T, b, hcols)
    if isnothing(m)
        _stage_records!(pkg.staging, b, hcols, _record_refs!(pkg, r), n)
        m = pkg.staging
    end
    copyto!(pkg.bound, 1, m, 1, size(pkg.staging, 1) * n)
    return pkg
end

function _stage_records!(staging, b::BoundSource, hcols, rrec, n)
    if Threads.nthreads() > 1 && n > 2STAGING_CHUNK
        Threads.@threads for k in 1:cld(n, STAGING_CHUNK)
            _stage_range!(staging, b, hcols, rrec, ((k - 1) * STAGING_CHUNK + 1):min(k * STAGING_CHUNK, n))
        end
    else
        _stage_range!(staging, b, hcols, rrec, 1:n)
    end
    return nothing
end

"""
    apply_period!(pkg::BoundaryPackage, kper, r; ibound = nothing) -> Bool

Apply the period data of stress period `kper` (`bnd_rp`): the last PERIOD block with
number `<= kper` (no block yet: no records). When that block is already in force nothing
changes and `false` is returned. Otherwise, on the host in Float64: convert cellids to
reduced nodes, take the columns of `bound` from the block (`bound_layout`), apply the
MULTIPLIER auxiliary variable, check the records (`check_records`: `riv_ck`, `drn_ck`,
`check_pxdp`), for RCH and EVT without FIXED_CELL move each record whose cell has
`ibound == 0` to the highest active cell below (`rch_cf`, `evt_cf`, `highest_active`), and
convert the head-like columns to offsets from the references `r` of the (moved) cell
(`to_offsets`; `ibound` is the host IBOUND by reduced node, `nothing` when every cell
is active). Then cast to `T`, copy to the device and rebuild the node→record map.
Returns `true`.

When the new block has the same cells as the block in force (array input with the same
layers; ModflowCoupling changing only the values), the reduced nodes, the highest-active
moves and the node→record map carry over and only `bound` is copied (T4.6); the records
are still checked. This fast path is not taken for RCH and EVT without FIXED_CELL when an
`ibound` is given. `bound_user` is filled from the block on demand (`bound_user!`).

Time-series values (`TsRef`, TIMEARRAYSERIES) are rejected (`MF6UnsupportedError`) until
T6.3.
"""
function apply_period!(pkg::BoundaryPackage, kper::Integer, r::AbstractVector{Float64}; ibound::Union{Nothing, AbstractVector{<:Integer}} = nothing)
    return _apply_period!(pkg, kper, r, ibound) !== :unchanged
end

# `apply_period!`; returns `:unchanged` (block in force), `:values` (same cells, new
# values) or `:structure` (new cells: nodes and the node→record map rebuilt).
function _apply_period!(pkg::BoundaryPackage{K, T}, kper::Integer, r::AbstractVector{Float64}, ibound) where {K, T}
    length(r) == pkg.conn.nodes || throw(ArgumentError("apply_period!: length(r) = $(length(r)), expected $(pkg.conn.nodes)"))
    ib = period_block(pkg.spec.periods, kper)
    ib == pkg.iblock && return :unchanged
    spec = pkg.spec
    moves = K in (:RCH, :EVT) && !spec.fixed_cell && !isnothing(ibound)
    data = ib == 0 ? nothing : spec.periods.data[ib]
    isnothing(data) || _check_period_ts(spec, data)
    same = !moves && ib > 0 && pkg.iblock > 0 && _same_cells(spec.periods.data[pkg.iblock], data)
    n = same ? pkg.nbound : isnothing(data) ? 0 : _period_nodes!(pkg, data)
    empty!(pkg.boundname)
    data isa PeriodList && append!(pkg.boundname, data.boundname)
    pkg.bound_version += 1
    pkg.iblock = ib
    pkg.nbound = n
    pkg.user_stale = !isnothing(data)
    check_records(BoundaryKind{K}(), pkg, _bound_source(pkg), n)
    if moves
        nodes = pkg.nodelist_host
        for i in 1:n
            nodes[i] > 0 && ibound[nodes[i]] == 0 && (nodes[i] = highest_active(pkg.conn, nodes[i], ibound))
        end
    end
    same || (_upload_structure!(pkg); pkg.rrec_for = nothing)
    _stage_bound!(pkg, r)
    return same ? :values : :structure
end

"""
    set_bound!(pkg::BoundaryPackage, r) -> pkg

After a change of `pkg.bound_user` (the stepping API, `set_value` of `BOUND`; get the array
with `bound_user!`): apply the MULTIPLIER aux variable of the PERIOD block in force and
convert the head-like columns to offsets from `r`, exactly as `apply_period!` does for
period data (host, Float64), then cast to `T` and copy `bound` to the device (one fused
pass into the staging buffer and one copy). Records are not checked (`riv_ck` etc. run in
`bnd_rp` only, as in MF6) and are not moved (RCH/EVT keep their cells).
"""
function set_bound!(pkg::BoundaryPackage, r::AbstractVector{Float64})
    pkg.user_stale = false
    pkg.bound_version += 1
    _stage_bound!(pkg, r)
    return pkg
end

# Copy the node list of `pkg` to the device and rebuild the node→record map (counting sort
# of the records by node; stable: record order within a node). Uses the host buffers
# `nodeptr_host` and `noderec_host`.
function _upload_structure!(pkg::BoundaryPackage)
    n = pkg.nbound
    nodes = pkg.conn.nodes
    ptr = pkg.nodeptr_host
    rec = pkg.noderec_host
    fill!(ptr, Int32(0))
    fill!(rec, Int32(0))
    for i in 1:n
        nd = pkg.nodelist_host[i]
        nd > 0 && (ptr[nd + 1] += Int32(1))
    end
    ptr[1] = 1
    for nd in 1:nodes
        ptr[nd + 1] += ptr[nd]
    end
    # Fill: `ptr[nd]` advances to the end of node nd's range, then shift back.
    for i in 1:n
        nd = pkg.nodelist_host[i]
        nd > 0 || continue
        rec[ptr[nd]] = i
        ptr[nd] += Int32(1)
    end
    for nd in nodes:-1:2
        ptr[nd] = ptr[nd - 1]
    end
    ptr[1] = 1
    copyto!(pkg.nodelist, pkg.nodelist_host)
    copyto!(pkg.nodeptr, ptr)
    copyto!(pkg.noderec, rec)
    return pkg
end

"""
    boundary_ibound(conn::Connectivity, packages) -> Vector{Int32}

The IBOUND array by reduced node for the boundary kernels: 1 for active cells and −1 for
constant-head cells (the nodes of the CHD packages among `packages`, `chd_rp`). A cell
given twice as constant head is an error ("Cell is already a constant head").
"""
function boundary_ibound(conn::Connectivity, packages)
    ibound = ones(Int32, conn.nodes)
    for pkg in packages
        boundary_kind(pkg) === :CHD || continue
        for i in 1:pkg.nbound
            n = pkg.nodelist_host[i]
            n > 0 || continue
            ibound[n] < 0 &&
                throw(MF6InputError(pkg.spec.file, 0, "Cell is already a constant head (reduced node $n, user node $(conn.nodeuser[n]))."))
            ibound[n] = -1
        end
    end
    return ibound
end

# ---------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------

"""
    formulate_record(kind, i, n, bound, h, area) -> (hcof, rhs)

`hcof` and `rhs` of record `i` in active cell `n` (the package `*_cf` routine), with the
current offset heads `h` and the cell areas `area`. Device-safe; one method per kind.
"""
function formulate_record end

@kernel function boundary_formulate_kernel!(hcof, rhs, kind, @Const(nodelist), @Const(bound), @Const(h), @Const(ibound), @Const(area))
    i = @index(Global)
    @inbounds begin
        n = nodelist[i]
        if n > 0 && ibound[n] > 0
            hc, r = formulate_record(kind, i, n, bound, h, area)
            hcof[i] = hc
            rhs[i] = r
        else
            # No cell, constant-head or inactive cell: no contribution (bnd_cf).
            hcof[i] = zero(eltype(hcof))
            rhs[i] = zero(eltype(rhs))
        end
    end
end

"""
    formulate!(pkg::BoundaryPackage, h, ibound, area) -> pkg

Compute `hcof` and `rhs` of every record (`bnd_cf` / `*_cf`) from the offset heads `h`,
the IBOUND array `ibound` (`boundary_ibound`) and the cell areas `area` (device arrays by
reduced node). Records in constant-head or inactive cells get zeros.
"""
function formulate!(pkg::BoundaryPackage{K}, h, ibound, area) where {K}
    pkg.nbound == 0 && return pkg
    backend = get_backend(pkg.hcof)
    boundary_formulate_kernel!(backend, _WG)(
        pkg.hcof, pkg.rhs, BoundaryKind{K}(), pkg.nodelist, pkg.bound, h, ibound, area; ndrange = pkg.nbound,
    )
    return pkg
end

@kernel function boundary_add_kernel!(amat, b, @Const(idiag), @Const(hcof), @Const(rhs), @Const(nodeptr), @Const(noderec))
    n = @index(Global)
    @inbounds begin
        k1 = nodeptr[n]
        k2 = nodeptr[n + 1] - Int32(1)
        if k2 >= k1
            id = idiag[n]
            d = amat[id]
            s = b[n]
            for k in k1:k2
                i = noderec[k]
                d += hcof[i]
                s += rhs[i]
            end
            amat[id] = d
            b[n] = s
        end
    end
end

"""
    add_to_system!(amat, b, idiag, pkg::BoundaryPackage) -> amat

Add `hcof` to the diagonal `amat[idiag[n]]` and `rhs` to `b[n]` of every node with
records (`bnd_fc`), one work item per node, records in order. For MF6's CSR layout
`idiag = ia[1:nodes]`; for a separate diagonal vector `idiag = 1:nodes`.
"""
function add_to_system!(amat, b, idiag, pkg::BoundaryPackage)
    pkg.nbound == 0 && return amat
    backend = get_backend(b)
    boundary_add_kernel!(backend, _WG)(amat, b, idiag, pkg.hcof, pkg.rhs, pkg.nodeptr, pkg.noderec; ndrange = pkg.conn.nodes)
    return amat
end

@kernel function boundary_simvals_kernel!(simvals, @Const(nodelist), @Const(hcof), @Const(rhs), @Const(h), @Const(ibound))
    i = @index(Global)
    @inbounds begin
        n = nodelist[i]
        q = zero(eltype(simvals))
        if n > 0 && ibound[n] > 0
            q = hcof[i] * h[n] - rhs[i]
        end
        simvals[i] = q
    end
end

"""
    calc_simvals!(pkg::BoundaryPackage, h, ibound) -> pkg

The flow from the package into the model per record, `simvals[i] = hcof·h' − rhs`
(`bnd_cq_simrate`; equal to MF6's `hcof·h − rhs`, see the file header), zero in
constant-head and inactive cells. Not for CHD: use `calc_chd_rate!`.
"""
function calc_simvals!(pkg::BoundaryPackage{K}, h, ibound) where {K}
    K === :CHD && throw(ArgumentError("calc_simvals!: CHD flows come from calc_chd_rate!"))
    pkg.nbound == 0 && return pkg
    backend = get_backend(pkg.simvals)
    boundary_simvals_kernel!(backend, _WG)(pkg.simvals, pkg.nodelist, pkg.hcof, pkg.rhs, h, ibound; ndrange = pkg.nbound)
    return pkg
end

"""
    synchronize_boundaries(pkg::BoundaryPackage)

Wait for the kernels launched on the backend of `pkg`.
"""
synchronize_boundaries(pkg::BoundaryPackage) = KernelAbstractions.synchronize(get_backend(pkg.hcof))
