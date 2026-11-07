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
# columns of `bound` (CHD head, GHB bhead) are converted to offsets on the host in Float64
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
support yet (WEL `AUTO_FLOW_REDUCE`, MVR) and `ArgumentError` for kinds without a
formulation.
"""
function check_boundary_spec(spec::BoundarySpec{K}) where {K}
    K in (:CHD, :WEL, :RCH, :GHB) ||
        throw(ArgumentError("$(spec.name): boundary package $K has no engine formulation yet"))
    spec.mover && throw(MF6UnsupportedError(spec.file, "unsupported: MOVER option of $(spec.name)"))
    K === :WEL && spec.auto_flow_reduce != 0 &&
        throw(MF6UnsupportedError(spec.file, "unsupported: AUTO_FLOW_REDUCE of $(spec.name) (needs convertible cells, T6.1)"))
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
first one), `nodelist_host`, `bound_host` (Float64 values before the cast to `T`),
`boundname`.
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
    const bound_host::Matrix{Float64}
    const boundname::Vector{String}
end

function BoundaryPackage{K, T}(backend::Backend, spec::BoundarySpec{K}, conn::Connectivity, dis::AbstractDisSpec) where {K, T <: AbstractFloat}
    check_boundary_spec(spec)
    grid = grid_dims(dis)
    maxbound = spec.readasarrays ? prod(grid.mshape[2:end]) : Int(spec.maxbound)
    ncol = length(bound_columns(BoundaryKind{K}()))
    zi(n) = KernelAbstractions.zeros(backend, Int32, n)
    zt(dims...) = KernelAbstractions.zeros(backend, T, dims...)
    nchd = K === :CHD ? maxbound : 0
    return BoundaryPackage(
        spec, conn, grid, zi(maxbound), zt(ncol, maxbound), zt(maxbound), zt(maxbound), zt(maxbound),
        zt(nchd), zt(nchd), KernelAbstractions.ones(backend, Int32, conn.nodes + 1), zi(maxbound), 0, 0,
        zeros(Int32, maxbound), zeros(Float64, ncol, maxbound), String[],
    )
end

BoundaryPackage(backend::Backend, ::Type{T}, spec::BoundarySpec{K}, conn::Connectivity, dis::AbstractDisSpec) where {K, T} =
    BoundaryPackage{K, T}(backend, spec, conn, dis)

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

# Fill `pkg.nodelist_host`, `pkg.bound_host` and `pkg.boundname` from PERIOD block `ib`;
# returns nbound.
function _period_records!(pkg::BoundaryPackage{K}, ib::Integer) where {K}
    spec = pkg.spec
    kind = BoundaryKind{K}()
    cols = bound_columns(kind)
    mcol = mult_column(kind)
    imult = Int(spec.auxmultcol)
    empty!(pkg.boundname)
    data = spec.periods.data[ib]
    if data isa ArrayPeriod
        isempty(data.tas) ||
            throw(MF6UnsupportedError(spec.file, "unsupported: TIMEARRAYSERIES in $(spec.name) (time series arrive with T6.3)"))
        nodeu = array_nodes(data, pkg.grid)
        n = length(nodeu)
        n <= length(pkg.nodelist_host) || throw(ArgumentError("$(spec.name): $n cells for $(length(pkg.nodelist_host)) records"))
        for i in 1:n
            pkg.nodelist_host[i] = _reduced(pkg.conn, nodeu[i])
            for (c, name) in enumerate(cols)
                v = data.values[name][i]
                imult > 0 && c == mcol && (v *= data.auxvar[imult, i])
                pkg.bound_host[c, i] = v
            end
        end
    else
        p = data::PeriodList
        isempty(p.ts) ||
            throw(MF6UnsupportedError(spec.file, "unsupported: time series $(first(p.ts).name) in $(spec.name) (time series arrive with T6.3)"))
        n = length(p.nodeu)
        n <= length(pkg.nodelist_host) ||
            throw(MF6InputError(spec.file, 0, "$(spec.name): $n records in a PERIOD block, but MAXBOUND is $(length(pkg.nodelist_host))"))
        rows = map(cols) do name
            c = list_column(spec.schema, name)
            isnothing(c) && throw(ArgumentError("$(spec.name): no column $name in the list schema"))
            c.first
        end
        for i in 1:n
            pkg.nodelist_host[i] = _reduced(pkg.conn, p.nodeu[i])
            for c in eachindex(cols)
                v = p.bound[rows[c], i]
                imult > 0 && c == mcol && (v *= p.auxvar[imult, i])
                pkg.bound_host[c, i] = v
            end
        end
        append!(pkg.boundname, p.boundname)
    end
    return n
end

"""
    apply_period!(pkg::BoundaryPackage, kper, r; ibound = nothing) -> Bool

Apply the period data of stress period `kper` (`bnd_rp`): the last PERIOD block with
number `<= kper` (no block yet: no records). When that block is already in force nothing
changes and `false` is returned. Otherwise, on the host in Float64: convert cellids to
reduced nodes, apply the MULTIPLIER auxiliary variable, convert the head-like columns to
offsets from the references `r` (`to_offsets`), and for RCH without FIXED_CELL move each
record whose cell has `ibound == 0` to the highest active cell below (`rch_cf`,
`highest_active`; `ibound` is the host IBOUND by reduced node, `nothing` when every cell
is active). Then cast to `T`, copy to the device and rebuild the node→record map.
Returns `true`.

Time-series values (`TsRef`, TIMEARRAYSERIES) are rejected (`MF6UnsupportedError`) until
T6.3.
"""
function apply_period!(pkg::BoundaryPackage{K, T}, kper::Integer, r::AbstractVector{Float64}; ibound::Union{Nothing, AbstractVector{<:Integer}} = nothing) where {K, T}
    length(r) == pkg.conn.nodes || throw(ArgumentError("apply_period!: length(r) = $(length(r)), expected $(pkg.conn.nodes)"))
    ib = period_block(pkg.spec.periods, kper)
    ib == pkg.iblock && return false
    n = ib == 0 ? 0 : _period_records!(pkg, ib)
    nodes = view(pkg.nodelist_host, 1:n)
    if K === :RCH && !pkg.spec.fixed_cell && !isnothing(ibound)
        for i in 1:n
            nodes[i] > 0 && ibound[nodes[i]] == 0 && (nodes[i] = highest_active(pkg.conn, nodes[i], ibound))
        end
    end
    for c in head_columns(BoundaryKind{K}())
        pkg.bound_host[c, 1:n] .= to_offsets(r, view(pkg.bound_host, c, 1:n), nodes)
    end
    pkg.nbound = n
    pkg.iblock = ib
    _upload_records!(pkg)
    return true
end

# Copy the host records of `pkg` to the device and rebuild the node→record map.
function _upload_records!(pkg::BoundaryPackage{K, T}) where {K, T}
    n = pkg.nbound
    nodes = pkg.conn.nodes
    # Counting sort of the records by node (stable: record order within a node).
    ptr = zeros(Int32, nodes + 1)
    for i in 1:n
        nd = pkg.nodelist_host[i]
        nd > 0 && (ptr[nd + 1] += Int32(1))
    end
    ptr[1] = 1
    for nd in 1:nodes
        ptr[nd + 1] += ptr[nd]
    end
    rec = zeros(Int32, length(pkg.nodelist_host))
    next = ptr[1:nodes]
    for i in 1:n
        nd = pkg.nodelist_host[i]
        nd > 0 || continue
        rec[next[nd]] = i
        next[nd] += Int32(1)
    end
    copyto!(pkg.nodelist, pkg.nodelist_host)
    copyto!(pkg.bound, T.(pkg.bound_host))
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
    boundary_formulate_kernel!(backend)(
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
    boundary_add_kernel!(backend)(amat, b, idiag, pkg.hcof, pkg.rhs, pkg.nodeptr, pkg.noderec; ndrange = pkg.conn.nodes)
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
    boundary_simvals_kernel!(backend)(pkg.simvals, pkg.nodelist, pkg.hcof, pkg.rhs, h, ibound; ndrange = pkg.nbound)
    return pkg
end

"""
    synchronize_boundaries(pkg::BoundaryPackage)

Wait for the kernels launched on the backend of `pkg`.
"""
synchronize_boundaries(pkg::BoundaryPackage) = KernelAbstractions.synchronize(get_backend(pkg.hcof))
