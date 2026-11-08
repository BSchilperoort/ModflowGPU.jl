# Multicolour symmetric Gauss–Seidel and ILU0 preconditioners (T4.2), on the scaled
# operator `Â` of the Krylov solver (src/solver/precond/precond.jl).
#
# Colouring (host, once): DIS grids are coloured red–black by the parity of `i + j + k`
# (`dis_parity_colors`; checked, since vertical pass-through cells connect cells two
# layers apart), other grids greedily in natural order (`greedy_colors`). Nodes of one
# colour are not connected, so each colour is one parallel kernel. The system is stored
# once more in Tp, in colour order: row `i` of the coloured CSR is node `perm[i]`, with its
# diagonal first, then the entries of lower colours (sorted by colour), then those of higher
# colours (from `cup[i]` on); columns keep the original node numbers.
#
# - `:sgs`: symmetric Gauss–Seidel in colour order from `z = 0`, `M = (D + L)·D⁻¹·(D + U)`:
#   a forward sweep over the colours (`(D + L)·y = r`) and a backward sweep from the
#   second-to-last colour down (the last colour's backward update equals its forward one).
#   SPD for SPD `Â`, so it can precondition CG.
# - `:ilu0`: ILU(0) of `Â` in colour order (IKJ, as ims_base_pcilu0 in MF6's natural order,
#   with optional MILU relaxation, `MULTICOLOR_OPTIONS`), factorised colour by colour on the device at every setup;
#   applied as a forward (unit lower) and a backward (upper) solve per colour
#   (ims_base_ilu0a). For symmetric `Â`, `L·U = L·D·Lᵀ` is symmetric. A pivot that changes
#   sign or vanishes is replaced by the unfactorised diagonal (MF6 restarts the
#   factorisation with a larger diagonal shift DELTA instead).
# For two colours (red–black) ILU0 only modifies the diagonal of the second colour.

"""
    greedy_colors(ia, ja) -> Vector{Int32}

Greedy colouring of the graph of a CSR structure (diagonal entries ignored) in natural node
order: each node gets the smallest colour (from 1) not used by a lower-numbered neighbour.
"""
function greedy_colors(ia::AbstractVector{<:Integer}, ja::AbstractVector{<:Integer})
    n = length(ia) - 1
    colors = zeros(Int32, n)
    used = zeros(Int, 0)  # used[c] == node: colour c is taken by a neighbour of node
    for i in 1:n
        for ii in ia[i]:(ia[i + 1] - 1)
            j = ja[ii]
            c = j != i ? colors[j] : 0
            c > 0 && (used[c] = i)
        end
        c = findfirst(!=(i), used)
        if isnothing(c)
            push!(used, 0)
            c = length(used)
        end
        colors[i] = c
    end
    return colors
end

"""
    is_valid_coloring(ia, ja, colors) -> Bool

Whether no two connected nodes share a colour.
"""
function is_valid_coloring(ia::AbstractVector{<:Integer}, ja::AbstractVector{<:Integer}, colors::AbstractVector{<:Integer})
    n = length(ia) - 1
    length(colors) == n || return false
    all(>(0), colors) || return false
    for i in 1:n, ii in ia[i]:(ia[i + 1] - 1)
        j = ja[ii]
        j != i && colors[j] == colors[i] && return false
    end
    return true
end

"""
    dis_parity_colors(conn::Connectivity, dis::DisSpec) -> Vector{Int32}

Red–black colours of the reduced nodes of a DIS grid: `1 + (i + j + k) mod 2` from the
user node's column `j`, row `i` and layer `k`. Not a valid colouring when vertical
pass-through cells (IDOMAIN < 0) connect cells two layers apart (`grid_colors` checks).
"""
function dis_parity_colors(conn::Connectivity, dis::DisSpec)
    ncol, nrow = Int(dis.ncol), Int(dis.nrow)
    colors = Vector{Int32}(undef, conn.nodes)
    for n in 1:conn.nodes
        u = Int(conn.nodeuser[n]) - 1
        j = u % ncol
        i = (u ÷ ncol) % nrow
        k = u ÷ (ncol * nrow)
        colors[n] = 1 + (i + j + k) % 2
    end
    return colors
end

"""
    grid_colors(conn::Connectivity, dis) -> Vector{Int32}

The colouring the multicolour preconditioners use: `dis_parity_colors` for a DIS grid
when it is valid, otherwise `greedy_colors`.
"""
function grid_colors(conn::Connectivity, dis)
    if dis isa DisSpec
        c = dis_parity_colors(conn, dis)
        is_valid_coloring(conn.ia, conn.ja, c) && return c
    end
    return greedy_colors(conn.ia, conn.ja)
end

"""
    MULTICOLOR_OPTIONS

Settings of `:mc_ilu0` (a `Ref` to a NamedTuple, read when the preconditioner is built):
`relax`, the MILU relaxation factor (0: ILU0; IMS RELAXATION_FACTOR).
"""
const MULTICOLOR_OPTIONS = Ref((; relax = 0.0))

"""
    MulticolorPreconditioner{Tp}

Multicolour symmetric Gauss–Seidel (`kind = :sgs`) or ILU0 (`:ilu0`) on the scaled
operator (src/solver/precond/multicolor.jl). Host: `cptr` (rows `cptr[c]:cptr[c+1]-1` of
the coloured CSR have colour `c`). Device (Int32): `perm` (coloured row → node), `iperm`,
`cia`, `cja`, `cup` (first entry of a higher colour in each row), `csrc` (position of each
entry in the original CSR); in `Tp`: `vals` (`Â` or the ILU factors) and `dinv` (inverse
diagonal or pivots, by coloured row); `relax` (MILU relaxation factor, `MULTICOLOR_OPTIONS`).

    MulticolorPreconditioner(backend, csr, kind; Ts, Tp, colors = nothing, relax)

`colors` is a host vector of colours `1:ncolors` per node (`greedy_colors` of the structure
when `nothing`).
"""
struct MulticolorPreconditioner{VI <: AbstractVector{Int32}, VP <: AbstractVector, C <: CsrStructure} <: AbstractPreconditioner
    csr::C
    ilu::Bool
    cptr::Vector{Int}
    perm::VI
    iperm::VI
    cia::VI
    cja::VI
    cup::VI
    csrc::VI
    vals::VP
    dinv::VP
    relax::Float64
end

ncolors(P::MulticolorPreconditioner) = length(P.cptr) - 1

function MulticolorPreconditioner(
        backend::Backend, csr::CsrStructure, kind::Symbol; Ts::Type, Tp::Type, colors = nothing,
        relax::Real = MULTICOLOR_OPTIONS[].relax,
    )
    kind in (:sgs, :ilu0) || throw(ArgumentError("unknown multicolour preconditioner $kind (expected :sgs or :ilu0)"))
    ia = Array(csr.ia)
    ja = Array(csr.ja)
    n = Int(csr.nodes)
    colors = isnothing(colors) ? greedy_colors(ia, ja) : Int32.(colors)
    is_valid_coloring(ia, ja, colors) || throw(ArgumentError("invalid colouring: connected nodes share a colour"))
    nc = Int(maximum(colors; init = Int32(0)))
    perm = sortperm(colors; alg = Base.Sort.DEFAULT_STABLE)
    cnt = zeros(Int, nc)
    for c in colors
        cnt[c] += 1
    end
    cptr = cumsum(vcat(1, cnt))
    iperm = zeros(Int32, n)
    iperm[perm] .= Int32.(1:n)
    cia = Vector{Int32}(undef, n + 1)
    cja = Vector{Int32}(undef, length(ja))
    csrc = Vector{Int32}(undef, length(ja))
    cup = Vector{Int32}(undef, n)
    pos = 1
    for i in 1:n
        node = perm[i]
        cn = colors[node]
        cia[i] = pos
        i0, i1 = Int(ia[node]), Int(ia[node + 1]) - 1
        # diagonal, then lower colours (by colour), then higher colours
        off = [ii for ii in (i0 + 1):i1]
        lower = sort!([ii for ii in off if colors[ja[ii]] < cn]; by = ii -> colors[ja[ii]], alg = Base.Sort.DEFAULT_STABLE)
        upper = [ii for ii in off if colors[ja[ii]] > cn]
        for ii in Iterators.flatten(((i0,), lower, upper))
            cja[pos] = ja[ii]
            csrc[pos] = ii
            pos += 1
        end
        cup[i] = cia[i] + 1 + length(lower)
    end
    cia[n + 1] = pos
    pos == length(ja) + 1 || throw(ArgumentError("CSR rows must hold their diagonal first"))
    dev(x) = _device_copy(backend, x)
    return MulticolorPreconditioner(
        csr, kind === :ilu0, cptr, dev(Int32.(perm)), dev(iperm), dev(cia), dev(cja), dev(cup), dev(csrc),
        KernelAbstractions.zeros(backend, Tp, length(ja)), KernelAbstractions.zeros(backend, Tp, n), Float64(relax),
    )
end

Adapt.adapt_structure(to, P::MulticolorPreconditioner) = MulticolorPreconditioner(
    adapt(to, P.csr), P.ilu, copy(P.cptr), adapt(to, P.perm), adapt(to, P.iperm), adapt(to, P.cia), adapt(to, P.cja),
    adapt(to, P.cup), adapt(to, P.csrc), adapt(to, P.vals), adapt(to, P.dinv), P.relax,
)

# --- setup

# Scaled values of coloured row i; for SGS also the inverse diagonal.
@inline function _mc_copy_row(i, vals, dinv, perm, cia, cja, csrc, amat, rowfac, colfac, sgs::Bool)
    Tp = eltype(vals)
    Ts = eltype(rowfac)
    @inbounds begin
        n = perm[i]
        rf = rowfac[n]
        for kk in cia[i]:(cia[i + 1] - Int32(1))
            vals[kk] = Tp(rf * Ts(amat[csrc[kk]]) * colfac[cja[kk]])
        end
        if sgs
            d = vals[cia[i]]
            dinv[i] = d != zero(Tp) ? one(Tp) / d : one(Tp)
        end
    end
    return nothing
end

# Position of column j among the diagonal i0 and the entries j0:i1 of a coloured row (0
# when absent).
@inline function _find_col(cja, j, i0, j0, i1)
    @inbounds begin
        cja[i0] == j && return i0
        for jj in j0:i1
            cja[jj] == j && return jj
        end
    end
    return zero(i0)
end

# ILU0 (IKJ) of coloured row i = off + k, after the rows of all lower colours
# (ims_base_pcilu0: fill outside the pattern is dropped, and with MILU relaxation `relax`
# its row sum `rs` is taken off the pivot, `tl = d − relax·rs`).
@inline function _mc_ilu_row(k, off, vals, dinv, iperm, cia, cja, cup, relax)
    Tp = eltype(vals)
    @inbounds begin
        i = off + k
        i0 = cia[i]
        i1 = cia[i + 1] - Int32(1)
        d0 = vals[i0]
        rs = zero(Tp)
        for kk in (i0 + Int32(1)):(cup[i] - Int32(1))
            im = iperm[cja[kk]]
            l = vals[kk] * dinv[im]
            vals[kk] = l
            # a_ij -= l·u_mj for the upper entries j of row m: in row i they are its
            # diagonal or after kk (the earlier entries have colours <= that of m).
            for mm in cup[im]:(cia[im + 1] - Int32(1))
                jj = _find_col(cja, cja[mm], i0, kk + Int32(1), i1)
                if jj != zero(jj)
                    vals[jj] -= l * vals[mm]
                else
                    rs += l * vals[mm]
                end
            end
        end
        d = vals[i0] - relax * rs
        if d == zero(Tp) || signbit(d) != signbit(d0)
            d = d0
        end
        vals[i0] = d
        dinv[i] = d != zero(Tp) ? one(Tp) / d : one(Tp)
    end
    return nothing
end

function precond_setup!(P::MulticolorPreconditioner, solver, amat)
    backend = get_backend(P.vals)
    n = Int(P.csr.nodes)
    _map!(backend, _mc_copy_row, n, P.vals, P.dinv, P.perm, P.cia, P.cja, P.csrc, amat, solver.rowfac, solver.colfac, !P.ilu)
    if P.ilu
        for c in 1:ncolors(P)
            off = Int32(P.cptr[c] - 1)
            _map!(backend, _mc_ilu_row, P.cptr[c + 1] - P.cptr[c], off, P.vals, P.dinv, P.iperm, P.cia, P.cja, P.cup, eltype(P.vals)(P.relax))
        end
    end
    return P
end

# --- application

# Forward: z_n = (r_n − Σ_lower v_nm·z_m), times the inverse diagonal for SGS.
@inline function _mc_forward(k, off, z, r, vals, dinv, perm, cia, cja, cup, sgs::Bool)
    Ts = eltype(z)
    @inbounds begin
        i = off + k
        n = perm[i]
        t = r[n]
        for kk in (cia[i] + Int32(1)):(cup[i] - Int32(1))
            t -= Ts(vals[kk]) * z[cja[kk]]
        end
        z[n] = sgs ? t * Ts(dinv[i]) : t
    end
    return nothing
end

# Backward, ILU0: z_n = (z_n − Σ_upper u_nm·z_m) / u_nn.
@inline function _mc_backward_ilu(k, off, z, vals, dinv, perm, cia, cja, cup)
    Ts = eltype(z)
    @inbounds begin
        i = off + k
        n = perm[i]
        t = z[n]
        for kk in cup[i]:(cia[i + 1] - Int32(1))
            t -= Ts(vals[kk]) * z[cja[kk]]
        end
        z[n] = t * Ts(dinv[i])
    end
    return nothing
end

# Backward, SGS: z_n = (r_n − Σ_{m≠n} a_nm·z_m) / a_nn.
@inline function _mc_backward_sgs(k, off, z, r, vals, dinv, perm, cia, cja)
    Ts = eltype(z)
    @inbounds begin
        i = off + k
        n = perm[i]
        t = r[n]
        for kk in (cia[i] + Int32(1)):(cia[i + 1] - Int32(1))
            t -= Ts(vals[kk]) * z[cja[kk]]
        end
        z[n] = t * Ts(dinv[i])
    end
    return nothing
end

function precond_apply!(z, P::MulticolorPreconditioner, r)
    backend = get_backend(z)
    nc = ncolors(P)
    len(c) = P.cptr[c + 1] - P.cptr[c]
    off(c) = Int32(P.cptr[c] - 1)
    for c in 1:nc
        _map!(backend, _mc_forward, len(c), off(c), z, r, P.vals, P.dinv, P.perm, P.cia, P.cja, P.cup, !P.ilu)
    end
    if P.ilu
        for c in nc:-1:1
            _map!(backend, _mc_backward_ilu, len(c), off(c), z, P.vals, P.dinv, P.perm, P.cia, P.cja, P.cup)
        end
    else
        for c in (nc - 1):-1:1
            _map!(backend, _mc_backward_sgs, len(c), off(c), z, r, P.vals, P.dinv, P.perm, P.cia, P.cja)
        end
    end
    return z
end
