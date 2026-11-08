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
#   with optional MILU relaxation, option `relax`), factorised colour by colour on the device at every setup;
#   applied as a forward (unit lower) and a backward (upper) solve per colour
#   (ims_base_ilu0a). For symmetric `Â`, `L·U = L·D·Lᵀ` is symmetric. A pivot that changes
#   sign or vanishes is replaced by the unfactorised diagonal (MF6 restarts the
#   factorisation with a larger diagonal shift DELTA instead).
# For two colours (red–black) ILU0 only modifies the diagonal of the second colour.
#
# Red–black ILU0 setup (T4.5): `ctr` holds, for each lower entry (n, m) of a second-colour
# row, the position of the transposed entry (m, n) in row m, so the pivot update reads one
# value instead of searching row m (with MILU relaxation the row is still scanned). When the
# off-diagonal entries of the matrix are unchanged since the last setup (the
# `offdiag_version` that `solve!` passes on; e.g. confined models between outer iterations
# and time steps) and the scaling is `:none`, the first-colour rows keep their off-diagonal
# values (unless their row factor changed) and only their pivots are recomputed. Both give
# bitwise the same factors.

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
    MulticolorPreconditioner{Tp}

Multicolour symmetric Gauss–Seidel (`kind = :sgs`) or ILU0 (`:ilu0`) on the scaled
operator (src/solver/precond/multicolor.jl). Host: `cptr` (rows `cptr[c]:cptr[c+1]-1` of
the coloured CSR have colour `c`). Device (Int32): `perm` (coloured row → node), `iperm`,
`cia`, `cja`, `cup` (first entry of a higher colour in each row), `csrc` (position of each
entry in the original CSR); in `Tp`: `vals` (`Â` or the ILU factors) and `dinv` (inverse
diagonal or pivots, by coloured row); `relax` (MILU relaxation factor).

    MulticolorPreconditioner(backend, csr, kind; Ts, Tp, colors = nothing, relax = 0.0)

`relax` (`precond_options` of `KrylovSettings` with `preconditioner = :mc_ilu0`): the
MILU relaxation factor of `:ilu0` (0: ILU0; IMS RELAXATION_FACTOR).

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
    ctr::VI
    rf1::VP
    version::Base.RefValue{Int}
    pending::Base.RefValue{Int}
end

ncolors(P::MulticolorPreconditioner) = length(P.cptr) - 1

function MulticolorPreconditioner(
        backend::Backend, csr::CsrStructure, kind::Symbol; Ts::Type, Tp::Type, colors = nothing,
        relax::Real = 0.0,
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
    lower = Int[]
    for i in 1:n
        node = perm[i]
        cn = colors[node]
        cia[i] = pos
        i0, i1 = Int(ia[node]), Int(ia[node + 1]) - 1
        # diagonal, then lower colours (stable insertion sort by colour), then higher colours
        cja[pos] = ja[i0]
        csrc[pos] = i0
        pos += 1
        empty!(lower)
        for ii in (i0 + 1):i1
            c = colors[ja[ii]]
            c < cn || continue
            push!(lower, ii)
            k = length(lower)
            while k > 1 && colors[ja[lower[k - 1]]] > c
                lower[k], lower[k - 1] = lower[k - 1], lower[k]
                k -= 1
            end
        end
        for ii in lower
            cja[pos] = ja[ii]
            csrc[pos] = ii
            pos += 1
        end
        cup[i] = pos
        for ii in (i0 + 1):i1
            colors[ja[ii]] > cn || continue
            cja[pos] = ja[ii]
            csrc[pos] = ii
            pos += 1
        end
    end
    cia[n + 1] = pos
    pos == length(ja) + 1 || throw(ArgumentError("CSR rows must hold their diagonal first"))
    rb = kind === :ilu0 && nc == 2
    # Red–black: every coloured row is the original row (all neighbours have the other
    # colour and keep their order), which the natural-order setup copy relies on.
    ctr = rb ? _transpose_positions(perm, iperm, cia, cja, cup, cptr[2]) : Int32[]
    dev(x) = _device_copy(backend, x)
    return MulticolorPreconditioner(
        csr, kind === :ilu0, cptr, dev(Int32.(perm)), dev(iperm), dev(cia), dev(cja), dev(cup), dev(csrc),
        KernelAbstractions.zeros(backend, Tp, length(ja)), KernelAbstractions.zeros(backend, Tp, n), Float64(relax),
        dev(ctr), KernelAbstractions.zeros(backend, Tp, rb ? cptr[2] - 1 : 0), Ref(-1), Ref(-1),
    )
end

# For each lower entry kk of a second-colour row i (node n, column m), the position of entry
# (m, n) in the coloured row of m (0 elsewhere).
function _transpose_positions(perm, iperm, cia, cja, cup, first2)
    ctr = zeros(Int32, length(cja))
    for i in first2:length(perm)
        n = perm[i]
        for kk in (cia[i] + 1):(cup[i] - 1)
            im = iperm[cja[kk]]
            for mm in cup[im]:(cia[im + 1] - 1)
                cja[mm] == n && (ctr[kk] = mm)
            end
            ctr[kk] > 0 || throw(ArgumentError("the CSR structure is not structurally symmetric"))
        end
    end
    return ctr
end

Adapt.adapt_structure(to, P::MulticolorPreconditioner) = MulticolorPreconditioner(
    adapt(to, P.csr), P.ilu, copy(P.cptr), adapt(to, P.perm), adapt(to, P.iperm), adapt(to, P.cia), adapt(to, P.cja),
    adapt(to, P.cup), adapt(to, P.csrc), adapt(to, P.vals), adapt(to, P.dinv), P.relax, adapt(to, P.ctr), adapt(to, P.rf1),
    Ref(-1), Ref(-1),
)

_set_offdiag_version!(P::MulticolorPreconditioner, v) = (P.pending[] = v; nothing)

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

# Red–black ILU0, step 1: copy the scaled row of node n, in natural order (coalesced reads
# of `amat`; with two colours the coloured row is the original row, see the constructor).
# First-colour rows also get their pivot (the diagonal). With `keep` (off-diagonal entries
# unchanged, no scaling) a first-colour row only gets its pivot unless its row factor
# changed (`rf1`: the row factor of its last copy).
@inline function _mc_rb_copy(n, n1, keep::Bool, vals, dinv, iperm, cia, rf1, amat, ia, ja, rowfac, colfac)
    Tp = eltype(vals)
    Ts = eltype(rowfac)
    @inbounds begin
        i = iperm[n]
        rf = rowfac[n]
        p0 = cia[i]
        a0 = ia[n]
        len = ia[n + 1] - a0
        first = i <= n1
        diag_only = first && keep && Tp(rf) == rf1[i]
        last = diag_only ? zero(len) : len - one(len)
        for t in zero(len):last
            vals[p0 + t] = Tp(rf * Ts(amat[a0 + t]) * colfac[ja[a0 + t]])
        end
        if first
            rf1[i] = Tp(rf)
            d = vals[p0]
            dinv[i] = d != zero(Tp) ? one(Tp) / d : one(Tp)
        end
    end
    return nothing
end

# Step 2, second-colour row i = off + k: only the diagonal receives fill inside the pattern
# (second-colour nodes are not connected), so each lower entry (n, m) updates
# `d −= l_nm·u_mn` (`u_mn` at `ctr[kk]`) and drops the rest of row m (MILU, `milu`: their
# sum `rs`, found by scanning row m).
@inline function _mc_rb_factor(k, off, vals, dinv, perm, iperm, cia, cja, cup, ctr, relax, milu::Bool)
    Tp = eltype(vals)
    @inbounds begin
        i = off + k
        n = perm[i]
        i0 = cia[i]
        d0 = vals[i0]
        d = d0
        rs = zero(Tp)
        for kk in (i0 + Int32(1)):(cup[i] - Int32(1))
            im = iperm[cja[kk]]
            l = vals[kk] * dinv[im]
            vals[kk] = l
            if milu
                for mm in cup[im]:(cia[im + 1] - Int32(1))
                    if cja[mm] == n
                        d -= l * vals[mm]
                    else
                        rs += l * vals[mm]
                    end
                end
            else
                d -= l * vals[ctr[kk]]
            end
        end
        d -= relax * rs
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
    nc = ncolors(P)
    relax = eltype(P.vals)(P.relax)
    len(c) = P.cptr[c + 1] - P.cptr[c]
    off(c) = Int32(P.cptr[c] - 1)
    if P.ilu && nc == 2
        # Off-diagonal values of the first colour still valid: same matrix off-diagonals,
        # no scaling (column factors 1; the row factors are checked per row).
        v = P.pending[]
        keep = v >= 0 && v == P.version[] && solver.settings.scaling === :none
        csr = P.csr
        _map!(
            backend, _mc_rb_copy, n, Int32(len(1)), keep, P.vals, P.dinv, P.iperm, P.cia, P.rf1, amat, csr.ia, csr.ja,
            solver.rowfac, solver.colfac,
        )
        _map!(backend, _mc_rb_factor, len(2), off(2), P.vals, P.dinv, P.perm, P.iperm, P.cia, P.cja, P.cup, P.ctr, relax, P.relax != 0)
        P.version[] = v
        return P
    end
    _map!(backend, _mc_copy_row, n, P.vals, P.dinv, P.perm, P.cia, P.cja, P.csrc, amat, solver.rowfac, solver.colfac, !P.ilu)
    if P.ilu
        for c in 1:nc
            _map!(backend, _mc_ilu_row, len(c), off(c), P.vals, P.dinv, P.iperm, P.cia, P.cja, P.cup, relax)
        end
    end
    return P
end

# --- application

# Forward: z_n = r_n − Σ_lower v_nm·y_m with the lower-colour values `y` read from `src`
# (`z`, or `r` when all lower neighbours have the first colour, whose `y` is `r`), times
# the inverse diagonal when `scale` (SGS; ILU0 for the last colour, which has no upper
# entries, so its backward step is fused here).
@inline function _mc_forward(k, off, z, src, r, vals, dinv, perm, cia, cja, cup, scale::Bool)
    Ts = eltype(z)
    @inbounds begin
        i = off + k
        n = perm[i]
        t = r[n]
        for kk in (cia[i] + Int32(1)):(cup[i] - Int32(1))
            t -= Ts(vals[kk]) * src[cja[kk]]
        end
        z[n] = scale ? t * Ts(dinv[i]) : t
    end
    return nothing
end

# Backward, ILU0: z_n = (y_n − Σ_upper u_nm·z_m) / u_nn, with `y` = `z` (the forward
# result) or `r` (the first colour of a red–black ordering, whose forward step is y = r).
@inline function _mc_backward_ilu(k, off, z, y, vals, dinv, perm, cia, cja, cup)
    Ts = eltype(z)
    @inbounds begin
        i = off + k
        n = perm[i]
        t = y[n]
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

# The red–black kernels as stage 1 of the reduction of r·z (each returns r_n·z_n for the
# rows it finishes: the second colour in the forward kernel, the first in the backward one).
@inline function _mc_forward_dot(::Type{Ta}, k, off, z, src, r, vals, dinv, perm, cia, cja, cup) where {Ta}
    _mc_forward(k, off, z, src, r, vals, dinv, perm, cia, cja, cup, true)
    n = @inbounds perm[off + k]
    return (@inbounds(Ta(r[n]) * Ta(z[n])), zero(Ta), zero(Ta), zero(Ta))
end

@inline function _mc_backward_ilu_dot(::Type{Ta}, k, off, z, y, vals, dinv, perm, cia, cja, cup) where {Ta}
    _mc_backward_ilu(k, off, z, y, vals, dinv, perm, cia, cja, cup)
    n = @inbounds perm[off + k]
    return (@inbounds(Ta(y[n]) * Ta(z[n])), zero(Ta), zero(Ta), zero(Ta))
end

# Red–black ILU0: r·z fused into the two application kernels (T4.5; the partial sums are
# grouped by colour, so the sum is rounded differently from a separate dot product).
function _precond_apply_dot!(z, P::MulticolorPreconditioner, r, red, post, sa, ss)
    (P.ilu && ncolors(P) == 2) || return (precond_apply!(z, P, r); false)
    backend = get_backend(z)
    len(c) = P.cptr[c + 1] - P.cptr[c]
    off(c) = Int32(P.cptr[c] - 1)
    g = _reduce_stage1!(red, backend, 0, _mc_forward_dot, _SUM4, len(2), off(2), z, r, r, P.vals, P.dinv, P.perm, P.cia, P.cja, P.cup)
    g += _reduce_stage1!(red, backend, g, _mc_backward_ilu_dot, _SUM4, len(1), off(1), z, r, P.vals, P.dinv, P.perm, P.cia, P.cja, P.cup)
    _finalize_async!(red, backend, _SUM4, g, post, sa, ss)
    return true
end

function precond_apply!(z, P::MulticolorPreconditioner, r)
    backend = get_backend(z)
    nc = ncolors(P)
    len(c) = P.cptr[c + 1] - P.cptr[c]
    off(c) = Int32(P.cptr[c] - 1)
    if P.ilu && nc == 2
        # Red–black: two kernels (y_1 = r; the second colour's backward step is fused).
        _map!(backend, _mc_forward, len(2), off(2), z, r, r, P.vals, P.dinv, P.perm, P.cia, P.cja, P.cup, true)
        _map!(backend, _mc_backward_ilu, len(1), off(1), z, r, P.vals, P.dinv, P.perm, P.cia, P.cja, P.cup)
    elseif P.ilu
        for c in 1:nc
            _map!(backend, _mc_forward, len(c), off(c), z, z, r, P.vals, P.dinv, P.perm, P.cia, P.cja, P.cup, c == nc)
        end
        for c in (nc - 1):-1:1
            _map!(backend, _mc_backward_ilu, len(c), off(c), z, z, P.vals, P.dinv, P.perm, P.cia, P.cja, P.cup)
        end
    else
        for c in 1:nc
            _map!(backend, _mc_forward, len(c), off(c), z, z, r, P.vals, P.dinv, P.perm, P.cia, P.cja, P.cup, true)
        end
        for c in (nc - 1):-1:1
            _map!(backend, _mc_backward_sgs, len(c), off(c), z, r, P.vals, P.dinv, P.perm, P.cia, P.cja)
        end
    end
    return z
end
