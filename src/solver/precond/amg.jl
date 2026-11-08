# Smoothed-aggregation algebraic multigrid (T4.3; PLAN.md section 3.5), our own
# implementation (Vaněk, Mandel and Brezina 1996, as in PyAMG's `smoothed_aggregation_solver`
# and ML), used as the preconditioner of CG and BiCGSTAB (`preconditioner = :amg`). MF6 has
# no AMG (IMS uses ILU0); the solution is the same, only the inner iteration counts differ.
#
# Operator: the scaled (for CG sign-adjusted) matrix `Â = diag(rowfac)·A·diag(colfac)` that
# the Krylov solvers iterate on (src/solver/krylov.jl). Its near-null space is
# `B = 1 ./ colfac` (A's is the constant vector).
#
# Setup on the host, in Float64 (`_build_hierarchy`), from a copy of `Â`:
# 1. strength: `j` is a strong neighbour of `i` when `|â_ij| ≥ θ·max_k |â_ik|` (rows
#    without off-diagonal entries, such as constant heads, are isolated and never
#    aggregated). With the strong vertical coupling of 2-layer models only the vertical
#    connections are strong at the finest level, so the first level aggregates each
#    vertical column, and the next levels coarsen the 2-D problem.
# 2. aggregation: the three passes of standard aggregation (a node and its free strong
#    neighbours; leftovers join a neighbouring aggregate; the rest form new aggregates).
# 3. tentative prolongator from `B` (one column per aggregate, normalised), and the
#    coarse `B`.
# 4. prolongator smoothing `P = (I − ω·D_F⁻¹·A_F)·P_t` with the filtered matrix `A_F` (weak
#    connections lumped into the diagonal so that `A_F·B = A·B`), `ω = 4/(3·ρ(D_F⁻¹A_F))`
#    (Gershgorin bound of ρ).
# 5. Galerkin coarse operator `A_c = R·A·P` with `R = Pᵀ` (threaded Gustavson products).
# 6. repeat until `max_coarse` nodes; the coarsest level is solved with a dense inverse.
#    The Gershgorin bound of `ρ(D⁻¹A)` of each level gives the smoother interval.
#
# Device (KA kernels, data in `Tp`): the V-cycle with a Chebyshev (default) or damped
# Jacobi smoother on `D⁻¹A`, the same polynomial before and after the coarse correction so
# that the V-cycle is symmetric (CG stays valid), CSR restriction and prolongation, and the
# dense coarsest solve. The finest level uses the current `Â` itself (the Krylov operator,
# `_op_row`) and the solver's Jacobi diagonal.
#
# Reuse: the aggregates (and the strong mask, B and the tentative prolongators) are kept
# across outer iterations and time steps. When the matrix changed since the last update
# (row signatures, `update_tol`), `precond_setup!` recomputes on the device the smoothed
# prolongators (`_update_prolongator!`), the Galerkin products `R·A·P`
# (`galerkin_kernel!`, accumulated in Float64 where the backend has it, rule 4), the
# Gershgorin bounds, and on the host the dense coarsest inverse. With the default
# `strength = :offdiag` the aggregates depend only on the off-diagonals, which are
# constant for confined models. The hierarchy is rebuilt on the host when a solve takes
# more than `rebuild_factor` times the inner iterations of the first solve after the last
# build (at least `rebuild_min` more), and before a failed solve is repeated
# (`_precond_retry!`).

"""
    AMGOptions(; theta = 0.08, strength = :offdiag, smoother = :chebyshev, degree = 3,
               fine_degree = 1, cheb_ratio = 0.3,
               max_coarse = 400, max_levels = 20, smooth = true, rebuild_factor = 2.0,
               rebuild_min = 10, reuse = true, update_tol = 0.0)

Options of `AMGPreconditioner` (`precond_options` of `KrylovSettings`):

- `theta`: strength threshold (`|a_ij| ≥ θ·√|a_ii·a_jj|`);
- `smoother`: `:chebyshev` (polynomial of degree `degree` on `[cheb_ratio·λ, λ]`, λ the
  estimated largest eigenvalue of `D⁻¹A`) or `:jacobi` (`degree` sweeps of damped Jacobi,
  `ω = 4/(3λ)`), before and after the coarse correction; `fine_degree` on the finest level
  (0: `degree`; 1, the default, fuses the pre-smoothing into the residual and the
  post-smoothing with the dot product of CG);
- `strength`: `:offdiag` (`|a_ij| ≥ θ·max_k |a_ik|`, independent of the diagonal, so the
  aggregates stay valid when boundary and storage terms change) or `:symmetric`
  (`|a_ij| ≥ θ·√|a_ii·a_jj|`);
- `max_coarse`: coarsest size (dense inverse), `max_levels`: most levels;
- `smooth`: smooth the tentative prolongator (smoothed aggregation) or not (plain
  aggregation);
- `rebuild_factor`, `rebuild_min`: rebuild the hierarchy when a solve takes more than
  `max(rebuild_factor·n₀, n₀ + rebuild_min)` iterations, with `n₀` those of the first
  solve after the last build;
- `reuse`: keep the aggregates between setups (`false`: rebuild the hierarchy at every
  `precond_setup!`, for diagnostics);
- `update_tol`: recompute the prolongators and coarse operators only when the matrix
  changed by more than this since the last update (the larger relative 1-norm change of
  the row sums `Σ_j a_ij`, which are the storage and boundary terms, and of `Σ_j |a_ij|`);
  otherwise the coarse levels are kept (the finest level always uses the
  current matrix). 0 updates at every setup.

The largest eigenvalue of `D⁻¹A` on each level is its Gershgorin bound
`max_i Σ_j |a_ij/a_ii|`, recomputed at every setup.
"""
Base.@kwdef struct AMGOptions
    theta::Float64 = 0.08
    strength::Symbol = :offdiag
    smoother::Symbol = :chebyshev
    degree::Int = 3
    fine_degree::Int = 1
    cheb_ratio::Float64 = 0.3
    max_coarse::Int = 400
    max_levels::Int = 20
    smooth::Bool = true
    rebuild_factor::Float64 = 2.0
    rebuild_min::Int = 10
    reuse::Bool = true
    update_tol::Float64 = 0.0
    function AMGOptions(theta, strength, smoother, degree, fine_degree, cheb_ratio, max_coarse, max_levels, smooth, rebuild_factor, rebuild_min, reuse, update_tol)
        theta >= 0 || throw(ArgumentError("AMG theta must be >= 0"))
        strength in (:offdiag, :symmetric) || throw(ArgumentError("unknown AMG strength measure $strength (expected :offdiag or :symmetric)"))
        smoother in (:chebyshev, :jacobi) || throw(ArgumentError("unknown AMG smoother $smoother (expected :chebyshev or :jacobi)"))
        degree >= 1 || throw(ArgumentError("AMG smoother degree must be >= 1"))
        fine_degree >= 0 || throw(ArgumentError("AMG fine_degree must be >= 0"))
        0 < cheb_ratio < 1 || throw(ArgumentError("AMG cheb_ratio must be in (0, 1)"))
        max_coarse >= 1 && max_levels >= 1 || throw(ArgumentError("AMG max_coarse and max_levels must be >= 1"))
        return new(theta, strength, smoother, degree, fine_degree, cheb_ratio, max_coarse, max_levels, smooth, rebuild_factor, rebuild_min, reuse, update_tol)
    end
end

# ---------------------------------------------------------------------------------------
# Host CSR matrices and products (Float64)

struct _HostCsr
    nrow::Int
    ncol::Int
    ia::Vector{Int}
    ja::Vector{Int32}
    a::Vector{Float64}
end

_nnz(A::_HostCsr) = length(A.ja)

# Row ranges of `1:n` for the threads.
_chunks(n) = (k = max(1, min(Threads.nthreads(), cld(n, 4096))); [(1 + div((c - 1) * n, k)):div(c * n, k) for c in 1:k])

# Concatenate per-chunk rows (lengths, columns, values) into a CSR matrix.
function _concat_rows(nrow, ncol, parts)
    ia = Vector{Int}(undef, nrow + 1)
    ia[1] = 1
    i = 1
    for (len, _, _) in parts, l in len
        ia[i + 1] = ia[i] + l
        i += 1
    end
    nnz = ia[end] - 1
    ja = Vector{Int32}(undef, nnz)
    a = Vector{Float64}(undef, nnz)
    pos = 1
    for (_, cols, vals) in parts
        copyto!(ja, pos, cols, 1, length(cols))
        copyto!(a, pos, vals, 1, length(vals))
        pos += length(cols)
    end
    return _HostCsr(nrow, ncol, ia, ja, a)
end

# Insertion sort of cols[lo:hi] with vals alongside (rows are short).
function _sort_row!(cols, vals, lo, hi)
    @inbounds for k in (lo + 1):hi
        c = cols[k]
        v = vals[k]
        j = k - 1
        while j >= lo && cols[j] > c
            cols[j + 1] = cols[j]
            vals[j + 1] = vals[j]
            j -= 1
        end
        cols[j + 1] = c
        vals[j + 1] = v
    end
    return nothing
end

# C = A·B (Gustavson, rows in parallel), columns sorted per row. Every structural product
# is kept (also when its value is zero), so the pattern does not depend on the values.
function _spgemm(A::_HostCsr, B::_HostCsr)
    A.ncol == B.nrow || throw(DimensionMismatch("AMG product"))
    chunks = _chunks(A.nrow)
    parts = Vector{Tuple{Vector{Int32}, Vector{Int32}, Vector{Float64}}}(undef, length(chunks))
    Threads.@threads for c in eachindex(chunks)
        marker = zeros(Int32, B.ncol)
        len = Int32[]
        cols = Int32[]
        vals = Float64[]
        sizehint!(len, length(chunks[c]))
        # At most one entry per product.
        est = 0
        @inbounds for i in chunks[c], ii in A.ia[i]:(A.ia[i + 1] - 1)
            est += B.ia[A.ja[ii] + 1] - B.ia[A.ja[ii]]
        end
        sizehint!(cols, est)
        sizehint!(vals, est)
        @inbounds for i in chunks[c]
            start = length(cols)
            for ii in A.ia[i]:(A.ia[i + 1] - 1)
                j = A.ja[ii]
                av = A.a[ii]
                for jj in B.ia[j]:(B.ia[j + 1] - 1)
                    k = B.ja[jj]
                    pos = marker[k]
                    if pos <= start
                        push!(cols, k)
                        push!(vals, av * B.a[jj])
                        marker[k] = length(cols) % Int32
                    else
                        vals[pos] += av * B.a[jj]
                    end
                end
            end
            _sort_row!(cols, vals, start + 1, length(cols))
            push!(len, length(cols) - start)
        end
        parts[c] = (len, cols, vals)
    end
    return _concat_rows(A.nrow, B.ncol, parts)
end

# Aᵀ, and for each entry of Aᵀ the index of the entry of A it comes from.
function _transpose(A::_HostCsr)
    cnt = zeros(Int, A.ncol + 1)
    @inbounds for k in A.ja
        cnt[k + 1] += 1
    end
    ia = cumsum!(cnt, cnt) .+ 1
    ia[1] = 1
    next = ia[1:(end - 1)]
    ja = Vector{Int32}(undef, _nnz(A))
    a = Vector{Float64}(undef, _nnz(A))
    perm = Vector{Int32}(undef, _nnz(A))
    @inbounds for i in 1:A.nrow, ii in A.ia[i]:(A.ia[i + 1] - 1)
        k = A.ja[ii]
        p = next[k]
        ja[p] = i
        a[p] = A.a[ii]
        perm[p] = ii
        next[k] = p + 1
    end
    return _HostCsr(A.ncol, A.nrow, ia, ja, a), perm
end

# Position of the diagonal in each row (0 when absent) and its value.
function _diagonal(A::_HostCsr)
    idx = zeros(Int, A.nrow)
    Threads.@threads for i in 1:A.nrow
        @inbounds for ii in A.ia[i]:(A.ia[i + 1] - 1)
            if A.ja[ii] == i
                idx[i] = ii
                break
            end
        end
    end
    d = [k > 0 ? A.a[k] : 0.0 for k in idx]
    return idx, d
end

# Gershgorin bound of the largest eigenvalue of D⁻¹A: max_i Σ_j |a_ij / a_ii|.
function _lambda_max(A::_HostCsr, dinv)
    λ = 0.0
    @inbounds for i in 1:A.nrow
        s = 0.0
        for ii in A.ia[i]:(A.ia[i + 1] - 1)
            s += abs(A.a[ii])
        end
        λ = max(λ, s * abs(dinv[i]))
    end
    return λ
end

# ---------------------------------------------------------------------------------------
# Host setup: strength, aggregation, prolongators, Galerkin products

# Strong connections (a mask over the entries of A; never the diagonal): `|a_ij| ≥
# θ·max_k |a_ik|` (`:offdiag`) or `|a_ij| ≥ θ·√|a_ii·a_jj|` (`:symmetric`).
function _strength(A::_HostCsr, d, θ, measure::Symbol)
    n = A.nrow
    isolated = fill(true, n)
    Threads.@threads for i in 1:n
        @inbounds for ii in A.ia[i]:(A.ia[i + 1] - 1)
            if A.ja[ii] != i && A.a[ii] != 0
                isolated[i] = false
                break
            end
        end
    end
    strong = fill(false, _nnz(A))
    offdiag = measure === :offdiag
    Threads.@threads for i in 1:n
        isolated[i] && continue
        amax = 0.0
        if offdiag
            @inbounds for ii in A.ia[i]:(A.ia[i + 1] - 1)
                (A.ja[ii] == i || isolated[A.ja[ii]]) && continue
                amax = max(amax, abs(A.a[ii]))
            end
        end
        @inbounds for ii in A.ia[i]:(A.ia[i + 1] - 1)
            j = A.ja[ii]
            (j == i || isolated[j]) && continue
            v = abs(A.a[ii])
            strong[ii] = v > 0 && v >= (offdiag ? θ * amax : θ * sqrt(abs(d[i] * d[j])))
        end
    end
    return strong
end

# Standard aggregation (three passes); returns the aggregate of each node (0: none) and
# the number of aggregates.
function _aggregate(A::_HostCsr, strong)
    n = A.nrow
    agg = zeros(Int32, n)
    nc = Int32(0)
    # pass 1: a node whose strong neighbours are all free forms an aggregate with them
    @inbounds for i in 1:n
        agg[i] == 0 || continue
        has = false
        free = true
        for ii in A.ia[i]:(A.ia[i + 1] - 1)
            strong[ii] || continue
            has = true
            if agg[A.ja[ii]] != 0
                free = false
                break
            end
        end
        if !has
            agg[i] = -1  # no strong neighbours: not aggregated
        elseif free
            nc += Int32(1)
            agg[i] = nc
            for ii in A.ia[i]:(A.ia[i + 1] - 1)
                strong[ii] && (agg[A.ja[ii]] = nc)
            end
        end
    end
    # pass 2: join the aggregate of a strong neighbour from pass 1
    p1 = copy(agg)
    @inbounds for i in 1:n
        agg[i] == 0 || continue
        for ii in A.ia[i]:(A.ia[i + 1] - 1)
            if strong[ii] && p1[A.ja[ii]] > 0
                agg[i] = p1[A.ja[ii]]
                break
            end
        end
    end
    # pass 3: the rest form aggregates with their unaggregated strong neighbours
    @inbounds for i in 1:n
        agg[i] == 0 || continue
        nc += Int32(1)
        agg[i] = nc
        for ii in A.ia[i]:(A.ia[i + 1] - 1)
            strong[ii] && agg[A.ja[ii]] == 0 && (agg[A.ja[ii]] = nc)
        end
    end
    @inbounds for i in 1:n
        agg[i] < 0 && (agg[i] = 0)
    end
    return agg, Int(nc)
end

# Tentative prolongator (n × nc, one entry per aggregated node) and the coarse B.
function _tentative(agg, nc, B)
    n = length(agg)
    nrm = zeros(nc)
    @inbounds for i in 1:n
        agg[i] > 0 && (nrm[agg[i]] += B[i]^2)
    end
    nrm .= sqrt.(nrm)
    ia = Vector{Int}(undef, n + 1)
    ia[1] = 1
    @inbounds for i in 1:n
        ia[i + 1] = ia[i] + (agg[i] > 0)
    end
    ja = Vector{Int32}(undef, ia[end] - 1)
    a = Vector{Float64}(undef, ia[end] - 1)
    Threads.@threads for i in 1:n
        @inbounds if agg[i] > 0
            k = ia[i]
            ja[k] = agg[i]
            a[k] = nrm[agg[i]] > 0 ? B[i] / nrm[agg[i]] : 0.0
        end
    end
    return _HostCsr(n, nc, ia, ja, a), nrm
end

# S = I − ω·D_F⁻¹·A_F with the filtered matrix A_F: strong entries, and the weak ones
# lumped into the diagonal so that A_F·B = A·B; ω = 4/(3·ρ) with the Gershgorin bound ρ.
# The device recomputes P = S·P_t in the same way (`_prolongator_row`).
function _smoother_matrix(A::_HostCsr, d, strong, B)
    n = A.nrow
    dF = copy(d)
    cnt = zeros(Int, n)
    rho = zeros(n)
    Threads.@threads for i in 1:n
        @inbounds begin
            w = 0.0
            c = 0
            for ii in A.ia[i]:(A.ia[i + 1] - 1)
                j = A.ja[ii]
                j == i && continue
                if strong[ii]
                    c += 1
                elseif B[i] != 0
                    w += A.a[ii] * B[j] / B[i]
                end
            end
            f = d[i] + w
            # Keep the original diagonal where lumping would flip or nearly cancel it.
            (sign(f) != sign(d[i]) || abs(f) < 1.0e-3 * abs(d[i])) && (f = d[i])
            dF[i] = f
            cnt[i] = c
            if f != 0
                g = abs(f)
                for ii in A.ia[i]:(A.ia[i + 1] - 1)
                    strong[ii] && (g += abs(A.a[ii]))
                end
                rho[i] = g / abs(f)
            end
        end
    end
    ρ = maximum(rho; init = 0.0)
    ω = ρ > 0 ? 4 / (3ρ) : 0.0
    ia = Vector{Int}(undef, n + 1)
    ia[1] = 1
    @inbounds for i in 1:n
        ia[i + 1] = ia[i] + 1 + cnt[i]
    end
    ja = Vector{Int32}(undef, ia[end] - 1)
    a = Vector{Float64}(undef, ia[end] - 1)
    Threads.@threads for i in 1:n
        @inbounds begin
            f = dF[i] != 0 ? ω / dF[i] : 0.0
            k = ia[i]
            ja[k] = i
            a[k] = dF[i] != 0 ? 1 - ω : 1.0
            for ii in A.ia[i]:(A.ia[i + 1] - 1)
                if strong[ii]
                    k += 1
                    ja[k] = A.ja[ii]
                    a[k] = -f * A.a[ii]
                end
            end
        end
    end
    return _HostCsr(n, n, ia, ja, a)
end

# One level of the host hierarchy: the operator A, its D⁻¹ and λmax, and (except on the
# coarsest level) P to the next level and R = Pᵀ.
# (and the data to recompute P on the device: the strong mask over A's entries, B, the
# aggregate and tentative value of each node, and R's entries as indices into P's).
struct _HostLevel
    A::_HostCsr
    idiag::Vector{Int}
    dinv::Vector{Float64}
    lmax::Float64
    P::Union{Nothing, _HostCsr}
    R::Union{Nothing, _HostCsr}
    strong::Vector{Bool}
    B::Vector{Float64}
    agg::Vector{Int32}
    pt::Vector{Float64}
    rperm::Vector{Int32}
end

_inv_or_one(d) = d != 0 ? 1 / d : 1.0

function _build_hierarchy(A::_HostCsr, B::Vector{Float64}, opt::AMGOptions)
    levels = _HostLevel[]
    while true
        idiag, d = _diagonal(A)
        dinv = _inv_or_one.(d)
        lmax = _lambda_max(A, dinv)
        coarsest = A.nrow <= opt.max_coarse || length(levels) + 1 >= opt.max_levels
        if !coarsest
            strong = _strength(A, d, opt.theta, opt.strength)
            agg, nc = _aggregate(A, strong)
            if nc == 0 || nc >= A.nrow
                coarsest = true
            else
                Pt, Bc = _tentative(agg, nc, B)
                P = opt.smooth ? _spgemm(_smoother_matrix(A, d, strong, B), Pt) : Pt
                R, rperm = _transpose(P)
                Ac = _spgemm(R, _spgemm(A, P))
                pt = zeros(A.nrow)
                for i in 1:A.nrow
                    Pt.ia[i + 1] > Pt.ia[i] && (pt[i] = Pt.a[Pt.ia[i]])
                end
                push!(levels, _HostLevel(A, idiag, dinv, lmax, P, R, strong, B, agg, pt, rperm))
                A, B = Ac, Bc
                continue
            end
        end
        push!(levels, _HostLevel(A, idiag, dinv, lmax, nothing, nothing, Bool[], B, Int32[], Float64[], Int32[]))
        return levels
    end
    return
end

# Dense inverse of a (small) CSR matrix; the pseudo-inverse when it is singular.
function _dense_inverse(ia, ja, a, n)
    M = zeros(n, n)
    @inbounds for i in 1:n, ii in ia[i]:(ia[i + 1] - 1)
        M[i, ja[ii]] += a[ii]
    end
    F = LinearAlgebra.lu(M; check = false)
    return LinearAlgebra.issuccess(F) && all(isfinite, F.U) ? inv(F) : LinearAlgebra.pinv(M)
end

# ---------------------------------------------------------------------------------------
# Device hierarchy

# Operators in the kernels: the finest level is the Krylov operator itself (`_op_row`),
# coarser levels are CSR matrices.
struct _FineOp{A, VI, VS}
    amat::A
    ia::VI
    ja::VI
    rowfac::VS
    colfac::VS
end
Adapt.@adapt_structure _FineOp

struct _CsrOp{VI, VP}
    ia::VI
    ja::VI
    a::VP
end
Adapt.@adapt_structure _CsrOp

@inline _row(op::_FineOp, n, x) = _op_row(n, op.amat, op.ia, op.ja, op.rowfac, op.colfac, x)
@inline function _row(op::_CsrOp, n, x)
    T = eltype(x)
    s = zero(T)
    @inbounds for ii in op.ia[n]:(op.ia[n + 1] - Int32(1))
        s += T(op.a[ii]) * x[op.ja[ii]]
    end
    return s
end

@inline _val(op::_FineOp, ::Type{Ta}, ii, i, j) where {Ta} = @inbounds Ta(op.rowfac[i]) * Ta(op.amat[ii]) * Ta(op.colfac[j])
@inline _val(op::_CsrOp, ::Type{Ta}, ii, i, j) where {Ta} = @inbounds Ta(op.a[ii])

"""
    AMGLevel

One level of the device hierarchy of `AMGPreconditioner`: size `n`, the operator in CSR (`ia`, `ja`, `a`, diagonal positions
`idiag`, `dinv`; empty on the finest level, which uses the Krylov operator), the
prolongator `P` (`pia`, `pja`, `pa`, from the next level) and restriction `R = Pᵀ`
(`ria`, `rja`, `ra`; empty on the coarsest level), and work vectors `x`, `y`, `f`, `w`,
`d` (empty on the finest level). For the device recomputation of `P` (smoothed
aggregation): `rperm` (entries of `R` as indices into `P`), `smask` (strong entries of
the operator), `B` (near-null space), `agg` and `pt` (aggregate and tentative value per
row).
"""
struct AMGLevel{VI <: AbstractVector{Int32}, VP <: AbstractVector, VB <: AbstractVector{UInt8}}
    n::Int
    ia::VI
    ja::VI
    a::VP
    idiag::VI
    dinv::VP
    pia::VI
    pja::VI
    pa::VP
    ria::VI
    rja::VI
    ra::VP
    rperm::VI
    smask::VB
    B::VP
    agg::VI
    pt::VP
    x::VP
    y::VP
    f::VP
    w::VP
    d::VP
end

"""
    AMGStats

Counters of an `AMGPreconditioner`: `builds` (host hierarchy builds) and `build_time`
(s), `updates` (device Galerkin recomputations) and `update_time` (s, synchronised),
`solves` (preconditioned solves), `skipped` (setups that kept the coarse levels), `sizes` (rows per level), `nnz` (entries
of each level's operator) and `device_bytes` (hierarchy and work vectors on the device).
"""
mutable struct AMGStats
    builds::Int
    build_time::Float64
    updates::Int
    update_time::Float64
    solves::Int
    skipped::Int
    sizes::Vector{Int}
    nnz::Vector{Int}
    device_bytes::Int
end

AMGStats() = AMGStats(0, 0.0, 0, 0.0, 0, 0, Int[], Int[], 0)

"""
    AMGPreconditioner{Ts, Tp, Ta}

Smoothed-aggregation AMG preconditioner (see the header of src/solver/precond/amg.jl):
host setup in Float64, device hierarchy and V-cycle in `Tp` (finest level in `Ts`),
Galerkin products recomputed on the device in `Ta` once per `precond_setup!`. The
hierarchy is built at the first `precond_setup!`. `stats::AMGStats` counts builds,
updates and their times.

    AMGPreconditioner(backend, n, Ts, Tp; options...)

`options` are the keywords of `AMGOptions`.
"""
mutable struct AMGPreconditioner{Ts, Tp, Ta, B <: Backend, VS <: AbstractVector{Ts}, VP <: AbstractVector{Tp}, VA <: AbstractVector{Ta}, VI <: AbstractVector{Int32}, VB <: AbstractVector{UInt8}, R <: Reducer} <: AbstractPreconditioner
    const options::AMGOptions
    const backend::B
    const n::Int
    const reducer::R
    levels::Vector{AMGLevel{VI, VP, VB}}
    lmax::Vector{Float64}
    fy::VS
    fw::VS
    fd::VS
    sig::VS
    sig2::VS
    gacc::VA
    cinv::VP
    # host: coarsest pattern (dense inverse updates) and the finest CSR structure
    host_coarse::_HostCsr
    host_ia::Vector{Int}
    host_ja::Vector{Int32}
    fine::Any
    coefs::Vector{Vector{Tuple{Float64, Float64}}}
    coef_lmax::Vector{Float64}
    built::Bool
    rebuild::Bool
    fresh::Bool
    skipped::Bool
    force_update::Bool
    base_iters::Int
    const stats::AMGStats
end

function AMGPreconditioner(backend::Backend, n::Integer, ::Type{Ts}, ::Type{Tp}; options...) where {Ts, Tp}
    opt = AMGOptions(; options...)
    Ta = supports_float64(backend) ? Float64 : Tp
    zs = KernelAbstractions.zeros(backend, Ts, 0)
    zp = KernelAbstractions.zeros(backend, Tp, 0)
    za = KernelAbstractions.zeros(backend, Ta, 0)
    zi = KernelAbstractions.zeros(backend, Int32, 0)
    empty = _HostCsr(0, 0, [1], Int32[], Float64[])
    zb = KernelAbstractions.zeros(backend, UInt8, 0)
    red = Reducer(backend, n, Ta)
    return AMGPreconditioner{Ts, Tp, Ta, typeof(backend), typeof(zs), typeof(zp), typeof(za), typeof(zi), typeof(zb), typeof(red)}(
        opt, backend, Int(n), red, AMGLevel{typeof(zi), typeof(zp), typeof(zb)}[], Float64[], zs, similar(zs), similar(zs), similar(zs), similar(zs), za, zp, empty,
        Int[], Int32[], nothing, Vector{Tuple{Float64, Float64}}[], Float64[], false, true, false, false, false, 0, AMGStats(),
    )
end

# Adapt: an unbuilt preconditioner with the same options on the target backend.
function Adapt.adapt_structure(to, P::AMGPreconditioner{Ts, Tp}) where {Ts, Tp}
    backend = get_backend(adapt(to, P.fy))
    opt = P.options
    return AMGPreconditioner(backend, P.n, Ts, Tp; (f => getfield(opt, f) for f in fieldnames(AMGOptions))...)
end

_device(backend, ::Type{T}, x) where {T} = _device_copy(backend, convert(Vector{T}, x))

function _upload_level(backend, ::Type{Tp}, h::_HostLevel, first::Bool, smooth::Bool) where {Tp}
    zi() = KernelAbstractions.zeros(backend, Int32, 0)
    zp(len = 0) = KernelAbstractions.zeros(backend, Tp, len)
    A = h.A
    n = A.nrow
    if first
        ia, ja, a, idiag, dinv = zi(), zi(), zp(), zi(), zp()
        x, y, f, w, d = zp(), zp(), zp(), zp(), zp()
    else
        ia, ja, a = _device(backend, Int32, A.ia), _device(backend, Int32, A.ja), _device(backend, Tp, A.a)
        idiag, dinv = _device(backend, Int32, h.idiag), _device(backend, Tp, h.dinv)
        x, y, f, w, d = zp(n), zp(n), zp(n), zp(n), zp(n)
    end
    if isnothing(h.P)
        pia, pja, pa, ria, rja, ra = zi(), zi(), zp(), zi(), zi(), zp()
        rperm, smask, B, agg, pt = zi(), KernelAbstractions.zeros(backend, UInt8, 0), zp(), zi(), zp()
    else
        pia, pja, pa = _device(backend, Int32, h.P.ia), _device(backend, Int32, h.P.ja), _device(backend, Tp, h.P.a)
        ria, rja, ra = _device(backend, Int32, h.R.ia), _device(backend, Int32, h.R.ja), _device(backend, Tp, h.R.a)
        if smooth
            rperm, smask = _device(backend, Int32, h.rperm), _device(backend, UInt8, h.strong)
            B, agg, pt = _device(backend, Tp, h.B), _device(backend, Int32, h.agg), _device(backend, Tp, h.pt)
        else
            rperm, smask, B, agg, pt = zi(), KernelAbstractions.zeros(backend, UInt8, 0), zp(), zi(), zp()
        end
    end
    return AMGLevel(n, ia, ja, a, idiag, dinv, pia, pja, pa, ria, rja, ra, rperm, smask, B, agg, pt, x, y, f, w, d)
end

_bytes(x::AbstractArray) = length(x) * sizeof(eltype(x))
_bytes(l::AMGLevel) = sum(_bytes(getfield(l, f)) for f in fieldnames(AMGLevel) if getfield(l, f) isa AbstractArray)

# Host build from the current scaled operator (copied from the device).
function _build!(P::AMGPreconditioner{Ts, Tp, Ta}, solver, amat) where {Ts, Tp, Ta}
    t0 = time()
    backend = P.backend
    csr = solver.csr
    n = P.n
    if isempty(P.host_ia)
        P.host_ia = Int.(Array(csr.ia))
        P.host_ja = Array(csr.ja)
    end
    ia, ja = P.host_ia, P.host_ja
    a = Float64.(Array(amat))
    rf = Float64.(Array(solver.rowfac))
    cf = Float64.(Array(solver.colfac))
    Threads.@threads for i in 1:n
        @inbounds for ii in ia[i]:(ia[i + 1] - 1)
            a[ii] *= rf[i] * cf[ja[ii]]
        end
    end
    B = [c != 0 ? 1 / c : 1.0 for c in cf]
    hl = _build_hierarchy(_HostCsr(n, n, ia, ja, a), B, P.options)
    P.levels = [_upload_level(backend, Tp, hl[l], l == 1, P.options.smooth) for l in eachindex(hl)]
    P.lmax = [h.lmax for h in hl]
    P.fy = KernelAbstractions.zeros(backend, Ts, n)
    P.fw = KernelAbstractions.zeros(backend, Ts, n)
    P.fd = KernelAbstractions.zeros(backend, Ts, n)
    P.sig = KernelAbstractions.zeros(backend, Ts, 2n)
    P.sig2 = KernelAbstractions.zeros(backend, Ts, 2n)
    _matrix_change!(P, amat, csr)
    maxnnz = maximum((_nnz(h.A) for h in hl[2:end]); init = 0)
    P.gacc = Ta === Tp ? KernelAbstractions.zeros(backend, Ta, 0) : KernelAbstractions.zeros(backend, Ta, maxnnz)
    hc = last(hl).A
    P.host_coarse = hc
    nL = hc.nrow
    P.cinv = _device(backend, Tp, vec(_dense_inverse(hc.ia, hc.ja, hc.a, nL)))
    st = P.stats
    st.builds += 1
    st.sizes = [h.A.nrow for h in hl]
    st.nnz = [_nnz(h.A) for h in hl]
    st.device_bytes = sum(_bytes, P.levels) + sum(_bytes, (P.fy, P.fw, P.fd, P.sig, P.sig2, P.gacc, P.cinv))
    P.built = true
    P.rebuild = false
    P.fresh = true
    KernelAbstractions.synchronize(backend)
    st.build_time += time() - t0
    return P
end

# --- device Galerkin product A_c = R·A·P on the fixed pattern of A_c

# Position of column J in cja[lo:hi] (sorted); the pattern always contains it.
@inline function _bsearch(cja, lo, hi, J)
    @inbounds while lo < hi
        mid = (lo + hi) >>> 1
        if cja[mid] < J
            lo = mid + Int32(1)
        else
            hi = mid
        end
    end
    return lo
end

@kernel function galerkin_kernel!(acc, @Const(cia), @Const(cja), @Const(ria), @Const(rja), @Const(ra), op, @Const(pia), @Const(pja), @Const(pa))
    I = @index(Global, Linear)
    Ta = eltype(acc)
    @inbounds if I < length(cia)
        k0 = cia[I]
        k1 = cia[I + 1] - Int32(1)
        for k in k0:k1
            acc[k] = zero(Ta)
        end
        for t in ria[I]:(ria[I + 1] - Int32(1))
            i = rja[t]
            rv = Ta(ra[t])
            for ii in op.ia[i]:(op.ia[i + 1] - Int32(1))
                j = op.ja[ii]
                v = rv * _val(op, Ta, ii, i, j)
                for s in pia[j]:(pia[j + 1] - Int32(1))
                    k = _bsearch(cja, k0, k1, pja[s])
                    acc[k] += v * Ta(pa[s])
                end
            end
        end
    end
end

# The Galerkin result (in Ta) into the level's operator (Tp); exempt from the kernel
# hygiene test with Float64 accumulation, like galerkin_kernel!.
@kernel function galerkin_store_kernel!(a, @Const(acc))
    k = @index(Global, Linear)
    @inbounds if k <= length(a)
        a[k] = convert(eltype(a), acc[k])
    end
end

@inline function _coarse_dinv(i, dinv, a, idiag)
    T = eltype(dinv)
    @inbounds begin
        k = idiag[i]
        d = k > 0 ? a[k] : zero(T)
        dinv[i] = d != zero(T) ? one(T) / d : one(T)
    end
    return nothing
end

# Row i of the Gershgorin bound of D⁻¹A.
@inline function _gershgorin_term(::Type{Ta}, i, op, dinv) where {Ta}
    s = zero(Ta)
    @inbounds for ii in op.ia[i]:(op.ia[i + 1] - Int32(1))
        s += abs(_val(op, Ta, ii, i, op.ja[ii]))
    end
    return (s * abs(@inbounds Ta(dinv[i])), zero(Ta), zero(Ta), zero(Ta))
end

const _GERSH_OPS = (_MaxAbsOp(), _SumOp(), _SumOp(), _SumOp())

# Diagonal and filtered diagonal of row i (weak entries lumped with B; `_smoother_matrix`).
@inline function _filtered_diag(::Type{Ta}, i, op, smask, B) where {Ta}
    d = zero(Ta)
    w = zero(Ta)
    @inbounds begin
        bi = Ta(B[i])
        for ii in op.ia[i]:(op.ia[i + 1] - Int32(1))
            j = op.ja[ii]
            if j == i
                d += _val(op, Ta, ii, i, j)
            elseif smask[ii] == 0x00 && bi != zero(Ta)
                w += _val(op, Ta, ii, i, j) * Ta(B[j]) / bi
            end
        end
    end
    dF = d + w
    (sign(dF) != sign(d) || abs(dF) < abs(d) / Ta(1000)) && (dF = d)
    return dF
end

# Row i of the Gershgorin bound of D_F⁻¹·A_F (for ω).
@inline function _filtered_gershgorin_term(::Type{Ta}, i, op, smask, B) where {Ta}
    dF = _filtered_diag(Ta, i, op, smask, B)
    dF == zero(Ta) && return _zeros4(Ta)
    s = abs(dF)
    @inbounds for ii in op.ia[i]:(op.ia[i + 1] - Int32(1))
        smask[ii] != 0x00 && (s += abs(_val(op, Ta, ii, i, op.ja[ii])))
    end
    return (s / abs(dF), zero(Ta), zero(Ta), zero(Ta))
end

# Row i of P = (I − ω·D_F⁻¹·A_F)·P_t on P's fixed pattern, in the type of `zacc` (Tp).
@inline function _prolongator_row(i, pa, pia, pja, op, smask, B, agg, pt, ω, zacc)
    Ta = typeof(zacc)
    @inbounds begin
        dF = _filtered_diag(Ta, i, op, smask, B)
        f = dF != zero(Ta) ? ω / dF : zero(Ta)
        sii = dF != zero(Ta) ? one(Ta) - ω : one(Ta)
        for k in pia[i]:(pia[i + 1] - Int32(1))
            J = pja[k]
            v = agg[i] == J ? sii * Ta(pt[i]) : zero(Ta)
            for ii in op.ia[i]:(op.ia[i + 1] - Int32(1))
                j = op.ja[ii]
                if j != i && smask[ii] != 0x00 && agg[j] == J
                    v -= f * _val(op, Ta, ii, i, j) * Ta(pt[j])
                end
            end
            pa[k] = convert(eltype(pa), v)
        end
    end
    return nothing
end

@inline function _gather_entry(t, ra, pa, rperm)
    @inbounds ra[t] = pa[rperm[t]]
    return nothing
end

# Smoothed prolongator and restriction of level `lev` from its current operator `op`.
function _update_prolongator!(P::AMGPreconditioner{Ts, Tp, Ta}, lev::AMGLevel, op) where {Ts, Tp, Ta}
    backend = P.backend
    ρ = _mapreduce!(P.reducer, backend, _filtered_gershgorin_term, _GERSH_OPS, lev.n, op, lev.smask, lev.B)[1]
    ω = ρ > 0 ? 4 / (3ρ) : 0.0
    _map!(backend, _prolongator_row, lev.n, lev.pa, lev.pia, lev.pja, op, lev.smask, lev.B, lev.agg, lev.pt, Tp(ω), zero(Tp))
    _map!(backend, _gather_entry, length(lev.ra), lev.ra, lev.pa, lev.rperm)
    return nothing
end

# Recompute the coarse operators from the current matrix (device), their Gershgorin
# bounds, and the dense coarsest inverse (host).
function _update!(P::AMGPreconditioner{Ts, Tp, Ta}, fineop) where {Ts, Tp, Ta}
    t0 = time()
    backend = P.backend
    L = length(P.levels)
    op = fineop
    for l in 1:(L - 1)
        lev = P.levels[l]
        nxt = P.levels[l + 1]
        P.options.smooth && _update_prolongator!(P, lev, op)
        acc = Ta === Tp ? nxt.a : view(P.gacc, 1:length(nxt.a))
        galerkin_kernel!(backend, _WG)(acc, nxt.ia, nxt.ja, lev.ria, lev.rja, lev.ra, op, lev.pia, lev.pja, lev.pa; ndrange = nxt.n)
        Ta === Tp || galerkin_store_kernel!(backend, _WG)(nxt.a, acc; ndrange = length(nxt.a))
        _map!(backend, _coarse_dinv, nxt.n, nxt.dinv, nxt.a, nxt.idiag)
        P.lmax[l] = _mapreduce!(P.reducer, backend, _gershgorin_term, _GERSH_OPS, lev.n, op, l == 1 ? P.fine[2] : lev.dinv)[1]
        op = _CsrOp(nxt.ia, nxt.ja, nxt.a)
    end
    if L > 1
        hc = P.host_coarse
        vals = Float64.(Array(P.levels[L].a))
        copyto!(P.cinv, convert(Vector{Tp}, vec(_dense_inverse(hc.ia, hc.ja, vals, hc.nrow))))
    else
        # Single level: the dense inverse of the finest operator itself.
        _build!(P, P.fine[3], P.fine[4])
        return P
    end
    KernelAbstractions.synchronize(backend)
    P.stats.updates += 1
    P.stats.update_time += time() - t0
    return P
end

# Row signatures of the current matrix into `sig_new` (`Σ_j |a_ij|` at `i`, the row sum
# `Σ_j a_ij`, i.e. the storage and boundary terms, at `n + i`), and their summed absolute
# changes from `sig_old` and sums of the old values.
@inline function _signature_term(::Type{Ta}, i, sig_new, sig_old, amat, ia, n) where {Ta}
    s = zero(Ta)
    t = zero(Ta)
    @inbounds for ii in ia[i]:(ia[i + 1] - Int32(1))
        v = Ta(amat[ii])
        s += abs(v)
        t += v
    end
    # Rounded to the stored type, so that an unchanged matrix gives no change.
    s = Ta(convert(eltype(sig_new), s))
    t = Ta(convert(eltype(sig_new), t))
    @inbounds begin
        sig_new[i] = s
        sig_new[n + i] = t
        os = Ta(sig_old[i])
        ot = Ta(sig_old[n + i])
    end
    return (abs(s - os), abs(os), abs(t - ot), abs(ot))
end

# Relative change of the row signatures since the last build or update (`P.sig`); the new
# signatures are in `P.sig2` (swapped into `P.sig` when the hierarchy is updated).
function _matrix_change!(P::AMGPreconditioner, amat, csr)
    d1, s1, d2, s2 = _mapreduce!(P.reducer, P.backend, _signature_term, _SUM4, P.n, P.sig2, P.sig, amat, csr.ia, P.n)
    P.sig, P.sig2 = P.sig2, P.sig
    rel(d, s) = s > 0 ? d / s : (d > 0 ? Inf : 0.0)
    return max(rel(d1, s1), rel(d2, s2))
end

function precond_setup!(P::AMGPreconditioner, solver, amat)
    csr = solver.csr
    # The Jacobi diagonal of the scaled operator: the finest-level smoother.
    _map!(P.backend, _setup_jacobi, P.n, solver.minv, solver.rowfac, solver.colfac, amat, csr.ia)
    op = _FineOp(amat, csr.ia, csr.ja, solver.rowfac, solver.colfac)
    P.fine = (op, solver.minv, solver, amat)
    if !P.built || P.rebuild || !P.options.reuse
        _build!(P, solver, amat)
        return P
    end
    P.fresh = false
    change = _matrix_change!(P, amat, csr)
    if P.force_update || change > P.options.update_tol
        P.skipped = false
        P.force_update = false
        _update!(P, op)
    else
        # Keep the coarse levels of the last update (the finest level always uses the
        # current matrix); the signatures stay those of the last update.
        P.sig, P.sig2 = P.sig2, P.sig
        P.skipped = true
        P.stats.skipped += 1
    end
    return P
end

function _precond_record!(P::AMGPreconditioner, result)
    st = P.options
    P.stats.solves += 1
    if P.fresh
        P.base_iters = result.iterations
    elseif result.iterations > max(st.rebuild_factor * P.base_iters, P.base_iters + st.rebuild_min)
        # Degraded: update the coarse levels if the last setup skipped it, else rebuild.
        P.skipped ? (P.force_update = true) : (P.rebuild = true)
    end
    return nothing
end

function _precond_retry!(P::AMGPreconditioner)
    P.fresh && return false
    P.skipped ? (P.force_update = true) : (P.rebuild = true)
    return true
end

# --- V-cycle

# Smoother steps on A·x = f (D⁻¹ = dinv): the first from x = 0 (`_smooth_zero`), the others
# `x_out = x_in + d`, `d = c1·d + c2·D⁻¹(f − A·x_in)`.
@inline function _smooth_zero(n, xout, d, f, dinv, c2)
    T = eltype(xout)
    @inbounds begin
        dn = c2 * T(dinv[n]) * f[n]
        d[n] = dn
        xout[n] = dn
    end
    return nothing
end

@inline function _smooth_step(n, xout, xin, d, f, dinv, op, c1, c2)
    T = eltype(xout)
    @inbounds begin
        rn = f[n] - _row(op, n, xin)
        dn = c2 * T(dinv[n]) * rn
        c1 != zero(T) && (dn += c1 * d[n])
        d[n] = dn
        xout[n] = xin[n] + dn
    end
    return nothing
end

@inline function _residual(n, w, f, x, op)
    @inbounds w[n] = f[n] - _row(op, n, x)
    return nothing
end

# y = R·w (restriction, into the next level's type)
@inline function _restrict(I, fc, ria, rja, ra, w)
    T = eltype(fc)
    s = zero(T)
    @inbounds for t in ria[I]:(ria[I + 1] - Int32(1))
        s += ra[t] * T(w[rja[t]])
    end
    @inbounds fc[I] = s
    return nothing
end

# x += P·xc
@inline function _prolong_add(i, x, pia, pja, pa, xc)
    T = eltype(xc)
    s = zero(T)
    @inbounds for s_ in pia[i]:(pia[i + 1] - Int32(1))
        s += pa[s_] * xc[pja[s_]]
    end
    @inbounds x[i] += convert(eltype(x), s)
    return nothing
end

# x = M·f with a dense column-major n×n matrix M.
@inline function _dense_apply(i, x, M, f, nc)
    T = eltype(M)
    s = zero(T)
    @inbounds for j in 1:nc
        s += M[i + (j - 1) * nc] * T(f[j])
    end
    @inbounds x[i] = convert(eltype(x), s)
    return nothing
end

# Smoother coefficients (c1, c2) per step for λmax `lmax`.
function _smoother_coefficients(opt::AMGOptions, lmax, k = opt.degree)
    if opt.smoother === :jacobi
        return [(0.0, 4 / (3lmax)) for _ in 1:k]
    end
    b = lmax
    a = opt.cheb_ratio * lmax
    θ = (a + b) / 2
    δ = (b - a) / 2
    σ = θ / δ
    c = [(0.0, 1 / θ)]
    ρold = 1 / σ
    for _ in 2:k
        ρ = 1 / (2σ - ρold)
        push!(c, (ρ * ρold, 2ρ / δ))
        ρold = ρ
    end
    return c
end

# The smoother coefficients of level l, cached per level for its current λmax (T4.5: no
# allocation per V-cycle).
function _level_coefficients(P::AMGPreconditioner, l)
    L = length(P.lmax)
    if length(P.coefs) != L
        resize!(P.coefs, L)
        P.coef_lmax = fill(NaN, L)
    end
    if !(P.coef_lmax[l] === P.lmax[l])
        opt = P.options
        P.coefs[l] = _smoother_coefficients(opt, P.lmax[l], l == 1 && opt.fine_degree > 0 ? opt.fine_degree : opt.degree)
        P.coef_lmax[l] = P.lmax[l]
    end
    return P.coefs[l]
end

# Smooth A·x = f from x = 0 (`zero_guess`) or from `cur`; returns the buffer holding the
# result (`cur` or `oth`).
function _smooth!(backend, cur, oth, d, f, dinv, op, coefs, zero_guess)
    T = eltype(cur)
    n = length(f)
    for (s, (c1, c2)) in enumerate(coefs)
        if s == 1 && zero_guess
            _map!(backend, _smooth_zero, n, cur, d, f, dinv, T(c2))
        else
            _map!(backend, _smooth_step, n, oth, cur, d, f, dinv, op, T(c1), T(c2))
            cur, oth = oth, cur
        end
    end
    return cur, oth
end

# x ≈ A_l⁻¹·f on level l (x from zero); `x` holds the result.
function _cycle!(P::AMGPreconditioner, l, x, y, f, w, d, op, dinv)
    backend = P.backend
    L = length(P.levels)
    lev = P.levels[l]
    if l == L
        _map!(backend, _dense_apply, lev.n, x, P.cinv, f, lev.n)
        return x
    end
    opt = P.options
    coefs = _level_coefficients(P, l)
    # Pre-smoothing starts in `y` so that, after the 2·degree − 1 swaps of the pre- and
    # post-smoothing, the result is in `x`.
    cur, oth = _smooth!(backend, y, x, d, f, dinv, op, coefs, true)
    _map!(backend, _residual, lev.n, w, f, cur, op)
    nxt = P.levels[l + 1]
    _map!(backend, _restrict, nxt.n, nxt.f, lev.ria, lev.rja, lev.ra, w)
    _cycle!(P, l + 1, nxt.x, nxt.y, nxt.f, nxt.w, nxt.d, _CsrOp(nxt.ia, nxt.ja, nxt.a), nxt.dinv)
    _map!(backend, _prolong_add, lev.n, cur, lev.pia, lev.pja, lev.pa, nxt.x)
    cur, oth = _smooth!(backend, cur, oth, d, f, dinv, op, coefs, false)
    cur === x || copyto!(x, cur)
    return x
end

# `P.fine` is untyped (its operator type depends on the matrix): one dynamic dispatch per
# application, into the type-stable `_amg_apply!` (T4.5).
precond_apply!(z, P::AMGPreconditioner, r) = (_amg_apply!(P, z, r, P.fine[1], P.fine[2], nothing); z)

function _precond_apply_dot!(z, P::AMGPreconditioner, r, reducer, post, sa, ss)
    _fused_fine(P) || return (precond_apply!(z, P, r); false)
    _amg_apply!(P, z, r, P.fine[1], P.fine[2], (reducer, post, sa, ss))
    return true
end

function _amg_apply!(P::AMGPreconditioner, z, r, op, minv, red)
    _fused_fine(P) && return _fine_cycle1!(P, z, r, op, minv, red)
    _cycle!(P, 1, z, P.fy, r, P.fw, P.fd, op, minv)
    return nothing
end

_fused_fine(P::AMGPreconditioner) = P.options.fine_degree == 1 && length(P.levels) > 1

# `c·D⁻¹·f` as a lazily evaluated vector (the degree-1 pre-smoothing from zero).
struct _ScaledVec{V, D, T}
    f::V
    dinv::D
    c::T
end
Adapt.@adapt_structure _ScaledVec
Base.@propagate_inbounds Base.getindex(v::_ScaledVec{V, D, T}, j) where {V, D, T} = v.c * T(v.dinv[j]) * v.f[j]
Base.eltype(::Type{<:_ScaledVec{V, D, T}}) where {V, D, T} = T

# y = c·D⁻¹·f + P·xc (pre-smoothing plus coarse correction)
@inline function _prolong_set(i, y, f, dinv, c, pia, pja, pa, xc)
    T = eltype(xc)
    s = zero(T)
    @inbounds for k in pia[i]:(pia[i + 1] - Int32(1))
        s += pa[k] * xc[pja[k]]
    end
    @inbounds y[i] = c * eltype(y)(dinv[i]) * f[i] + convert(eltype(y), s)
    return nothing
end

# z = y + c·D⁻¹·(f − A·y) (post-smoothing), and f·z.
@inline function _smooth_last_dot(::Type{Ta}, n, z, y, f, dinv, op, c) where {Ta}
    T = eltype(z)
    @inbounds begin
        zn = y[n] + c * T(dinv[n]) * (f[n] - _row(op, n, y))
        z[n] = zn
        return (Ta(f[n]) * Ta(zn), zero(Ta), zero(Ta), zero(Ta))
    end
end

@inline function _smooth_last(n, z, y, f, dinv, op, c)
    T = eltype(z)
    @inbounds z[n] = y[n] + c * T(dinv[n]) * (f[n] - _row(op, n, y))
    return nothing
end

# The finest level with a degree-1 smoother, fused: residual of the pre-smoothed `c·D⁻¹·r`
# (one SpMV), restriction, coarse cycle, `y = c·D⁻¹·r + P·xc`, post-smoothing (one SpMV,
# with `r·z` when `red = (reducer, post, sa, ss)` is given, not synchronised). The same
# V-cycle as `_cycle!` with degree 1.
function _fine_cycle1!(P::AMGPreconditioner, z, r, op, dinv, red)
    backend = P.backend
    lev = P.levels[1]
    nxt = P.levels[2]
    n = lev.n
    T = eltype(z)
    c = T(_level_coefficients(P, 1)[1][2])
    _map!(backend, _residual, n, P.fw, r, _ScaledVec(r, dinv, c), op)
    _map!(backend, _restrict, nxt.n, nxt.f, lev.ria, lev.rja, lev.ra, P.fw)
    _cycle!(P, 2, nxt.x, nxt.y, nxt.f, nxt.w, nxt.d, _CsrOp(nxt.ia, nxt.ja, nxt.a), nxt.dinv)
    _map!(backend, _prolong_set, n, P.fy, r, dinv, c, lev.pia, lev.pja, lev.pa, nxt.x)
    if isnothing(red)
        _map!(backend, _smooth_last, n, z, P.fy, r, dinv, op, c)
        return nothing
    end
    reducer, post, sa, ss = red
    _reduce_async!(reducer, backend, _smooth_last_dot, _SUM4, n, z, P.fy, r, dinv, op, c; post, sa, ss)
    return nothing
end

"""
    amg_stats(solver::KrylovSolver) -> Union{AMGStats, Nothing}

The `AMGStats` of the solver's AMG preconditioner (`nothing` for other preconditioners).
"""
amg_stats(solver) = solver.precond isa AMGPreconditioner ? solver.precond.stats : nothing
