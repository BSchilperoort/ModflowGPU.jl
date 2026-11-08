# Chebyshev polynomial preconditioner (T4.2; not in MF6). `M⁻¹ = p(D⁻¹·Â)·D⁻¹`, with `Â`
# the scaled operator of the Krylov solver (src/solver/precond/precond.jl), `D` its diagonal
# and `p` the degree-`k − 1` polynomial of `k` steps of the Chebyshev iteration for
# `D⁻¹·Â·z = D⁻¹·r` from `z = 0` on the eigenvalue interval `[λmax/ratio, λmax]` (Saad,
# Iterative Methods for Sparse Linear Systems, 2nd ed., algorithm 12.1). `λmax` comes from a
# few Jacobi-PCG (Lanczos) steps at every setup, times a safety factor and capped by the
# Gershgorin bound `max_n Σ_m |â_nm| / |â_nn|`.
#
# For CG the preconditioner must be SPD: `p(D⁻¹Â)D⁻¹ = D^{-1/2}·p(D^{-1/2}ÂD^{-1/2})·D^{-1/2}`
# is symmetric, and `λ·p(λ) = 1 − T_k((b + a − 2λ)/(b − a))/T_k((b + a)/(b − a)) > 0` for
# `0 < λ <= b`, so it is positive definite as long as `b` bounds the spectrum from above
# (hence the safety factor and the Gershgorin cap). Each application costs `k − 1` SpMVs
# with the scaled operator stored in `Tp` and no reductions (no host synchronisation).

"""
    ChebyshevPreconditioner{Tp}

Jacobi-preconditioned Chebyshev polynomial (src/solver/precond/chebyshev.jl): the scaled
operator `vals` (`nja`, in `Tp`, CSR of `csr`), its inverse diagonal `dinv`, two work
vectors in `Ts`, the interval `bounds = [a, b]` (host, from the last setup) and the
settings.

    ChebyshevPreconditioner(backend, csr; Ts, Tp, degree = 4, ratio = 30.0, lanczos = 10,
                            safety = 1.1)

Settings (`precond_options` of `KrylovSettings` with `preconditioner = :chebyshev`):
`degree` (Chebyshev steps per application: `degree − 1` SpMVs), `ratio` (`λmax/λmin` of
the interval), `lanczos` (Jacobi-PCG steps for the `λmax` estimate) and `safety` (factor
on the estimate).
"""
struct ChebyshevPreconditioner{VP <: AbstractVector, VS <: AbstractVector, C <: CsrStructure} <: AbstractPreconditioner
    csr::C
    vals::VP
    dinv::VP
    d::VS
    xtmp::VS
    bounds::Vector{Float64}
    degree::Int
    ratio::Float64
    lanczos::Int
    safety::Float64
end

function ChebyshevPreconditioner(
        backend::Backend, csr::CsrStructure; Ts::Type, Tp::Type, degree::Integer = 4,
        ratio::Real = 30.0, lanczos::Integer = 10, safety::Real = 1.1,
    )
    degree >= 1 || throw(ArgumentError("Chebyshev degree must be >= 1"))
    ratio > 1 || throw(ArgumentError("Chebyshev ratio must be > 1"))
    n = Int(csr.nodes)
    return ChebyshevPreconditioner(
        csr, KernelAbstractions.zeros(backend, Tp, Int(csr.nja)), KernelAbstractions.zeros(backend, Tp, n),
        KernelAbstractions.zeros(backend, Ts, n), KernelAbstractions.zeros(backend, Ts, n),
        [1.0 / ratio, 1.0], Int(degree), Float64(ratio), Int(lanczos), Float64(safety),
    )
end

# The interval stays on the host (copied, not moved, by Adapt).
Adapt.adapt_structure(to, P::ChebyshevPreconditioner) = ChebyshevPreconditioner(
    adapt(to, P.csr), adapt(to, P.vals), adapt(to, P.dinv), adapt(to, P.d), adapt(to, P.xtmp), copy(P.bounds),
    P.degree, P.ratio, P.lanczos, P.safety,
)

@inline function _cheb_setup_row(n, vals, dinv, amat, ia, ja, rowfac, colfac)
    Tp = eltype(vals)
    @inbounds begin
        _scaled_row!(vals, ia[n], n, amat, ia, ja, rowfac, colfac)
        d = vals[ia[n]]
        dinv[n] = d != zero(Tp) ? one(Tp) / d : one(Tp)
    end
    return nothing
end

# Gershgorin bound of D⁻¹·Â for row n.
@inline function _gershgorin_term(::Type{Ta}, n, vals, dinv, ia) where {Ta}
    s = zero(Ta)
    @inbounds for ii in ia[n]:(ia[n + 1] - Int32(1))
        s += abs(Ta(vals[ii]))
    end
    return (@inbounds(s * abs(Ta(dinv[n]))), zero(Ta), zero(Ta), zero(Ta))
end

# Row n of the stored scaled operator: Σ_m vals_nm · x_m (in Ts).
@inline function _vals_row(n, vals, ia, ja, x)
    Ts = eltype(x)
    s = zero(Ts)
    @inbounds for ii in ia[n]:(ia[n + 1] - Int32(1))
        s += Ts(vals[ii]) * x[ja[ii]]
    end
    return s
end

# A deterministic pseudo-random start vector in [−1/2, 1/2) (integer hash, no Float64).
@inline function _hash_unit(::Type{Ts}, n) where {Ts}
    h = ((n % UInt32) * 0x9e3779b9) ⊻ 0x85ebca6b
    h = (h ⊻ (h >> 13)) * 0xc2b2ae35
    h = h ⊻ (h >> 16)
    return Ts(h >> 8) / Ts(16777216) - Ts(0.5)
end

# Lanczos via Jacobi-PCG (the tridiagonal from the CG coefficients).
@inline function _lz_init(::Type{Ta}, n, r, z, p, dinv) where {Ta}
    Ts = eltype(r)
    @inbounds begin
        rn = _hash_unit(Ts, n)
        r[n] = rn
        zn = Ts(dinv[n]) * rn
        z[n] = zn
        p[n] = zn
    end
    return (Ta(rn) * Ta(zn), zero(Ta), zero(Ta), zero(Ta))
end

@inline function _lz_apply(::Type{Ta}, n, q, p, vals, ia, ja) where {Ta}
    qn = _vals_row(n, vals, ia, ja, p)
    @inbounds q[n] = qn
    return (@inbounds(Ta(p[n]) * Ta(qn)), zero(Ta), zero(Ta), zero(Ta))
end

@inline function _lz_update(::Type{Ta}, n, r, z, q, dinv, alpha) where {Ta}
    Ts = eltype(r)
    @inbounds begin
        rn = r[n] - alpha * q[n]
        r[n] = rn
        zn = Ts(dinv[n]) * rn
        z[n] = zn
    end
    return (Ta(rn) * Ta(zn), zero(Ta), zero(Ta), zero(Ta))
end

"""
    lanczos_extremes(P::ChebyshevPreconditioner, red, r, z, p, q; steps) -> (λmin, λmax)

Ritz values of `D⁻¹·Â` after `steps` Jacobi-PCG steps from a pseudo-random vector (the
Lanczos tridiagonal from the CG coefficients); `r`, `z`, `p`, `q` are work vectors.
Stops early on breakdown (a non-positive `p·Âp`, i.e. an indefinite operator, or a zero
residual).
"""
function lanczos_extremes(P::ChebyshevPreconditioner, red, r, z, p, q; steps = P.lanczos)
    backend = get_backend(r)
    csr = P.csr
    n = Int(csr.nodes)
    Ts = eltype(r)
    rz = _mapreduce!(red, backend, _lz_init, _SUM4, n, r, z, p, P.dinv)[1]
    alphas = Float64[]
    betas = Float64[]
    for _ in 1:steps
        (rz > 0 && isfinite(rz)) || break
        pq = _mapreduce!(red, backend, _lz_apply, _SUM4, n, q, p, P.vals, csr.ia, csr.ja)[1]
        (pq > 0 && isfinite(pq)) || break
        alpha = rz / pq
        rz1 = _mapreduce!(red, backend, _lz_update, _SUM4, n, r, z, q, P.dinv, Ts(alpha))[1]
        beta = rz1 / rz
        push!(alphas, alpha)
        push!(betas, beta)
        _map!(backend, _cg_direction, n, p, z, Ts(beta))
        rz = rz1
    end
    k = length(alphas)
    k == 0 && return (NaN, NaN)
    dg = [1 / alphas[j] + (j > 1 ? betas[j - 1] / alphas[j - 1] : 0.0) for j in 1:k]
    ev = [sqrt(abs(betas[j])) / alphas[j] for j in 1:(k - 1)]
    λ = LinearAlgebra.eigvals(LinearAlgebra.SymTridiagonal(dg, ev))
    return (minimum(λ), maximum(λ))
end

function precond_setup!(P::ChebyshevPreconditioner, solver, amat)
    csr = P.csr
    backend = get_backend(P.vals)
    n = Int(csr.nodes)
    _map!(backend, _cheb_setup_row, n, P.vals, P.dinv, amat, csr.ia, csr.ja, solver.rowfac, solver.colfac)
    gersh = abs(_mapreduce!(solver.reducer, backend, _gershgorin_term, (_MaxAbsOp(), _SumOp(), _SumOp(), _SumOp()), n, P.vals, P.dinv, csr.ia)[1])
    # The Krylov vectors are free until the solve initialises them.
    _, lmax = lanczos_extremes(P, solver.reducer, solver.x, solver.r, solver.p, solver.q)
    b = isfinite(lmax) && lmax > 0 ? min(P.safety * lmax, gersh) : gersh
    b > 0 || (b = 1.0)
    P.bounds[1] = b / P.ratio
    P.bounds[2] = b
    return P
end

@inline function _cheb_first(n, x, d, dinv, r, c)
    Ts = eltype(x)
    @inbounds begin
        dn = c * Ts(dinv[n]) * r[n]
        d[n] = dn
        x[n] = dn
    end
    return nothing
end

@inline function _cheb_step(n, xnew, x, d, r, vals, ia, ja, dinv, c1, c2)
    Ts = eltype(x)
    @inbounds begin
        res = Ts(dinv[n]) * (r[n] - _vals_row(n, vals, ia, ja, x))
        dn = c1 * d[n] + c2 * res
        d[n] = dn
        xnew[n] = x[n] + dn
    end
    return nothing
end

function precond_apply!(z, P::ChebyshevPreconditioner, r)
    Ts = eltype(z)
    csr = P.csr
    backend = get_backend(z)
    n = Int(csr.nodes)
    a, b = P.bounds
    θ = (b + a) / 2
    δ = (b - a) / 2
    σ = θ / δ
    ρ = 1 / σ
    # Ping-pong between z and xtmp so that the last step writes z.
    xa = iseven(P.degree - 1) ? z : P.xtmp
    xb = xa === z ? P.xtmp : z
    _map!(backend, _cheb_first, n, xa, P.d, P.dinv, r, Ts(1 / θ))
    for _ in 2:P.degree
        ρn = 1 / (2σ - ρ)
        _map!(backend, _cheb_step, n, xb, xa, P.d, r, P.vals, csr.ia, csr.ja, P.dinv, Ts(ρn * ρ), Ts(2ρn / δ))
        xa, xb = xb, xa
        ρ = ρn
    end
    return z
end
