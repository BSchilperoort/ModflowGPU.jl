# Head update of an outer iteration with IMS under-relaxation (NumericalSolution.f90
# `sln_underrelax`, `sln_calcdx`). The correction `δ` of the linear solve is MF6's
# `x − xtemp`; the update `h' ← h' + w·δ` is applied to the offset heads of active cells
# (PLAN.md section 3.7, rule 3).
#
# - UNDER_RELAXATION NONE: `w = 1`.
# - SIMPLE (`nonmeth = 1`): `w = gamma`.
# - COOLEY (`nonmeth = 2`): one factor for all cells from the signed maximum change `bigch`
#   and its weighted history (host scalars, Float64).
# - DBD (`nonmeth = 3`): delta-bar-delta, a factor per cell with momentum (device state in
#   `T`: `wsave`, `hchold`, `deold`).
#
# MF6 applies the under-relaxation only when the outer iteration has not converged; a
# converged iteration keeps the full solution (`w = 1`).

"""
    UnderRelaxation{V}

Under-relaxation settings and state of one model (`sln_underrelax`): `method` (`:none`,
`:simple`, `:cooley`, `:dbd`), `gamma`, `theta`, `kappa`, `momentum` (IMS
UNDER_RELAXATION_*), the COOLEY history (`relaxold`, `bigchold`, host Float64) and the DBD
state per cell (`wsave`, `hchold`, `deold`, device vectors; empty for other methods).

    UnderRelaxation(backend, T, ims::ImsSpec, nodes)
"""
mutable struct UnderRelaxation{V <: AbstractVector}
    const method::Symbol
    const gamma::Float64
    const theta::Float64
    const kappa::Float64
    const momentum::Float64
    relaxold::Float64
    bigchold::Float64
    const wsave::V
    const hchold::V
    const deold::V
end

function UnderRelaxation(backend::Backend, ::Type{T}, ims::ImsSpec, nodes::Integer) where {T <: AbstractFloat}
    method = ims.under_relaxation
    method in (:none, :simple, :cooley, :dbd) || throw(ArgumentError("unknown IMS UNDER_RELAXATION $method"))
    n = method === :dbd ? Int(nodes) : 0
    z() = KernelAbstractions.zeros(backend, T, n)
    return UnderRelaxation(
        method, ims.under_relaxation_gamma, ims.under_relaxation_theta, ims.under_relaxation_kappa,
        ims.under_relaxation_momentum, 1.0, 0.0, z(), z(), z(),
    )
end

# Adapt: the DBD vectors move, the settings and COOLEY history are kept.
Adapt.adapt_structure(to, u::UnderRelaxation) = UnderRelaxation(
    u.method, u.gamma, u.theta, u.kappa, u.momentum, u.relaxold, u.bigchold,
    adapt(to, u.wsave), adapt(to, u.hchold), adapt(to, u.deold),
)

@kernel function relax_update_kernel!(h, @Const(δ), @Const(ibound), w)
    n = @index(Global)
    @inbounds if ibound[n] > 0
        h[n] = h[n] + w * δ[n]
    end
end

# sln_underrelax, nonmeth 3 (delta-bar-delta), per cell.
@kernel function dbd_update_kernel!(
        h, wsave, hchold, deold, @Const(δ), @Const(ibound), first::Bool, theta, kappa, gamma, amom,
    )
    n = @index(Global)
    T = eltype(h)
    @inbounds if ibound[n] > 0
        delx = δ[n]
        ws = first ? one(T) : wsave[n]
        de = first ? zero(T) : deold[n]
        # Flip-flop: decrease the factor; same sign: increase it.
        ww = de * delx < zero(T) ? theta * ws : ws + kappa
        ww = min(ww, one(T))
        wsave[n] = ww
        hc = first ? delx : (one(T) - gamma) * delx + gamma * hchold[n]
        hchold[n] = hc
        deold[n] = delx
        h[n] = h[n] + (delx * ww + amom * hc)
    end
end

"""
    underrelax!(ur::UnderRelaxation, h, δ, ibound, kiter, bigch; converged = false) -> Float64

Update the offset heads `h` of the active cells (`ibound > 0`) with the correction `δ` of
outer iteration `kiter` (`sln_underrelax`): `h += w·δ` with `w = 1` when the iteration
converged or without under-relaxation, `gamma` (SIMPLE), the COOLEY factor from `bigch`
(the signed maximum change, `hncg`), or per cell with DBD. Returns the COOLEY or SIMPLE
factor (1 otherwise).
"""
function underrelax!(ur::UnderRelaxation, h, δ, ibound, kiter::Integer, bigch::Real; converged::Bool = false)
    backend = get_backend(h)
    T = eltype(h)
    n = length(h)
    update!(w) = (relax_update_kernel!(backend, _WG)(h, δ, ibound, T(w); ndrange = n); w)
    (converged || ur.method === :none) && return update!(1.0)
    if ur.method === :simple
        return update!(ur.gamma)
    elseif ur.method === :cooley
        bigch = Float64(bigch)
        if kiter == 1
            relax = 1.0
            ur.relaxold = 1.0
            ur.bigchold = bigch
        else
            es = bigch / (ur.bigchold * ur.relaxold)
            aes = abs(es)
            relax = es < -1.0 ? 0.5 / aes : (3.0 + es) / (3.0 + aes)
        end
        ur.relaxold = relax
        ur.bigchold = (1.0 - ur.gamma) * bigch + ur.gamma * ur.bigchold
        # MF6 changes x only when relax < 1; otherwise x keeps the full solution.
        return update!(relax < 1.0 ? relax : 1.0)
    else
        amom = kiter > 4 ? ur.momentum : 0.0
        dbd_update_kernel!(backend, _WG)(
            h, ur.wsave, ur.hchold, ur.deold, δ, ibound, kiter == 1, T(ur.theta), T(ur.kappa), T(ur.gamma), T(amom);
            ndrange = n,
        )
        return 1.0
    end
end
