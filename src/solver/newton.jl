# Outer-iteration parts of the Newton–Raphson formulation (T6.2) that NumericalSolution.f90
# and gwf.f90 apply around the linear solve:
#
# - Pseudo-transient continuation (`sln_calc_ptc`, `gwf_ptc`, the PTC part of `sln_ls`): in
#   a steady-state stress period of a Newton model (unless IMS NO_PTC ALL, or NO_PTC FIRST
#   in the first period) `1/ptcdel` is subtracted from the diagonal of the active rows,
#   with `ptcdel = 1 / max_n(|res_n| / V_n)` in the first outer iteration (`V_n` the cell
#   volume at full saturation; `1/(10·delt)` when every residual is zero) and
#   `ptcdel·(l2norm0 / l2norm)` later. It is skipped in the first outer iteration of later
#   steps when the residual norm did not grow since the last PTC iteration, and in later
#   outer iterations when it is the same (`is_close`). The right-hand side gets
#   `−ptcval·h'`, so the residual is unchanged: only the matrix diagonal changes. `ptcdel`
#   and `l2norm0` carry over between time steps, as in MF6. The DEV_PTC options are
#   rejected by the loader (developer options).
# - Newton under-relaxation (NEWTON UNDER_RELAXATION; `gwf_nur`, `npf_nur`,
#   `sln_nur_has_converged`): after the update of a non-converged iteration, a convertible
#   cell whose head fell below the bottom of the lowest convertible cell under it
#   (`newton_ur_bottoms`) gets `0.1·h_prev + 0.9·bottom`; the step then counts as converged
#   when both the largest change of the iteration and the largest change of the cells
#   (`dxold`: the unrelaxed correction, or `h_prev − h` of an adjusted cell) meet
#   OUTER_DVCLOSE, and the reported change becomes the largest `h − h_prev`.

"""
    PtcState

Pseudo-transient continuation state of a solution (`ptcdel`, `l2norm0` of
NumericalSolution.f90), kept across time steps; host Float64.
"""
mutable struct PtcState
    ptcdel::Float64
    l2norm0::Float64
end

PtcState() = PtcState(0.0, 0.0)

# |res_n| / V_n and res_n² over the active cells (gwf_ptc, sln_l2norm).
@inline _ptc_term(::Type{Ta}, i, res, ibound, vol) where {Ta} =
    @inbounds ibound[i] > 0 ? (Ta(res[i]) / Ta(vol[i]), Ta(res[i]) * Ta(res[i]), zero(Ta), zero(Ta)) : (zero(Ta), zero(Ta), zero(Ta), zero(Ta))

const _PTC_OPS = (_MaxAbsOp(), _SumOp(), _SumOp(), _SumOp())

# `sln_ls`: the PTC term on the diagonal of the active rows (`rhs −= ptcval·x` leaves the
# residual unchanged and the correction solve does not read the right-hand side).
@inline function _ptc_diag(i, amat, diag, ia, ibound, ptcval)
    @inbounds if ibound[i] > 0
        d = diag[i] - ptcval
        diag[i] = d
        amat[ia[i]] = d
    end
    return nothing
end

"""
    ptc_allowed(no_ptc::Symbol, kper) -> Bool

Whether IMS allows pseudo-transient continuation in stress period `kper` (`sln_ls`'s
`iallowptc`): not with NO_PTC ALL, not in the first period with NO_PTC FIRST.
"""
ptc_allowed(no_ptc::Symbol, kper::Integer) = no_ptc === :all ? false : (no_ptc === :first ? kper > 1 : true)

"""
    apply_ptc!(ptc::PtcState, red::Reducer, m::GwfModel, kiter, kstp) -> Float64

Pseudo-transient continuation for outer iteration `kiter` of time step `kstp` (see the
header of src/solver/newton.jl), after the Newton formulation of a steady-state period:
decides from the residual `m.res` whether to apply it and subtracts `1/ptcdel` from the
diagonal of the active rows. Returns the value subtracted (0 when not applied). One device
reduction.
"""
function apply_ptc!(ptc::PtcState, red::Reducer, m::GwfModel{T}, kiter::Integer, kstp::Integer) where {T}
    backend = m.backend
    n = Int(m.conn.nodes)
    ptcf, s2, _, _ = @timed_section :outer_reduction _mapreduce!(red, backend, _ptc_term, _PTC_OPS, n, m.res, m.ibound, m.vol)
    ptcf = abs(ptcf)
    l2norm = sqrt(max(s2, 0.0))
    # gwf_ptc: protection for zero residuals.
    ptcf == 0.0 && (ptcf = 1.0 / (m.delt * 10.0))
    if kiter == 1
        (m.kper > 1 || kstp > 1) && l2norm <= ptc.l2norm0 && return 0.0
        ptc.ptcdel = 1.0 / ptcf
    else
        _is_close(l2norm, ptc.l2norm0) && return 0.0
        ptc.ptcdel = l2norm > 0.0 ? ptc.ptcdel * (ptc.l2norm0 / l2norm) : 0.0
    end
    ptcval = ptc.ptcdel > 0.0 ? 1.0 / ptc.ptcdel : 1.0
    _map!(backend, _ptc_diag, n, m.sys.amat, m.sys.diag, m.csr.ia, m.ibound, T(ptcval))
    ptc.l2norm0 = l2norm
    return ptcval
end

# npf_nur on node i after the update (the bottom `nurbot[i]` is -Inf where it does not
# apply), and the reductions of `sln_nur_has_converged`: the change `h − h_prev` (argmax,
# `sln_get_dxmax`), the largest |dxold| (`sln_maxval`; `δ` or `h_prev − h` of an adjusted
# cell) and the number of adjusted cells.
@inline function _nur_term(::Type{Ta}, i, h, hprev, δ, nurbot, ibound) where {Ta}
    T = eltype(h)
    @inbounds begin
        ibound[i] > 0 || return (zero(Ta), zero(Ta), zero(Ta), zero(Ta))
        hn = h[i]
        hp = hprev[i]
        botm = nurbot[i]
        dx = δ[i]
        adj = zero(Ta)
        if hn < botm
            hn = hp * (one(T) - T(0.9)) + botm * T(0.9)
            h[i] = hn
            dx = hp - hn
            adj = one(Ta)
        end
        return (Ta(hn) - Ta(hp), Ta(dx), adj, zero(Ta))
    end
end

const _NUR_OPS = (_ArgMaxAbsOp(), _MaxAbsOp(), _SumOp(), _SumOp())

"""
    newton_ur!(red::Reducer, m::GwfModel, hprev) -> (adjusted, dxold_max, hncg, lrch)

NEWTON UNDER_RELAXATION after the head update of a non-converged outer iteration (`npf_nur`;
see the header of src/solver/newton.jl): `hprev` holds the offset heads before the update.
Returns whether a cell was adjusted, the largest |dxold| and the largest change
`h − hprev` with its node. One device reduction.
"""
function newton_ur!(red::Reducer, m::GwfModel, hprev)
    (hncg, dxold, nadj, _), lrch = @timed_section :outer_reduction _mapreduce_loc!(
        red, m.backend, _nur_term, _NUR_OPS, Int(m.conn.nodes), m.h, hprev, m.δ, m.nurbot, m.ibound,
    )
    return nadj > 0, abs(dxold), hncg, lrch
end
