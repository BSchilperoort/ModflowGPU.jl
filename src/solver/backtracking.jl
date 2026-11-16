# Backtracking of the outer iterations (NumericalSolution.f90 `sln_backtracking`,
# `sln_backtracking_xupdate`, `get_backtracking_flag`, `apply_backtracking`, `sln_l2norm`,
# `sln_calc_residual`; T6.2a).
#
# With BACKTRACKING_NUMBER > 0 (COMPLEXITY COMPLEX sets 20), every outer iteration starts
# with the L2 norm of the residual of the system formulated at the current heads (active
# cells only; MF6's `A·x − b`, here the flux-form residual `m.res`, PLAN.md 3.7 rule 2,
# whose norm is the same). In the first outer iteration of a step it is stored as
# `res_prev`. Later, when it exceeds `res_prev · BACKTRACKING_TOLERANCE`, the head change
# of the previous outer iteration (`x − xtemp`: the under-relaxed update, `h' − hprev`)
# is reduced by BACKTRACKING_REDUCTION_FACTOR, up to BACKTRACKING_NUMBER times: stop when
# the reduced change would fall below OUTER_DVCLOSE (`breduc · max|Δh| < dvclose`, flag
# 4), at the last allowed reduction (flag 2), when the norm drops below
# `res_prev · tol` (flag 1) or below BACKTRACKING_RESIDUAL_LIMIT. Each reduction
# reformulates the system and computes its norm. The final norm becomes `res_prev`.
# MF6 applies it whenever BACKTRACKING_NUMBER > 0, whatever UNDER_RELAXATION is (its
# `ibflag` is not read). Newton (T6.2) reuses this with a formulation without the Newton
# terms (`sln_buildsystem(kiter, inewton = 0)`).

"""
    BacktrackRecord

Backtracking in one outer iteration (MF6's "Backtracking" row of the outer iteration
table): `kiter`, `flag` (0 none or first iteration, 1 the residual met the tolerance, 2
the last allowed reduction, 4 the reduced change was below OUTER_DVCLOSE; 0 also when the
residual limit ended it), `count` (reductions applied, BACKTRACK ITERATIONS), `resin`
(incoming L2 norm) and `resout` (outgoing, `res_prev` of the next iteration).
"""
struct BacktrackRecord
    kiter::Int
    flag::Int
    count::Int
    resin::Float64
    resout::Float64
end

@kernel function backtrack_update_kernel!(h, @Const(hprev), @Const(ibound), breduc)
    n = @index(Global)
    @inbounds if ibound[n] > 0
        h[n] = hprev[n] + breduc * (h[n] - hprev[n])
    end
end

"""
    backtrack_update!(h, hprev, ibound, breduc)

`apply_backtracking`: `h' ← hprev + breduc·(h' − hprev)` on the active cells
(`ibound > 0`), with `hprev` the offset heads at the start of the previous outer
iteration (MF6's `xtemp`).
"""
function backtrack_update!(h, hprev, ibound, breduc::Real)
    backend = get_backend(h)
    backtrack_update_kernel!(backend, _WG)(h, hprev, ibound, eltype(h)(breduc); ndrange = length(h))
    return h
end

# sln_l2norm (Σ r² over the active cells; Float64 accumulation where the backend has it)
# and get_backtracking_flag (max |h' − hprev| over the active cells).
@inline _res2_term(::Type{Ta}, i, res, ibound) where {Ta} =
    @inbounds (ibound[i] > 0 ? Ta(res[i]) * Ta(res[i]) : zero(Ta), zero(Ta), zero(Ta), zero(Ta))

@inline _btdx_term(::Type{Ta}, i, h, hprev, ibound) where {Ta} =
    @inbounds (ibound[i] > 0 ? Ta(h[i]) - Ta(hprev[i]) : zero(Ta), zero(Ta), zero(Ta), zero(Ta))

const _BTDX_OPS = (_MaxAbsOp(), _SumOp(), _SumOp(), _SumOp())

"""
    residual_l2norm(red::Reducer, res, ibound) -> Float64

`sln_l2norm`: the L2 norm of the residual `res` over the active cells (`ibound > 0`;
`sln_calc_residual` zeroes the others), accumulated in the reducer's type (Float64 where
the backend has it, PLAN.md 3.7).
"""
function residual_l2norm(red::Reducer, res, ibound)
    s = _mapreduce!(red, get_backend(res), _res2_term, _SUM4, length(res), res, ibound)[1]
    return sqrt(max(s, 0.0))
end

"""
    backtrack!(reformulate, red, h, hprev, res, ibound, kiter, res_prev; numtrack, btol, breduc, res_lim, dvclose)
        -> (record::BacktrackRecord, res_prev)

`sln_backtracking` for outer iteration `kiter`, after the system was formulated at the
current offset heads `h` (its residual in `res`). `reformulate()` formulates the system
again after a reduction (`sln_buildsystem(kiter, inewton = 0)`), `hprev` holds the heads
at the start of the previous outer iteration and `res_prev` the outgoing norm of that
iteration. Returns the record of this iteration and the new `res_prev`. The norms and the
maximum change are device reductions; one host decision per reduction.
"""
function backtrack!(
        reformulate, red::Reducer, h, hprev, res, ibound, kiter::Integer, res_prev::Float64;
        numtrack::Integer, btol::Float64, breduc::Float64, res_lim::Float64, dvclose::Float64,
    )
    backend = get_backend(h)
    res_new = @timed_section :outer_reduction residual_l2norm(red, res, ibound)
    kiter == 1 && return BacktrackRecord(kiter, 0, 0, res_new, res_new), res_new
    resin = res_new
    flag = 0
    count = 0
    if res_new > res_prev * btol
        for nb in 1:numtrack
            # get_backtracking_flag: only when the reduced change is at least OUTER_DVCLOSE.
            dxmax = @timed_section :outer_reduction _mapreduce!(red, backend, _btdx_term, _BTDX_OPS, length(h), h, hprev, ibound)[1]
            if !(breduc * abs(dxmax) >= dvclose)
                flag = 4
                break
            end
            @timed_section :underrelax backtrack_update!(h, hprev, ibound, breduc)
            count = nb
            reformulate()
            res_new = @timed_section :outer_reduction residual_l2norm(red, res, ibound)
            if nb == numtrack
                flag = 2
                break
            end
            if res_new < res_prev * btol
                flag = 1
                break
            end
            res_new < res_lim && break
        end
    end
    return BacktrackRecord(kiter, flag, count, resin, res_new), res_new
end
