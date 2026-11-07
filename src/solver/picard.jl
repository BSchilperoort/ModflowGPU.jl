# The outer (Picard) iteration of a time step (NumericalSolution.f90 `sln_ca`, `solve`,
# `sln_get_dxmax`, `sln_has_converged`, `sln_underrelax`; PLAN.md section 3.7, rule 3).
#
# Per outer iteration `kiter` (device work, one host decision):
# 1. formulate and assemble from the current offset heads and compute the flux-form
#    residual (`formulate!`);
# 2. solve the correction system `A·δ = res` with a zero initial guess (`solve!`; MF6
#    solves `A·x = b` from `x`, and `δ` is its `x − xtemp`);
# 3. `hncg`, the signed largest `δ` over the active cells, and its node `lrch`
#    (`sln_get_dxmax`), one device reduction;
# 4. converged when the inner solve converged and `|hncg| <= OUTER_DVCLOSE`
#    (`sln_has_converged`); then `h' += δ`, otherwise the under-relaxed update
#    (`sln_underrelax`, src/solver/underrelax.jl).
# The loop stops at convergence or after OUTER_MAXIMUM iterations.
#
# Not implemented: backtracking (T6.2), pseudo-transient continuation and Newton (T6.1),
# package convergence checks (none of the supported packages has one).

"""
    FLOAT32_CLOSURES

The tightest closures `:single` (Float32) runs use (PLAN.md section 3.7, "Closures in
Float32"; measured in docs/precision.md): IMS closures below these are raised to them in
Float32 by `PicardSettings`:

- `outer_dvclose`, `inner_dvclose`: fixed floors (m);
- `dvclose_head_factor`: OUTER_DVCLOSE is also at least this factor times
  `eps(Float32)·max_n |h'_n|`, the resolution of the offset heads (checked every outer
  iteration; e.g. 1.2e-4 m on ex-gwf-toth, whose offsets reach 1010 m);
- `inner_rclose_factor`: an absolute INNER_RCLOSE is at least this factor times
  `eps(Float32)·max_n Σ_m |terms|` of the first outer iteration of a step, the rounding
  level of the flux-form residual (rule 2).
"""
const FLOAT32_CLOSURES = (; outer_dvclose = 1.0e-6, inner_dvclose = 1.0e-6, dvclose_head_factor = 1.0, inner_rclose_factor = 1.0)

"""
    PicardSettings

Outer-iteration settings of one solution (host, Float64): `dvclose` (OUTER_DVCLOSE),
`maxiter` (OUTER_MAXIMUM), `krylov::KrylovSettings` (inner solver), `ims::ImsSpec` (for the
under-relaxation), `symmetric` (CG: the constant-head columns go to the right-hand side,
`sln_ls`), and the Float32 floors `dvclose_head_factor` and `rclose_floor_factor` (see
`FLOAT32_CLOSURES`; 0 for Float64).

    PicardSettings(ims::ImsSpec, T; closures = nothing)

Takes OUTER_DVCLOSE, OUTER_MAXIMUM and the inner settings from `ims`. `closures` (a
NamedTuple with any of `outer_dvclose`, `outer_maximum`, `inner_dvclose`, `inner_rclose`,
`inner_maximum`) overrides them. For `T = Float32`, closures below `FLOAT32_CLOSURES` are
raised to them (logged), since Float32 cannot reach MF6's tight closures (PLAN.md 3.7).
Warns when the input would use backtracking (BACKTRACKING_NUMBER > 0), which is not
implemented until T6.2.
"""
struct PicardSettings
    dvclose::Float64
    maxiter::Int
    krylov::KrylovSettings
    ims::ImsSpec
    symmetric::Bool
    dvclose_head_factor::Float64
    rclose_floor_factor::Float64
end

function PicardSettings(ims::ImsSpec, ::Type{T}; closures = nothing) where {T <: AbstractFloat}
    get_c(k, default) = (isnothing(closures) || !haskey(closures, k)) ? default : getfield(closures, k)
    ims.backtracking_number > 0 &&
        @warn "IMS BACKTRACKING_NUMBER = $(ims.backtracking_number) (COMPLEXITY $(uppercase(String(ims.complexity))) or given): backtracking is not implemented until T6.2; outer iterations may differ from MF6" ims.file
    ims.under_relaxation in (:none, :simple, :cooley, :dbd) || throw(ArgumentError("unknown IMS UNDER_RELAXATION $(ims.under_relaxation)"))
    dvclose = Float64(get_c(:outer_dvclose, ims.outer_dvclose))
    maxiter = Int(get_c(:outer_maximum, ims.outer_maximum))
    k = KrylovSettings(ims)
    kdv = Float64(get_c(:inner_dvclose, k.dvclose))
    krc = Float64(get_c(:inner_rclose, k.rclose))
    kmax = Int(get_c(:inner_maximum, k.maxiter))
    floor_factor = 0.0
    head_factor = 0.0
    if T === Float32
        f = FLOAT32_CLOSURES
        if dvclose < f.outer_dvclose || kdv < f.inner_dvclose
            @info "Float32 run: OUTER_DVCLOSE $dvclose and INNER_DVCLOSE $kdv raised to at least $(f.outer_dvclose) and $(f.inner_dvclose) (FLOAT32_CLOSURES)"
        end
        dvclose = max(dvclose, f.outer_dvclose)
        kdv = max(kdv, f.inner_dvclose)
        head_factor = f.dvclose_head_factor
        # The absolute residual criteria (options 0, 1, 2 and the absolute part of 4).
        k.icnvgopt != 3 && (floor_factor = f.inner_rclose_factor)
    end
    krylov = KrylovSettings(k.method, kmax, kdv, krc, k.icnvgopt, k.north, k.scaling)
    return PicardSettings(dvclose, maxiter, krylov, ims, ims.linear_acceleration === :cg, head_factor, floor_factor)
end

"""
    OuterResult

Outcome of the outer iterations of one time step: `converged`, `iterations` (outer),
`inner` (total inner iterations), `hncg` (signed maximum head change of the last outer
iteration) and `lrch` (its reduced node, 0 when no head changed; `sln_get_dxmax`), `dvclose` and `inner_rclose` (the OUTER_DVCLOSE and INNER_RCLOSE used, after
the Float32 floors).
"""
struct OuterResult
    converged::Bool
    iterations::Int
    inner::Int
    hncg::Float64
    lrch::Int
    dvclose::Float64
    inner_rclose::Float64
end

# Signed max |δ| over the active cells (sln_get_dxmax) and max |h'| over all cells.
@inline _dxmax_term(::Type{Ta}, i, δ, ibound, h) where {Ta} =
    @inbounds (ibound[i] > 0 ? Ta(δ[i]) : zero(Ta), Ta(h[i]), zero(Ta), zero(Ta))

@inline _maxscale_term(::Type{Ta}, i, s) where {Ta} = @inbounds (Ta(s[i]), zero(Ta), zero(Ta), zero(Ta))

const _DXMAX_OPS = (_ArgMaxAbsOp(), _MaxAbsOp(), _SumOp(), _SumOp())
const _MAXSCALE_OPS = (_MaxAbsOp(), _SumOp(), _SumOp(), _SumOp())

"""
    PicardSolver

Workspace of the outer iterations of one model: `settings::PicardSettings`, the inner
`KrylovSolver`, the `UnderRelaxation` state, a `Reducer` (Float64 accumulation where
available) and, for the Float32 rclose floor, a per-node scale vector.

    PicardSolver(m::GwfModel, settings; Ts = eltype(m), Tp = Ts)
"""
struct PicardSolver{K <: KrylovSolver, U <: UnderRelaxation, R <: Reducer, V}
    settings::PicardSettings
    krylov::K
    ur::U
    reducer::R
    scale::V
end

function PicardSolver(m::GwfModel{T}, settings::PicardSettings; Ts::Type = T, Tp::Type = Ts) where {T}
    backend = m.backend
    n = Int(m.conn.nodes)
    krylov = KrylovSolver(backend, m.csr, settings.krylov; Ts, Tp)
    ur = UnderRelaxation(backend, T, settings.ims, n)
    red = Reducer(backend, n, accumulation_type(backend, T))
    scale = KernelAbstractions.zeros(backend, T, settings.rclose_floor_factor > 0 ? n : 0)
    return PicardSolver(settings, krylov, ur, red, scale)
end

Adapt.@adapt_structure PicardSolver

# KrylovSettings with a different INNER_RCLOSE (the Float32 floor).
_with_rclose(k::KrylovSettings, rclose) = KrylovSettings(k.method, k.maxiter, k.dvclose, Float64(rclose), k.icnvgopt, k.north, k.scaling)

_with_settings(s::KrylovSolver, st::KrylovSettings) = KrylovSolver(
    st, s.csr, s.x, s.r, s.p, s.q, s.z, s.b0, s.dhat, s.phat, s.qhat, s.t, s.v, s.minv, s.rowfac, s.colfac, s.reducer,
)

"""
    solve_timestep!(ps::PicardSolver, m::GwfModel, kstp) -> OuterResult

Run the outer iterations of the current time step (`sln_ca`; see the header of
src/solver/picard.jl) on the offset heads `m.h`, after `start_timestep!`. `kstp` is the
time step number of the period (for `ims_epfact`).
"""
function solve_timestep!(ps::PicardSolver, m::GwfModel{T}, kstp::Integer) where {T}
    st = ps.settings
    backend = m.backend
    n = Int(m.conn.nodes)
    krylov = ps.krylov
    rclose = st.krylov.rclose
    inner = 0
    hncg = 0.0
    lrch = 0
    dvclose = st.dvclose
    converged = false
    kiter = 0
    for it in 1:st.maxiter
        kiter = it
        floor_step = it == 1 && st.rclose_floor_factor > 0
        formulate!(m; symmetric = st.symmetric, scale = floor_step ? ps.scale : nothing)
        if floor_step
            # Float32: the attainable residual, eps(T)·max Σ|terms| (rule 2 rounding).
            smax = _mapreduce!(ps.reducer, backend, _maxscale_term, _MAXSCALE_OPS, n, ps.scale)[1]
            floor = st.rclose_floor_factor * eps(T) * abs(smax)
            rc = st.krylov.icnvgopt == 2 || st.krylov.icnvgopt == 4 ? floor * sqrt(n) : floor
            rclose = max(st.krylov.rclose, rc)
            krylov = rclose == krylov.settings.rclose ? krylov : _with_settings(krylov, _with_rclose(st.krylov, rclose))
        end
        res = solve!(m.δ, krylov, m.sys.amat, m.res; kstp)
        inner += res.iterations
        (hncg, hmax, _, _), lrch = _mapreduce_loc!(ps.reducer, backend, _dxmax_term, _DXMAX_OPS, n, m.δ, m.ibound, m.h)
        dvclose = max(st.dvclose, st.dvclose_head_factor * eps(T) * abs(hmax))
        if !isfinite(hncg) || !isfinite(hmax)
            # The linear solve diverged (NaN or Inf): the step cannot converge.
            @warn "Outer iteration $it: the correction or the heads are not finite (the linear solver diverged); stopping the outer iterations of this time step"
            converged = false
            break
        end
        converged = res.converged && abs(hncg) <= dvclose
        underrelax!(ps.ur, m.h, m.δ, m.ibound, it, hncg; converged)
        converged && break
    end
    return OuterResult(converged, kiter, inner, hncg, lrch, dvclose, rclose)
end
