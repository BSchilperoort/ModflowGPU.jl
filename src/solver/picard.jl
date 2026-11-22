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
# With BACKTRACKING_NUMBER > 0, step 1 is followed by `sln_backtracking`
# (src/solver/backtracking.jl, T6.2a), which may reduce the previous head change and
# reformulate before the solve.
# The loop stops at convergence or after OUTER_MAXIMUM iterations.
#
# Newton–Raphson (T6.2, the model's NEWTON option): step 1 formulates with the Newton terms
# (`sln_buildsystem(kiter, inewton = 1)`); with backtracking it is preceded by the
# formulation without them (`inewton = 0`), on which `sln_backtracking` works. In a
# steady-state period the pseudo-transient continuation term is then added to the
# diagonal (`apply_ptc!`), and with NEWTON UNDER_RELAXATION step 4 ends with `npf_nur`
# (`newton_ur!`; src/solver/newton.jl). The linear solver must be BiCGSTAB (the matrix is
# not symmetric; CG is MF6's input error, checked by `initialize`).
#
# Not implemented: package convergence checks (none of the supported packages has one).

"""
    FLOAT32_CLOSURES

The tightest closures `:single` (Float32) runs use (PLAN.md section 3.7, "Closures in
Float32"; measured in docs/dev/precision.md): IMS closures below these are raised to them in
Float32 by `PicardSettings`:

- `outer_dvclose`, `inner_dvclose`: fixed floors (m);
- `dvclose_head_factor`: OUTER_DVCLOSE is also at least this factor times
  `eps(Float32)·max_n |h'_n|`, the resolution of the offset heads (checked every outer
  iteration; e.g. 1.2e-4 m on ex-gwf-toth, whose offsets reach 1010 m). Once a
  correction of the step meets OUTER_DVCLOSE but its inner solve did not converge, the
  next inner solves of the step use INNER_DVCLOSE >= that head resolution of the previous
  outer iteration (never more): the first inner correction of a converged Float32 state is
  rounding noise of about `eps32·max|h'|`, which a fixed 1e-6 m INNER_DVCLOSE rejects
  for ever (T5.3, T6.0; see `outer_iteration!`);
- `dvclose_head_cap` (T6.2): that head resolution is used up to this value (m; offsets up
  to 84 km). Heads of a diverging iteration (ex-gwf-nwt-p03b `:single`: 1.6e27 m after an
  unconverged inner solve) would otherwise raise OUTER_DVCLOSE with them and end the step as
  converged;
- `inner_rclose_factor`: an absolute INNER_RCLOSE is at least this factor times
  `eps(Float32)·max_n Σ_m |terms|` of the first outer iteration of a step, the rounding
  level of the flux-form residual (rule 2).
"""
const FLOAT32_CLOSURES = (;
    outer_dvclose = 1.0e-6, inner_dvclose = 1.0e-6, dvclose_head_factor = 1.0, dvclose_head_cap = 1.0e-2, inner_rclose_factor = 1.0,
)

"""
    PicardSettings

Outer-iteration settings of one solution (host, Float64): `dvclose` (OUTER_DVCLOSE),
`maxiter` (OUTER_MAXIMUM), `krylov::KrylovSettings` (inner solver), `ims::ImsSpec` (for the
under-relaxation), `symmetric` (CG: the constant-head columns go to the right-hand side,
`sln_ls`), and the Float32 floors `dvclose_head_factor` and `rclose_floor_factor` (see
`FLOAT32_CLOSURES`; 0 for Float64).

    PicardSettings(ims::ImsSpec, T; closures = nothing, preconditioner = :auto, precond_options = (;))

Takes OUTER_DVCLOSE, OUTER_MAXIMUM and the inner settings from `ims` (`preconditioner`
and `precond_options` as in `KrylovSettings(::ImsSpec)`). `closures` (a
NamedTuple with any of `outer_dvclose`, `outer_maximum`, `inner_dvclose`, `inner_rclose`,
`inner_maximum`) overrides them. For `T = Float32`, closures below `FLOAT32_CLOSURES` are
raised to them (logged), since Float32 cannot reach MF6's tight closures (PLAN.md 3.7).
Backtracking (BACKTRACKING_* of `ims`, T6.2a) is applied when BACKTRACKING_NUMBER > 0.
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

function PicardSettings(
        ims::ImsSpec, ::Type{T}; closures = nothing, preconditioner::Symbol = :auto,
        precond_options::NamedTuple = (;),
    ) where {T <: AbstractFloat}
    get_c(k, default) = (isnothing(closures) || !haskey(closures, k)) ? default : getfield(closures, k)
    ims.under_relaxation in (:none, :simple, :cooley, :dbd) || throw(ArgumentError("unknown IMS UNDER_RELAXATION $(ims.under_relaxation)"))
    dvclose = Float64(get_c(:outer_dvclose, ims.outer_dvclose))
    maxiter = Int(get_c(:outer_maximum, ims.outer_maximum))
    k = KrylovSettings(ims; preconditioner, precond_options)
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
    krylov = _with(k; maxiter = kmax, dvclose = kdv, rclose = krc)
    return PicardSettings(dvclose, maxiter, krylov, ims, ims.linear_acceleration === :cg, head_factor, floor_factor)
end

"""
    OuterResult

Outcome of the outer iterations of one time step: `converged`, `iterations` (outer),
`inner` (total inner iterations), `hncg` (signed maximum head change of the last outer
iteration) and `lrch` (its reduced node, 0 when no head changed; `sln_get_dxmax`), `dvclose` and `inner_rclose` (the OUTER_DVCLOSE and INNER_RCLOSE used, after
the Float32 floors), and `backtracking` (one `BacktrackRecord` per outer iteration with
BACKTRACKING_NUMBER > 0, empty otherwise; T6.2a).
"""
struct OuterResult
    converged::Bool
    iterations::Int
    inner::Int
    hncg::Float64
    lrch::Int
    dvclose::Float64
    inner_rclose::Float64
    backtracking::Vector{BacktrackRecord}
end

"""
    backtrack_count(r::OuterResult) -> Int

The number of outer iterations of the step in which backtracking reduced the head change
(MF6's outer table rows "Backtracking" with BACKTRACK ITERATIONS > 0).
"""
backtrack_count(r::OuterResult) = count(b -> b.count > 0, r.backtracking)

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
available), for the Float32 rclose floor a per-node scale vector, and for backtracking
(BACKTRACKING_NUMBER > 0, or NEWTON UNDER_RELAXATION; else empty) `hprev`, the offset heads
at the start of the last outer iteration's update (MF6's `xtemp`), and `ptc`, the
pseudo-transient continuation state of the Newton formulation (`PtcState`, T6.2).

    PicardSolver(m::GwfModel, settings; Ts = eltype(m), Tp = Ts)
"""
struct PicardSolver{K <: KrylovSolver, U <: UnderRelaxation, R <: Reducer, V}
    settings::PicardSettings
    krylov::K
    ur::U
    reducer::R
    scale::V
    hprev::V
    ptc::PtcState
end

function PicardSolver(m::GwfModel{T}, settings::PicardSettings; Ts::Type = T, Tp::Type = Ts) where {T}
    backend = m.backend
    n = Int(m.conn.nodes)
    colors = if !(settings.krylov.preconditioner in (:mc_sgs, :mc_ilu0))
        nothing
    elseif m.scsr === m.csr && !isnothing(m.solution)
        solution_colors(m.solution)
    elseif m.scsr === m.csr
        grid_colors(m.conn, m.spec.dis)
    else
        # T6.5: the extended XT3D stencil (neighbours of neighbours) is not two-colourable.
        greedy_colors(Array(m.scsr.ia), Array(m.scsr.ja))
    end
    krylov = KrylovSolver(backend, m.scsr, settings.krylov; Ts, Tp, colors)
    ur = UnderRelaxation(backend, T, settings.ims, n)
    red = Reducer(backend, n, accumulation_type(backend, T))
    scale = KernelAbstractions.zeros(backend, T, settings.rclose_floor_factor > 0 ? n : 0)
    hprev = KernelAbstractions.zeros(backend, T, (settings.ims.backtracking_number > 0 || !isempty(m.nurbot)) ? n : 0)
    return PicardSolver(settings, krylov, ur, red, scale, hprev, PtcState())
end

Adapt.@adapt_structure PicardSolver

# KrylovSettings with a different INNER_RCLOSE (the Float32 floor).
_with_rclose(k::KrylovSettings, rclose) = _with(k; rclose = Float64(rclose))

_with_settings(s::KrylovSolver, st::KrylovSettings) = KrylovSolver(
    st, s.csr, s.x, s.r, s.p, s.q, s.z, s.b0, s.dhat, s.phat, s.qhat, s.t, s.v, s.minv, s.rowfac, s.colfac, s.reducer, s.precond,
    s.reducer2, s.reducer3, s.sa, s.ss,
)

"""
    OuterState

The state of the outer iterations of the current time step, for one iteration at a time
(`outer_iteration!`, the stepping API's `solve`): the Krylov solver in use (with the Float32
rclose floor), the iteration count `kiter`, total inner iterations, the last `hncg`/`lrch`,
`dvclose`, `rclose`, `converged`, `diverged` (the correction or the heads were not
finite; no further iteration is done), `dvfloor` (Float32: `eps(T)·max|h'|` of the last
outer iteration; 0 before the first), `noise_floor` (Float32: INNER_DVCLOSE is at least
`dvfloor` for the rest of the step, see `outer_iteration!`), and for backtracking
`res_prev` (the outgoing residual norm of the last outer iteration) and `backtracking`
(the `BacktrackRecord`s of the step). T6.17: `trace`, the `InnerTrace` the inner solves
record into (IMS CSV_INNER_OUTPUT), or `nothing`.

    OuterState(ps::PicardSolver; trace = nothing)
"""
mutable struct OuterState{K <: KrylovSolver, TR}
    krylov::K
    kiter::Int
    inner::Int
    hncg::Float64
    lrch::Int
    dvclose::Float64
    rclose::Float64
    converged::Bool
    diverged::Bool
    dvfloor::Float64
    noise_floor::Bool
    res_prev::Float64
    backtracking::Vector{BacktrackRecord}
    const trace::TR
end

OuterState(ps::PicardSolver; trace = nothing) = OuterState(
    ps.krylov, 0, 0, 0.0, 0, ps.settings.dvclose, ps.settings.krylov.rclose, false, false, 0.0, false, 0.0,
    BacktrackRecord[], trace,
)

OuterResult(s::OuterState) = OuterResult(s.converged, s.kiter, s.inner, s.hncg, s.lrch, s.dvclose, s.rclose, copy(s.backtracking))

"""
    outer_iteration!(state::OuterState, ps::PicardSolver, m::GwfModel, kstp) -> Bool

One outer iteration of the current time step (one pass of the `sln_ca` loop: formulate and
assemble, `sln_backtracking` when BACKTRACKING_NUMBER > 0, the correction solve,
`sln_get_dxmax`, the convergence check and `sln_underrelax`), after `start_timestep!` and a fresh `OuterState`. Returns whether the
step has converged. After a non-finite correction (`state.diverged`) it returns `false`
without iterating.
"""
function outer_iteration!(state::OuterState, ps::PicardSolver, m::GwfModel{T}, kstp::Integer) where {T}
    state.diverged && return false
    st = ps.settings
    backend = m.backend
    n = Int(m.conn.nodes)
    it = state.kiter += 1
    # Float32 rclose floor: from the first formulation; with convertible cells (T6.1) the
    # terms change with the saturations, so it is the maximum over the step's iterations.
    # T6.5: also with XT3D, whose terms per face cancel and grow with the head differences
    # (a model starting from flat heads has none in the first formulation).
    floor_step = (it == 1 || !isnothing(m.npf.cv) || !isnothing(m.xt3d)) && st.rclose_floor_factor > 0
    # Wetting and drying of convertible cells (T6.1; `npf_cf` → `sgwf_npf_wetdry`), then
    # the formulation (the saturations, `npf_cf!`, are computed in `formulate!`).
    scale = floor_step ? ps.scale : nothing
    isnothing(m.wd) || @timed_section :assembly wetdry!(m, it)
    bt = st.ims.backtracking_number > 0
    # Newton (T6.2): backtracking works on the formulation without the Newton terms
    # (`sln_buildsystem(kiter, inewton = 0)`), then the system of the solve is formulated
    # with them; otherwise the first formulation is the one of the solve.
    @timed_section :assembly formulate!(m; symmetric = st.symmetric, terms = false, scale, jac = m.newton && !bt)
    bt && _backtracking!(state, ps, m, it, scale)
    m.newton && bt && @timed_section :assembly formulate!(m; symmetric = st.symmetric, terms = false, scale)
    # Pseudo-transient continuation of a steady-state Newton period (T6.2).
    if m.newton && (isnothing(m.sto) || !m.sto.transient) && ptc_allowed(st.ims.no_ptc, m.kper)
        @timed_section :assembly apply_ptc!(ps.ptc, ps.reducer, m, it, kstp)
    end
    if floor_step
        # Float32: the attainable residual, eps(T)·max Σ|terms| (rule 2 rounding).
        smax = @timed_section :outer_reduction _mapreduce!(ps.reducer, backend, _maxscale_term, _MAXSCALE_OPS, n, ps.scale)[1]
        floor = st.rclose_floor_factor * eps(T) * abs(smax)
        rc = st.krylov.icnvgopt == 2 || st.krylov.icnvgopt == 4 ? floor * sqrt(n) : floor
        state.rclose = max(st.krylov.rclose, rc, it == 1 ? 0.0 : state.rclose)
        k = state.krylov
        state.krylov = state.rclose == k.settings.rclose ? k : _with_settings(k, _with_rclose(st.krylov, state.rclose))
    end
    if st.dvclose_head_factor > 0 && it > 1 && abs(state.hncg) <= state.dvclose
        # Float32: the last correction met OUTER_DVCLOSE, so only the inner test kept the
        # step from converging. Such a correction is rounding noise of about eps32·max|h'|
        # (the residual of a converged Float32 state), and an inner solve from it cannot get
        # its per-iteration change below a fixed INNER_DVCLOSE under that size: it stalls at
        # INNER_MAXIMUM (T5.3, Indus: 1.45e-6 m with offsets up to 62 m), or with STRICT
        # (ImsLinearBase.f90 `ims_base_testcnvg`: converged only in the first inner
        # iteration) never converges (T6.0, ex-gwf-lgrv-gc: 2–3e-6 m, offsets up to 49 m).
        # From here to the end of the step INNER_DVCLOSE is at least that noise level,
        # `dvfloor` = eps32·max|h'| of the last outer iteration, and never above it: a loose
        # OUTER_DVCLOSE does not loosen the inner test. Not applied before the trigger: a
        # head-resolution INNER_DVCLOSE from the first outer iteration on costs accuracy
        # (ex-gwf-toth, offsets 1010 m: 1.4e-3 m instead of 6.6e-5 m, T5.3 and T6.0).
        state.noise_floor = true
    end
    if state.noise_floor
        kdv = max(st.krylov.dvclose, state.dvfloor)
        k = state.krylov
        kdv == k.settings.dvclose || (state.krylov = _with_settings(k, _with(k.settings; dvclose = kdv)))
    end
    res = @timed_section :linear_solve solve!(m.δ, state.krylov, m.sys.amat, m.res; kstp, diag = m.sys.diag, offdiag_id = m.offdiag_id, newton = m.newton, trace = state.trace)
    state.inner += res.iterations
    (hncg, hmax, _, _), lrch = @timed_section :outer_reduction _mapreduce_loc!(ps.reducer, backend, _dxmax_term, _DXMAX_OPS, n, m.δ, m.ibound, m.h)
    state.hncg, state.lrch = hncg, lrch
    state.dvfloor = min(st.dvclose_head_factor * eps(T) * abs(hmax), FLOAT32_CLOSURES.dvclose_head_cap)
    state.dvclose = max(st.dvclose, state.dvfloor)
    if !isfinite(hncg) || !isfinite(hmax)
        # The linear solve diverged (NaN or Inf): the step cannot converge.
        @warn "Outer iteration $it: the correction or the heads are not finite (the linear solver diverged); stopping the outer iterations of this time step"
        state.converged = false
        state.diverged = true
        return false
    end
    state.converged = res.converged && abs(hncg) <= state.dvclose
    # MF6's `xtemp` (the heads before this iteration's update) for the next backtracking
    # and for NEWTON UNDER_RELAXATION.
    nur = !isempty(m.nurbot)
    (bt || nur) && !state.converged && save_hprev!(ps.hprev, m.h, m.ibound)
    @timed_section :underrelax underrelax!(ps.ur, m.h, m.δ, m.ibound, it, hncg; converged = state.converged)
    if nur && !state.converged
        # npf_nur, sln_nur_has_converged (T6.2).
        adjusted, dxold, hncg_nur, lrch_nur = @timed_section :underrelax newton_ur!(ps.reducer, m, ps.hprev)
        if adjusted && dxold <= state.dvclose && abs(hncg) <= state.dvclose
            state.converged = true
            state.hncg, state.lrch = hncg_nur, lrch_nur
        end
    end
    return state.converged
end

# sln_backtracking at the start of outer iteration `it`, after the formulation at the
# current heads. Every MF6 `sln_buildsystem` runs `npf_cf` and so the wetting and drying
# (T6.1): once for each backtracking formulation, and again for the formulation of the
# solve, which here is the last backtracking one unless that second `wetdry!` changed
# IBOUND (only models with convertible cells have `m.wd`).
function _backtracking!(state::OuterState, ps::PicardSolver, m::GwfModel, it::Integer, scale)
    st = ps.settings
    ims = st.ims
    reformulate() = begin
        isnothing(m.wd) || @timed_section :assembly wetdry!(m, it)
        @timed_section :assembly formulate!(m; symmetric = st.symmetric, terms = false, scale, jac = false)
    end
    rec, state.res_prev = backtrack!(
        reformulate, ps.reducer, m.h, ps.hprev, m.res, m.ibound, it, state.res_prev;
        numtrack = ims.backtracking_number, btol = ims.backtracking_tolerance, breduc = ims.backtracking_reduction_factor,
        res_lim = ims.backtracking_residual_limit, dvclose = state.dvclose,
    )
    push!(state.backtracking, rec)
    if !isnothing(m.wd) && @timed_section(:assembly, wetdry!(m, it))
        @timed_section :assembly formulate!(m; symmetric = st.symmetric, terms = false, scale)
    end
    return nothing
end

"""
    solve_timestep!(ps::PicardSolver, m::GwfModel, kstp) -> OuterResult

Run the outer iterations of the current time step (`sln_ca`; see the header of
src/solver/picard.jl) on the offset heads `m.h`, after `start_timestep!`: up to
OUTER_MAXIMUM calls of `outer_iteration!`. `kstp` is the time step number of the period
(for `ims_epfact`).
"""
function solve_timestep!(ps::PicardSolver, m::GwfModel, kstp::Integer)
    state = OuterState(ps)
    for _ in 1:ps.settings.maxiter
        (outer_iteration!(state, ps, m, kstp) || state.diverged) && break
    end
    return OuterResult(state)
end
