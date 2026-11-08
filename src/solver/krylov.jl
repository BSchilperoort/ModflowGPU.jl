# Linear solvers for the correction solve `A·δ = res` (PLAN.md section 3.7, rule 3): PCG
# and BiCGSTAB with a Jacobi preconditioner, IMS scaling and the IMS inner convergence
# tests. Ports, from ImsLinearBase.f90 and ImsLinear.f90:
#
# - `ims_base_cg`: preconditioned CG. MF6 runs it on `A` itself, whose active block is
#   negative definite. We run it on `S·A` with `S = diag(sign(a_nn))`, i.e. on `−A` for
#   the active rows (SPD for a symmetric problem) and on the identity rows of constant
#   heads as they are. In exact arithmetic the iterates are those of MF6 for the
#   correction solve (the constant-head residual is zero); unlike MF6's form it is also
#   PCG proper when some rows have a positive diagonal.
# - `ims_base_bcgs`: BiCGSTAB, right-preconditioned, as in MF6 (no sign change).
# - `ims_base_scale`: SCALING_METHOD DIAGONAL (`a_nm/√|a_nn·a_mm|`) and L2NORM (row
#   then column 2-norms), applied on the fly as row and column factors (the matrix is not
#   modified). As in MF6, CG tests the scaled `δ` and residual and BiCGSTAB the unscaled
#   ones (`tv·DSCALE`, `r/DSCALE`, with DSCALE the row factor also for L2NORM);
#   `L2NORM0` is the scaled initial residual norm in both.
# - `ims_base_testcnvg` and `ims_base_epfact`: inner convergence (INNER_DVCLOSE,
#   INNER_RCLOSE with the options of `rclose_option`), and the stagnation exits of the two
#   solvers (`is_close` on rho, alpha, omega; `rho == 0`, `rho·omega == 0`).
# - `ims_base_residual` for NUMBER_ORTHOGONALIZATIONS (recompute the residual every
#   `north` iterations).
#
# The preconditioner is Jacobi by default, fused into the vector kernels; others
# (`preconditioner = :amg`, src/solver/precond/) are applied with `precond_apply!`. IMS's
# ILU0, MILU0 and ILUT (`resolve_ipc`) map to Jacobi with a warning unless a preconditioner
# is chosen (`KrylovSettings(::ImsSpec; preconditioner)`). REORDERING_METHOD only changes
# the ILU factors in MF6 and is ignored.
#
# Precision: vectors in `Ts`, the preconditioner in `Tp`, and every dot product and norm
# accumulated in `Ta` (Float64 when the backend has it, otherwise `Ts` with pairwise tree
# summation; rule 4) by a two-stage device reduction. The scalars (alpha, beta, omega) and
# the convergence tests are Float64 on the host; one small device-to-host copy per
# reduction (2 per CG iteration, 3 per BiCGSTAB iteration).

# ---------------------------------------------------------------------------------------
# Device reductions

# Work-group size and number of values reduced at once (every reduction carries 4 values;
# unused ones are zero). The tree below is unrolled for _RED_WG = 256, and the kernels
# write `@localmem eltype(partials) (256, 4)` literally: `@localmem` with the element type
# in a kernel-local variable fails at run time on KA 0.10's POCL backend ("undefined
# variable").
const _RED_WG = 256
const _RED_K = 4

struct _SumOp end
struct _MaxAbsOp end
# Like _MaxAbsOp, and the reduction also returns the index of the value (only as the first
# of the 4 ops): the first index of largest magnitude, 0 when every value is zero
# (sln_get_dxmax `lrch`, ims_base_testcnvg `locdv`; for the IMS convergence tables).
struct _ArgMaxAbsOp end

@inline _combine(::_SumOp, a, b) = a + b
# Signed value of largest magnitude (the first one on ties); NaN propagates, so a
# diverged vector is never reported as converged.
@inline _combine(::Union{_MaxAbsOp, _ArgMaxAbsOp}, a, b) = (abs(b) > abs(a) || isnan(b)) ? b : a

# Whether (b, ib) replaces (a, ia) in an argmax: larger magnitude, or equal magnitude at a
# lower index (the tree does not visit the indices in order), or the first NaN. Nonzero
# values always carry their index, zeros carry 0.
@inline _argmax_takes(a, ia, b, ib) = (isnan(b) && !isnan(a)) || abs(b) > abs(a) || (abs(b) == abs(a) && 0 < ib < ia)

const _SUM4 = (_SumOp(), _SumOp(), _SumOp(), _SumOp())

@inline function _tree_step!(shared, sloc, ops, li, s)
    if li <= s
        @inbounds begin
            if ops[1] isa _ArgMaxAbsOp
                if _argmax_takes(shared[li, 1], sloc[li], shared[li + s, 1], sloc[li + s])
                    shared[li, 1] = shared[li + s, 1]
                    sloc[li] = sloc[li + s]
                end
            else
                shared[li, 1] = _combine(ops[1], shared[li, 1], shared[li + s, 1])
            end
            shared[li, 2] = _combine(ops[2], shared[li, 2], shared[li + s, 2])
            shared[li, 3] = _combine(ops[3], shared[li, 3], shared[li + s, 3])
            shared[li, 4] = _combine(ops[4], shared[li, 4], shared[li + s, 4])
        end
    end
    return nothing
end

@inline function _store4!(shared, li, v)
    @inbounds begin
        shared[li, 1] = v[1]
        shared[li, 2] = v[2]
        shared[li, 3] = v[3]
        shared[li, 4] = v[4]
    end
    return nothing
end

@inline _zeros4(::Type{Ta}) where {Ta} = (zero(Ta), zero(Ta), zero(Ta), zero(Ta))

# The tree is written out in each kernel: KA 0.9's CPU backend only accepts `@synchronize`
# in the kernel body itself.

# Stage 1: work item `i` evaluates `f(Ta, i, args...)` (which may also write vectors) and
# returns 4 values in `Ta`; each work group reduces them in local memory (a pairwise tree)
# into one column of `partials` (4 × groups), and for an `_ArgMaxAbsOp` the index of the
# first value into `ploc`.
@kernel function mapreduce_kernel!(partials, ploc, f, ops, len, args)
    Ta = eltype(partials)
    shared = @localmem eltype(partials) (256, 4)
    sloc = @localmem Int32 (256,)
    i = @index(Global, Linear)
    li = @index(Local, Linear)
    vals = i <= len ? f(Ta, i, args...) : _zeros4(Ta)
    _store4!(shared, li, vals)
    if ops[1] isa _ArgMaxAbsOp
        @inbounds sloc[li] = vals[1] != zero(Ta) ? i % Int32 : Int32(0)
    end
    @synchronize
    _tree_step!(shared, sloc, ops, li, 128)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 64)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 32)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 16)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 8)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 4)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 2)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 1)
    @synchronize
    if li == 1
        g = @index(Group, Linear)
        @inbounds for k in 1:_RED_K
            partials[k, g] = shared[1, k]
        end
        if ops[1] isa _ArgMaxAbsOp
            @inbounds ploc[g] = sloc[1]
        end
    end
end

@inline function _fold_partials(partials, ploc, ops, li, ngroups)
    Ta = eltype(partials)
    a1, a2, a3, a4 = _zeros4(Ta)
    l1 = Int32(0)
    g = li
    @inbounds while g <= ngroups
        if ops[1] isa _ArgMaxAbsOp
            if _argmax_takes(a1, l1, partials[1, g], ploc[g])
                a1 = partials[1, g]
                l1 = ploc[g]
            end
        else
            a1 = _combine(ops[1], a1, partials[1, g])
        end
        a2 = _combine(ops[2], a2, partials[2, g])
        a3 = _combine(ops[3], a3, partials[3, g])
        a4 = _combine(ops[4], a4, partials[4, g])
        g += _RED_WG
    end
    return (a1, a2, a3, a4), l1
end

# Stage 2: one work group reduces the columns of `partials` into `out` (4 values), and
# `ploc` into `outloc[1]` for an `_ArgMaxAbsOp`.
@kernel function finalize_reduction_kernel!(out, outloc, @Const(partials), @Const(ploc), ops, ngroups)
    shared = @localmem eltype(partials) (256, 4)
    sloc = @localmem Int32 (256,)
    li = @index(Local, Linear)
    vals, loc = _fold_partials(partials, ploc, ops, li, ngroups)
    _store4!(shared, li, vals)
    if ops[1] isa _ArgMaxAbsOp
        @inbounds sloc[li] = loc
    end
    @synchronize
    _tree_step!(shared, sloc, ops, li, 128)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 64)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 32)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 16)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 8)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 4)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 2)
    @synchronize
    _tree_step!(shared, sloc, ops, li, 1)
    @synchronize
    if li == 1
        @inbounds for k in 1:_RED_K
            out[k] = shared[1, k]
        end
        if ops[1] isa _ArgMaxAbsOp
            @inbounds outloc[1] = sloc[1]
        end
    end
end

"""
    Reducer{Ta}

Device buffers for fused map-reductions of up to 4 values over vectors of up to `len`
entries, accumulated in `Ta` (PLAN.md section 3.7, rule 4): a pairwise tree in each work
group, then one work group over the group results. With an argmax as the first operation
it also returns the index of the first value. `Reducer(backend, len, Ta)`.
"""
struct Reducer{Ta, MA <: AbstractMatrix{Ta}, VA <: AbstractVector{Ta}, VI <: AbstractVector{Int32}, VJ <: AbstractVector{Int32}}
    partials::MA
    out::VA
    ploc::VI
    outloc::VJ
    host::Vector{Ta}
    hostloc::Vector{Int32}
    mapped::Bool
end

# `out` and `outloc` alias `host` and `hostloc` where the backend maps host memory
# (`host_mapped_zeros`), so reading a result needs no device-to-host copy.
function Reducer(backend::Backend, len::Integer, ::Type{Ta}) where {Ta}
    ng = max(cld(len, _RED_WG), 1)
    partials = KernelAbstractions.zeros(backend, Ta, _RED_K, ng)
    ploc = KernelAbstractions.zeros(backend, Int32, ng)
    mo = host_mapped_zeros(backend, Ta, _RED_K)
    ml = host_mapped_zeros(backend, Int32, 1)
    if isnothing(mo) || isnothing(ml)
        out = KernelAbstractions.zeros(backend, Ta, _RED_K)
        outloc = KernelAbstractions.zeros(backend, Int32, 1)
        return Reducer(partials, out, ploc, outloc, zeros(Ta, _RED_K), zeros(Int32, 1), false)
    end
    return Reducer(partials, mo[1], ploc, ml[1], mo[2], ml[2], true)
end

# Adapt: a Reducer is scratch space, so a new one is made on the target backend.
Adapt.adapt_structure(to, r::Reducer{Ta}) where {Ta} =
    Reducer(get_backend(adapt(to, r.partials)), size(r.partials, 2) * _RED_WG, Ta)

"""
    accumulation_type(backend, Ts) -> Type

The type of dot products and norms (PLAN.md section 3.7, rule 4): `Float64` when the
backend supports it, otherwise `Ts` (with pairwise summation).
"""
accumulation_type(backend::Backend, ::Type{Ts}) where {Ts} = supports_float64(backend) ? Float64 : Ts

# Launch `f(Ta, i, args...)` for i in 1:len and reduce its 4 results with `ops`; return
# them on the host as a 4-tuple of Float64 (one synchronisation and one 4-value copy).
function _mapreduce!(red::Reducer{Ta}, backend, f, ops, len, args...) where {Ta}
    _launch_reduction!(red, backend, f, ops, len, args...) || return (0.0, 0.0, 0.0, 0.0)
    red.mapped || copyto!(red.host, red.out)
    h = red.host
    return (Float64(h[1]), Float64(h[2]), Float64(h[3]), Float64(h[4]))
end

# The same with `ops[1] isa _ArgMaxAbsOp`: returns `(values, index)`, the index of the
# first value (0 when it is zero).
function _mapreduce_loc!(red::Reducer, backend, f, ops::Tuple{_ArgMaxAbsOp, Vararg}, len, args...)
    vals = _mapreduce!(red, backend, f, ops, len, args...)
    len > 0 || return vals, 0
    red.mapped || copyto!(red.hostloc, red.outloc)
    return vals, Int(red.hostloc[1])
end

function _launch_reduction!(red::Reducer, backend, f, ops, len, args...)
    ng = cld(len, _RED_WG)
    ng <= size(red.partials, 2) || throw(ArgumentError("reduction over $len entries exceeds the Reducer"))
    ng == 0 && return false
    mapreduce_kernel!(backend, _RED_WG)(red.partials, red.ploc, f, ops, len, args; ndrange = ng * _RED_WG)
    finalize_reduction_kernel!(backend, _RED_WG)(red.out, red.outloc, red.partials, red.ploc, ops, ng; ndrange = _RED_WG)
    KernelAbstractions.synchronize(backend)
    return true
end

@kernel function map_kernel!(f, len, args)
    i = @index(Global, Linear)
    if i <= len
        f(i, args...)
    end
end

_map!(backend, f, len, args...) = (len > 0 && map_kernel!(backend)(f, len, args; ndrange = len); nothing)

@inline _dot_term(::Type{Ta}, i, x, y) where {Ta} = @inbounds (Ta(x[i]) * Ta(y[i]), zero(Ta), zero(Ta), zero(Ta))
@inline _maxabs_term(::Type{Ta}, i, x) where {Ta} = @inbounds (Ta(x[i]), zero(Ta), zero(Ta), zero(Ta))

"""
    device_dot(x, y; Ta = accumulation_type(backend, eltype(x))) -> Float64
    device_maxabs(x; Ta) -> Float64

`x·y` and `max |x|` of device vectors as device reductions accumulated in `Ta`
(convenience functions that allocate a `Reducer`).
"""
function device_dot(x, y; Ta = accumulation_type(get_backend(x), eltype(x)))
    backend = get_backend(x)
    red = Reducer(backend, length(x), Ta)
    return _mapreduce!(red, backend, _dot_term, _SUM4, length(x), x, y)[1]
end

function device_maxabs(x; Ta = accumulation_type(get_backend(x), eltype(x)))
    backend = get_backend(x)
    red = Reducer(backend, length(x), Ta)
    ops = (_MaxAbsOp(), _SumOp(), _SumOp(), _SumOp())
    return abs(_mapreduce!(red, backend, _maxabs_term, ops, length(x), x)[1])
end

"""
    device_argmaxabs(x; Ta) -> (value, index)

The signed value of largest magnitude of a device vector and its first index (0 when `x`
is all zero), as `sln_get_dxmax` finds `hncg` and `lrch`; a device reduction.
"""
function device_argmaxabs(x; Ta = accumulation_type(get_backend(x), eltype(x)))
    backend = get_backend(x)
    red = Reducer(backend, length(x), Ta)
    ops = (_ArgMaxAbsOp(), _SumOp(), _SumOp(), _SumOp())
    vals, loc = _mapreduce_loc!(red, backend, _maxabs_term, ops, length(x), x)
    return vals[1], loc
end

# ---------------------------------------------------------------------------------------
# Settings

"""
    KrylovSettings(; method = :cg, maxiter = 50, dvclose = 1e-3, rclose = 0.1,
                   rclose_option = :infinity_norm, north = 0, scaling = :none,
                   preconditioner = :jacobi, precond_options = (;))
    KrylovSettings(ims::ImsSpec; preconditioner = nothing, precond_options = (;))

Inner solver settings (host): `method` (`:cg` or `:bicgstab`, IMS LINEAR_ACCELERATION),
`maxiter` (INNER_MAXIMUM), `dvclose` (INNER_DVCLOSE), `rclose` (INNER_RCLOSE),
`rclose_option` (`:infinity_norm`, `:strict`, `:l2norm`, `:relative`,
`:l2norm_relative`: `ims_base_testcnvg` options 0–4), `north`
(NUMBER_ORTHOGONALIZATIONS), `scaling` (`:none`, `:diagonal`, `:l2norm`;
SCALING_METHOD), `preconditioner` (`:jacobi`, `:amg`; see `make_preconditioner`) and
`precond_options` (a NamedTuple of keywords for the preconditioner, e.g. `AMGOptions`).

From an `ImsSpec` (with its COMPLEXITY defaults already applied by the loader), IMS's
preconditioner (`resolve_ipc`: ILU0, MILU0 when RELAXATION_FACTOR > 0, ILUT/MILUT when
PRECONDITIONER_LEVELS > 0) maps to Jacobi with a logged warning when `preconditioner` is
`nothing`; otherwise `preconditioner` replaces it (logged). REORDERING_METHOD is ignored
with a logged message. See docs/mf6_notes.md (IMS linear solver mapping).
"""
struct KrylovSettings
    method::Symbol
    maxiter::Int
    dvclose::Float64
    rclose::Float64
    icnvgopt::Int
    north::Int
    scaling::Symbol
    preconditioner::Symbol
    precond_options::NamedTuple
end

const _RCLOSE_OPTIONS = (:infinity_norm, :strict, :l2norm, :relative, :l2norm_relative)

function KrylovSettings(;
        method::Symbol = :cg, maxiter::Integer = 50, dvclose::Real = 1.0e-3, rclose::Real = 0.1,
        rclose_option::Symbol = :infinity_norm, north::Integer = 0, scaling::Symbol = :none,
        preconditioner::Symbol = :jacobi, precond_options::NamedTuple = (;),
    )
    method in (:cg, :bicgstab) || throw(ArgumentError("unknown linear acceleration $method (expected :cg or :bicgstab)"))
    scaling in (:none, :diagonal, :l2norm) || throw(ArgumentError("unknown scaling method $scaling"))
    preconditioner in PRECONDITIONERS || throw(ArgumentError("unknown preconditioner $preconditioner (expected one of $PRECONDITIONERS)"))
    icnvgopt = findfirst(==(rclose_option), _RCLOSE_OPTIONS)
    isnothing(icnvgopt) && throw(ArgumentError("unknown rclose option $rclose_option (expected one of $(_RCLOSE_OPTIONS))"))
    maxiter >= 1 || throw(ArgumentError("inner maximum must be >= 1"))
    north >= 0 || throw(ArgumentError("number of orthogonalizations must be >= 0"))
    # ImsLinear.f90 imslinear_ar: RCLOSE must be > 0 except for RELATIVE_RCLOSE.
    (rclose > 0 || (rclose == 0 && rclose_option === :relative)) ||
        throw(ArgumentError("INNER_RCLOSE must be > 0"))
    return KrylovSettings(
        method, Int(maxiter), Float64(dvclose), Float64(rclose), icnvgopt - 1, Int(north), scaling, preconditioner, precond_options,
    )
end

# A copy of `k` with some fields replaced (keywords as the field names).
_with(k::KrylovSettings; kw...) = KrylovSettings((get(kw, f, getfield(k, f)) for f in fieldnames(KrylovSettings))...)

"""
    ims_preconditioner(levels, relax) -> Symbol

MF6's preconditioner for PRECONDITIONER_LEVELS and RELAXATION_FACTOR
(ImsLinearSettings.f90 `resolve_ipc`): `:ilu0`, `:milu0`, `:ilut` or `:milut`.
"""
ims_preconditioner(levels, relax) = levels > 0 ? (relax > 0 ? :milut : :ilut) : (relax > 0 ? :milu0 : :ilu0)

function KrylovSettings(ims::ImsSpec; preconditioner::Union{Nothing, Symbol} = nothing, precond_options::NamedTuple = (;))
    pc = ims_preconditioner(ims.preconditioner_levels, ims.relaxation_factor)
    if isnothing(preconditioner)
        @warn "IMS preconditioner $(uppercase(String(pc))) is not available until Phase 4; using Jacobi" ims.file
        preconditioner = :jacobi
    else
        @info "IMS preconditioner $(uppercase(String(pc))) replaced by the $preconditioner preconditioner (option `preconditioner`)" ims.file
    end
    ims.reordering_method === :none ||
        @info "IMS REORDERING_METHOD $(uppercase(String(ims.reordering_method))) has no effect with the $preconditioner preconditioner" ims.file
    # As in MF6 this is allowed, but L2NORM scaling makes the scaled matrix nonsymmetric.
    ims.linear_acceleration === :cg && ims.scaling_method === :l2norm &&
        @warn "IMS SCALING_METHOD L2NORM with LINEAR_ACCELERATION CG: the scaled matrix is not symmetric, CG may not converge" ims.file
    return KrylovSettings(;
        method = ims.linear_acceleration, maxiter = ims.inner_maximum, dvclose = ims.inner_dvclose,
        rclose = ims.inner_rclose, rclose_option = ims.rclose_option, north = ims.number_orthogonalizations,
        scaling = ims.scaling_method, preconditioner, precond_options,
    )
end

"""
    ims_epfact(icnvgopt, kstp) -> Float64

Factor applied to the initial L2 norm in the inner convergence test
(ImsLinearBase.f90 `ims_base_epfact`): 0.01 (first time step) or 0.1 for L2NORM_RCLOSE,
1e-4 for L2NORM_RELATIVE_RCLOSE, 1 otherwise. MF6 writes 0.01 and 0.10 as default-real
(single-precision) literals, so the values are `Float64(0.01f0)` and `Float64(0.10f0)`.
"""
function ims_epfact(icnvgopt::Integer, kstp::Integer)
    icnvgopt == 2 && return kstp == 1 ? Float64(0.01f0) : Float64(0.1f0)
    icnvgopt == 4 && return 1.0e-4
    return 1.0
end

"""
    ims_testcnvg(icnvgopt, iiter, dvmax, rmax, rmax0, epfact, dvclose, rclose) -> Int

The IMS inner convergence test (ImsLinearBase.f90 `ims_base_testcnvg`): 1 converged, −1
stop without convergence (STRICT after the first iteration; the relative criteria), 0
continue. `rmax` is the infinity norm of the residual for options 0 and 1 and its L2
norm for options 2–4; `rmax0` is the initial L2 norm.
"""
function ims_testcnvg(icnvgopt, iiter, dvmax, rmax, rmax0, epfact, dvclose, rclose)
    if icnvgopt == 0
        abs(dvmax) <= dvclose && abs(rmax) <= rclose && return 1
    elseif icnvgopt == 1
        abs(dvmax) <= dvclose && abs(rmax) <= rclose && return iiter == 1 ? 1 : -1
    elseif icnvgopt == 2
        (abs(dvmax) <= dvclose || rmax <= rclose) && return 1
        rmax <= rmax0 * epfact && return -1
    elseif icnvgopt == 3
        abs(dvmax) <= dvclose && return 1
        rmax <= rmax0 * rclose && return -1
    elseif icnvgopt == 4
        abs(dvmax) <= dvclose && rmax <= rclose && return 1
        rmax <= rmax0 * epfact && return -1
    end
    return 0
end

# MathUtil.f90 is_close with the defaults (rtol = DSAME = 100·eps, atol = 0, symmetric).
_is_close(a, b) = a == b || abs(a - b) <= 100 * eps(Float64) * max(abs(a), abs(b))

# ---------------------------------------------------------------------------------------
# Solver

"""
    KrylovResult

Outcome of one `solve!`: `converged` (`ims_base_testcnvg` returned 1), `iterations`
(inner iterations of this call), `status` (`:converged`, `:criterion` (stopped by the
STRICT or relative test without convergence), `:maxiter`, `:stagnation` (`is_close`
or zero rho/omega exits)), `dvmax` and `rmax` (`max |δ update|` and the residual
measure of the last iteration, as tested), `l2norm0` (initial residual L2 norm),
`l2norm` (last residual L2 norm). Norms are in the units the IMS test uses (scaled for
CG, see the header of src/solver/krylov.jl).
"""
struct KrylovResult
    converged::Bool
    iterations::Int
    status::Symbol
    dvmax::Float64
    rmax::Float64
    l2norm0::Float64
    l2norm::Float64
end

"""
    KrylovSolver{Ts, Tp, Ta}

Workspace of the inner solver for one CSR structure: `settings::KrylovSettings`, the
structure, Krylov vectors in `Ts`, the Jacobi inverse diagonal in `Tp` (the default
preconditioner, and the fine-level smoother diagonal of others), the row and column
factors of the scaled (and for CG sign-adjusted) operator, a `Reducer` accumulating in
`Ta`, and the preconditioner `precond` (`make_preconditioner(settings.preconditioner, ...)`).

    KrylovSolver(backend, csr::CsrStructure, settings; Ts = Float64, Tp = Ts,
                 Ta = accumulation_type(backend, Ts))
"""
struct KrylovSolver{Ts, Tp, Ta, VS <: AbstractVector{Ts}, VP <: AbstractVector{Tp}, C <: CsrStructure, R <: Reducer{Ta}, PC <: AbstractPreconditioner}
    settings::KrylovSettings
    csr::C
    x::VS
    r::VS
    p::VS
    q::VS
    z::VS
    b0::VS
    dhat::VS
    phat::VS
    qhat::VS
    t::VS
    v::VS
    minv::VP
    rowfac::VS
    colfac::VS
    reducer::R
    precond::PC
end

function KrylovSolver(
        backend::Backend, csr::CsrStructure, settings::KrylovSettings;
        Ts::Type = Float64, Tp::Type = Ts, Ta::Type = accumulation_type(backend, Ts),
    )
    n = Int(csr.nodes)
    z(len) = KernelAbstractions.zeros(backend, Ts, len)
    bc = settings.method === :bicgstab ? n : 0
    cg = settings.method === :cg ? n : 0
    return KrylovSolver(
        settings, csr, z(n), z(n), z(n), z(n), z(cg), z(settings.north > 0 ? n : 0),
        z(bc), z(bc), z(bc), z(bc), z(bc), KernelAbstractions.zeros(backend, Tp, n), z(n), z(n),
        Reducer(backend, n, Ta),
        make_preconditioner(settings.preconditioner, backend, csr, Ts, Tp; settings.precond_options...),
    )
end

Adapt.@adapt_structure KrylovSolver

KernelAbstractions.get_backend(s::KrylovSolver) = get_backend(s.x)

# --- setup: row and column factors and the Jacobi preconditioner (ims_base_scale, pcjac)

# Row factor: the sign of the diagonal for CG (S·A), times the DIAGONAL scaling
# 1/√|a_nn| or the L2NORM row scaling 1/‖a_n·‖₂; column factor for DIAGONAL scaling.
@inline function _setup_row(n, rowfac, colfac, amat, ia, signed::Bool, scaling::Int32)
    Ts = eltype(rowfac)
    @inbounds begin
        d = Ts(amat[ia[n]])
        s = (signed && d < zero(Ts)) ? -one(Ts) : one(Ts)
        c = one(Ts)
        if scaling == Int32(1)
            d != zero(Ts) && (c = one(Ts) / sqrt(abs(d)))
            colfac[n] = c
        elseif scaling == Int32(2)
            ss = zero(Ts)
            for ii in ia[n]:(ia[n + 1] - Int32(1))
                a = Ts(amat[ii])
                ss += a * a
            end
            ss != zero(Ts) && (c = one(Ts) / sqrt(ss))
        else
            colfac[n] = one(Ts)
        end
        rowfac[n] = s * c
    end
    return nothing
end

# L2NORM column scaling: 1/‖column m of the row-scaled matrix‖₂, the column read through
# `isym` (entry (ja[ii], m) of row ja[ii]).
@inline function _setup_l2col(m, colfac, rowfac, amat, ia, ja, isym)
    Ts = eltype(colfac)
    @inbounds begin
        ss = zero(Ts)
        for ii in ia[m]:(ia[m + 1] - Int32(1))
            a = rowfac[ja[ii]] * Ts(amat[isym[ii]])
            ss += a * a
        end
        colfac[m] = ss != zero(Ts) ? one(Ts) / sqrt(ss) : one(Ts)
    end
    return nothing
end

# Jacobi: inverse diagonal of the scaled (sign-adjusted) operator (ims_base_pcjac).
@inline function _setup_jacobi(n, minv, rowfac, colfac, amat, ia)
    Tp = eltype(minv)
    Ts = eltype(rowfac)
    @inbounds begin
        d = rowfac[n] * Ts(amat[ia[n]]) * colfac[n]
        minv[n] = d != zero(Ts) ? Tp(one(Ts) / d) : one(Tp)
    end
    return nothing
end

@inline _precond(minv, x, n) = @inbounds oftype(x, minv[n] * convert(eltype(minv), x))
# Not fused (`precond_apply!` follows): the kernels copy the vector.
@inline _precond(::Nothing, x, n) = x

# `z = M⁻¹·r` with an unfused preconditioner, and `r·z` (one more reduction).
function _apply_precond_dot!(s::KrylovSolver, z, r)
    backend = get_backend(s)
    @timed_section :precond_apply precond_apply!(z, s.precond, r)
    return @timed_section :update_reduce _mapreduce!(s.reducer, backend, _dot_term, _SUM4, Int(s.csr.nodes), r, z)[1]
end

# Row n of the operator: rowfac_n · Σ_m a_nm · colfac_m · p_m (amux on the scaled matrix).
@inline function _op_row(n, amat, ia, ja, rowfac, colfac, p)
    Ts = eltype(p)
    s = zero(Ts)
    @inbounds begin
        for ii in ia[n]:(ia[n + 1] - Int32(1))
            m = ja[ii]
            s += Ts(amat[ii]) * colfac[m] * p[m]
        end
        return rowfac[n] * s
    end
end

# --- PCG (ims_base_cg)

@inline function _cg_init(::Type{Ta}, n, x, r, p, z, b0, res, rowfac, minv) where {Ta}
    Ts = eltype(r)
    @inbounds begin
        rn = rowfac[n] * Ts(res[n])
        r[n] = rn
        x[n] = zero(Ts)
        p[n] = zero(Ts)
        length(b0) > 0 && (b0[n] = rn)
        zn = _precond(minv, rn, n)
        z[n] = zn
    end
    return (Ta(rn) * Ta(rn), Ta(rn) * Ta(zn), zero(Ta), zero(Ta))
end

@inline function _cg_direction(n, p, z, beta)
    @inbounds p[n] = z[n] + beta * p[n]
    return nothing
end

@inline function _apply_dot(::Type{Ta}, n, q, p, w, amat, ia, ja, rowfac, colfac) where {Ta}
    qn = _op_row(n, amat, ia, ja, rowfac, colfac, p)
    @inbounds q[n] = qn
    return (@inbounds(Ta(w[n]) * Ta(qn)), zero(Ta), zero(Ta), zero(Ta))
end

@inline function _cg_update(::Type{Ta}, n, x, r, z, p, q, minv, alpha) where {Ta}
    @inbounds begin
        tv = alpha * p[n]
        x[n] += tv
        rn = r[n] - alpha * q[n]
        r[n] = rn
        zn = _precond(minv, rn, n)
        z[n] = zn
    end
    return (Ta(tv), Ta(rn), Ta(rn) * Ta(rn), Ta(rn) * Ta(zn))
end

@inline function _cg_true_residual(::Type{Ta}, n, r, z, x, b0, amat, ia, ja, rowfac, colfac, minv) where {Ta}
    @inbounds begin
        rn = b0[n] - _op_row(n, amat, ia, ja, rowfac, colfac, x)
        r[n] = rn
        zn = _precond(minv, rn, n)
        z[n] = zn
    end
    return (Ta(rn) * Ta(zn), zero(Ta), zero(Ta), zero(Ta))
end

# --- BiCGSTAB (ims_base_bcgs)

@inline function _bcgs_init(::Type{Ta}, n, x, r, dhat, p, v, b0, res, rowfac) where {Ta}
    Ts = eltype(r)
    @inbounds begin
        rn = rowfac[n] * Ts(res[n])
        r[n] = rn
        dhat[n] = rn
        x[n] = zero(Ts)
        p[n] = zero(Ts)
        v[n] = zero(Ts)
        length(b0) > 0 && (b0[n] = rn)
    end
    return (Ta(rn) * Ta(rn), Ta(rn) * Ta(rn), zero(Ta), zero(Ta))
end

@inline function _bcgs_direction(n, p, phat, r, v, minv, beta, omega0)
    @inbounds begin
        pn = r[n] + beta * (p[n] - omega0 * v[n])
        p[n] = pn
        phat[n] = _precond(minv, pn, n)
    end
    return nothing
end

@inline function _bcgs_half(n, q, qhat, r, v, minv, alpha)
    @inbounds begin
        qn = r[n] - alpha * v[n]
        q[n] = qn
        qhat[n] = _precond(minv, qn, n)
    end
    return nothing
end

@inline function _bcgs_apply_t(::Type{Ta}, n, t, qhat, q, amat, ia, ja, rowfac, colfac) where {Ta}
    tn = _op_row(n, amat, ia, ja, rowfac, colfac, qhat)
    @inbounds t[n] = tn
    return (@inbounds(Ta(tn) * Ta(q[n])), Ta(tn) * Ta(tn), zero(Ta), zero(Ta))
end

@inline function _bcgs_update(::Type{Ta}, n, x, r, dhat, phat, qhat, q, t, rowfac, alpha, omega) where {Ta}
    @inbounds begin
        tv = alpha * phat[n] + omega * qhat[n]
        x[n] += tv
        rn = q[n] - omega * t[n]
        r[n] = rn
        # Unscaled values for the convergence test (ims_base_bcgs: tv·DSCALE, r/DSCALE).
        dv = tv * rowfac[n]
        ru = rn / rowfac[n]
        return (Ta(dv), Ta(ru), Ta(ru) * Ta(ru), Ta(dhat[n]) * Ta(rn))
    end
end

@inline function _bcgs_true_residual(::Type{Ta}, n, r, dhat, x, b0, amat, ia, ja, rowfac, colfac) where {Ta}
    @inbounds begin
        rn = b0[n] - _op_row(n, amat, ia, ja, rowfac, colfac, x)
        r[n] = rn
        return (Ta(dhat[n]) * Ta(rn), zero(Ta), zero(Ta), zero(Ta))
    end
end

@inline function _unscale(n, δ, x, colfac)
    @inbounds δ[n] = convert(eltype(δ), colfac[n] * x[n])
    return nothing
end

const _CG_UPDATE_OPS = (_MaxAbsOp(), _MaxAbsOp(), _SumOp(), _SumOp())

# `num / d` with a guarded denominator. MF6 adds the absolute SIGN(DPREC, d) (ims_base_cg,
# ims_base_bcgs), which is not scale invariant: in the correction solves the dot products
# fall to |r|^2 ~ 1e-16 well before INNER_RCLOSE, and the guard then stalls BiCGSTAB and CG
# (ex-gwf-hanir :double: 16 outer / 14402 inner iterations instead of 4 / 222). The guard
# here is relative to the numerator, so |num/d| <= 1/eps as in MF6 for O(1) values.
_safe_div(num, d) = num / (d + copysign(max(eps(Float64) * abs(num), floatmin(Float64)), d))

"""
    solve!(δ, solver::KrylovSolver, amat, res; kstp = 1) -> KrylovResult

Solve the correction system `A·δ = res` (PLAN.md section 3.7, rule 3) with a zero
initial guess: `amat` is MF6's matrix in the CSR layout of `solver.csr` (a
`LinearSystem`'s `amat`, any float type), `res` the flux-form residual (`flux_residual!`),
`δ` the output. Scaling and the Jacobi preconditioner are set up from `amat` on every call
(as `imslinear_ap` does). `kstp` (time step number) only matters for
`rclose_option = :l2norm` (`ims_base_epfact`). Returns a `KrylovResult` with the
iteration count of this call.
"""
function solve!(δ, solver::KrylovSolver{Ts}, amat, res; kstp::Integer = 1) where {Ts}
    st = solver.settings
    csr = solver.csr
    backend = get_backend(solver)
    n = Int(csr.nodes)
    cg = st.method === :cg
    scaling = Int32(st.scaling === :diagonal ? 1 : st.scaling === :l2norm ? 2 : 0)
    @timed_section :precond_setup begin
        _map!(backend, _setup_row, n, solver.rowfac, solver.colfac, amat, csr.ia, cg, scaling)
        scaling == 2 && _map!(backend, _setup_l2col, n, solver.colfac, solver.rowfac, amat, csr.ia, csr.ja, csr.isym)
        _map!(backend, _setup_jacobi, n, solver.minv, solver.rowfac, solver.colfac, amat, csr.ia)
        precond_setup!(solver.precond, solver, amat)
    end
    result = cg ? _solve_cg!(solver, amat, kstp, res) : _solve_bcgs!(solver, amat, kstp, res)
    _precond_record!(solver.precond, result)
    if !result.converged && _precond_retry!(solver.precond)
        # A preconditioner set up from an earlier matrix (AMG hierarchy reuse): rebuild it
        # and solve again from zero; the iterations of both attempts count.
        @timed_section :precond_setup precond_setup!(solver.precond, solver, amat)
        first = result
        result = cg ? _solve_cg!(solver, amat, kstp, res) : _solve_bcgs!(solver, amat, kstp, res)
        _precond_record!(solver.precond, result)
        result = KrylovResult(
            result.converged, first.iterations + result.iterations, result.status, result.dvmax, result.rmax,
            result.l2norm0, result.l2norm,
        )
    end
    @timed_section :vector_update _map!(backend, _unscale, n, δ, solver.x, solver.colfac)
    return result
end

function _status(icnvg, iters, maxiter)
    icnvg == 1 && return :converged
    icnvg == -1 && return :criterion
    return iters >= maxiter ? :maxiter : :stagnation
end

function _solve_cg!(s::KrylovSolver{Ts}, amat, kstp, res) where {Ts}
    st = s.settings
    csr = s.csr
    backend = get_backend(s)
    red = s.reducer
    n = Int(csr.nodes)
    epfact = ims_epfact(st.icnvgopt, kstp)
    fused = s.precond isa JacobiPreconditioner
    minv = fused ? s.minv : nothing
    rr, rho, _, _ = @timed_section :update_reduce _mapreduce!(red, backend, _cg_init, _SUM4, n, s.x, s.r, s.p, s.z, s.b0, res, s.rowfac, minv)
    fused || rr == 0 || (rho = _apply_precond_dot!(s, s.z, s.r))
    l2norm0 = sqrt(rr)
    icnvg = 0
    itmax = st.maxiter
    if l2norm0 == 0
        itmax = 0
        icnvg = 1
    end
    rho0 = 0.0
    innerit = 0
    deltax = rmax = 0.0
    l2norm = l2norm0
    for iiter in 1:itmax
        innerit += 1
        beta = iiter == 1 ? 0.0 : rho / rho0
        @timed_section :vector_update _map!(backend, _cg_direction, n, s.p, s.z, Ts(beta))
        pq = @timed_section :spmv_dot _mapreduce!(red, backend, _apply_dot, _SUM4, n, s.q, s.p, s.p, amat, csr.ia, csr.ja, s.rowfac, s.colfac)[1]
        alpha = _safe_div(rho, pq)
        deltax, rmax, l2sq, rho_next = @timed_section :update_reduce _mapreduce!(
            red, backend, _cg_update, _CG_UPDATE_OPS, n, s.x, s.r, s.z, s.p, s.q, minv, Ts(alpha),
        )
        l2norm = sqrt(l2sq)
        rcnvg = st.icnvgopt >= 2 ? l2norm : rmax
        icnvg = ims_testcnvg(st.icnvgopt, innerit, deltax, rcnvg, l2norm0, epfact, st.dvclose, st.rclose)
        rcnvg == 0 && (icnvg = 1)
        icnvg != 0 && break
        _is_close(rho, rho0) && break
        if st.north > 0 && mod(iiter + 1, st.north) == 0
            rho_next = @timed_section :true_residual _mapreduce!(
                red, backend, _cg_true_residual, _SUM4, n, s.r, s.z, s.x, s.b0, amat, csr.ia, csr.ja,
                s.rowfac, s.colfac, minv,
            )[1]
        end
        fused || (rho_next = _apply_precond_dot!(s, s.z, s.r))
        rho == 0 && break
        rho0 = rho
        rho = rho_next
    end
    return KrylovResult(icnvg == 1, innerit, _status(icnvg, innerit, st.maxiter), abs(deltax), abs(rmax), l2norm0, l2norm)
end

function _solve_bcgs!(s::KrylovSolver{Ts}, amat, kstp, res) where {Ts}
    st = s.settings
    csr = s.csr
    backend = get_backend(s)
    red = s.reducer
    n = Int(csr.nodes)
    epfact = ims_epfact(st.icnvgopt, kstp)
    rr, rho, _, _ = @timed_section :update_reduce _mapreduce!(red, backend, _bcgs_init, _SUM4, n, s.x, s.r, s.dhat, s.p, s.v, s.b0, res, s.rowfac)
    fused = s.precond isa JacobiPreconditioner
    minv = fused ? s.minv : nothing
    l2norm0 = sqrt(rr)
    icnvg = 0
    itmax = st.maxiter
    if l2norm0 == 0
        itmax = 0
        icnvg = 1
    end
    rho0 = alpha0 = omega0 = 0.0
    innerit = 0
    deltax = rmax = 0.0
    l2norm = l2norm0
    for iiter in 1:itmax
        innerit += 1
        beta = iiter == 1 ? 0.0 : (rho / rho0) * (alpha0 / omega0)
        @timed_section :vector_update _map!(backend, _bcgs_direction, n, s.p, s.phat, s.r, s.v, minv, Ts(beta), Ts(omega0))
        fused || @timed_section :precond_apply precond_apply!(s.phat, s.precond, s.p)
        dv = @timed_section :spmv_dot _mapreduce!(red, backend, _apply_dot, _SUM4, n, s.v, s.phat, s.dhat, amat, csr.ia, csr.ja, s.rowfac, s.colfac)[1]
        alpha = _safe_div(rho, dv)
        @timed_section :vector_update _map!(backend, _bcgs_half, n, s.q, s.qhat, s.r, s.v, minv, Ts(alpha))
        fused || @timed_section :precond_apply precond_apply!(s.qhat, s.precond, s.q)
        tq, tt, _, _ = @timed_section :spmv_dot _mapreduce!(red, backend, _bcgs_apply_t, _SUM4, n, s.t, s.qhat, s.q, amat, csr.ia, csr.ja, s.rowfac, s.colfac)
        omega = _safe_div(tq, tt)
        deltax, rmax, l2sq, rho_next = @timed_section :update_reduce _mapreduce!(
            red, backend, _bcgs_update, _CG_UPDATE_OPS, n, s.x, s.r, s.dhat, s.phat, s.qhat, s.q, s.t, s.rowfac,
            Ts(alpha), Ts(omega),
        )
        l2norm = sqrt(l2sq)
        rcnvg = st.icnvgopt >= 2 ? l2norm : rmax
        icnvg = ims_testcnvg(st.icnvgopt, innerit, deltax, rcnvg, l2norm0, epfact, st.dvclose, st.rclose)
        rcnvg == 0 && (icnvg = 1)
        icnvg != 0 && break
        (_is_close(rho, rho0) || _is_close(alpha, alpha0) || _is_close(omega, omega0)) && break
        if st.north > 0 && mod(iiter + 1, st.north) == 0
            rho_next = @timed_section :true_residual _mapreduce!(
                red, backend, _bcgs_true_residual, _SUM4, n, s.r, s.dhat, s.x, s.b0, amat, csr.ia, csr.ja,
                s.rowfac, s.colfac,
            )[1]
        end
        rho * omega == 0 && break
        rho0 = rho
        alpha0 = alpha
        omega0 = omega
        rho = rho_next
    end
    return KrylovResult(icnvg == 1, innerit, _status(icnvg, innerit, st.maxiter), abs(deltax), abs(rmax), l2norm0, l2norm)
end
