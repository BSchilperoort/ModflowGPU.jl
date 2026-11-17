# Preconditioner interface of the Krylov solvers (src/solver/krylov.jl; PLAN.md section
# 3.5, T4.2/T4.3).
#
# A preconditioner works on the operator the Krylov solver iterates on, the scaled (and for
# CG sign-adjusted) matrix `Â = R·A·C` with `R = diag(rowfac)` and `C = diag(colfac)` (see
# the header of src/solver/krylov.jl). Two generic functions:
#
# - `precond_setup!(P, solver, amat)`: once per `solve!` (once per outer iteration), after
#   the row and column factors are set up, from the assembled matrix `amat` (MF6 layout,
#   engine precision `T`);
# - `precond_apply!(z, P, r)`: `z = M⁻¹·r` for whole device vectors in `Ts`, the
#   preconditioner's own data in `Tp`. For CG, `M⁻¹` must be symmetric positive definite.
#
# `JacobiPreconditioner` is the default: CG and BiCGSTAB fuse its application into their
# update kernels (`_precond` in krylov.jl) and only call `precond_setup!` for it; any other
# preconditioner is applied with `precond_apply!` (non-fused kernels). Selection:
# `KrylovSettings(; preconditioner, precond_options)`, from `ModflowGPU.run` / `initialize`
# (`preconditioner = :jacobi`, `:chebyshev`, `:mc_sgs`, `:mc_ilu0`, `:amg`), built by
# `build_preconditioner` with `precond_options` as the keywords of its constructor.
#
# Optional hooks (defaults below): `_precond_apply_dot!` (fuse `r·z` into the last kernel
# of the application), `_precond_record!` and `_precond_retry!` (after each solve; AMG's
# hierarchy reuse and rebuild).

"""
    AbstractPreconditioner

Supertype of the preconditioners of `KrylovSolver` (src/solver/precond/precond.jl). Methods:
`precond_setup!(P, solver, amat)` and `precond_apply!(z, P, r)`.
"""
abstract type AbstractPreconditioner end

"""
    PRECONDITIONERS

The preconditioners `KrylovSettings(; preconditioner)` accepts: `:jacobi` (default, fused
into the Krylov kernels), `:chebyshev` (Jacobi-preconditioned Chebyshev polynomial,
src/solver/precond/chebyshev.jl), `:mc_sgs` and `:mc_ilu0` (multicolour symmetric
Gauss–Seidel and ILU0, src/solver/precond/multicolor.jl), `:amg` (smoothed-aggregation
algebraic multigrid, src/solver/precond/amg.jl).
"""
const PRECONDITIONERS = (:jacobi, :chebyshev, :mc_sgs, :mc_ilu0, :amg)

"""
    DEFAULT_PRECONDITIONER

What `preconditioner = :auto` (the default of `ModflowGPU.run`, `initialize` and
`KrylovSettings(ims)`) selects: `:mc_ilu0`, the fastest preconditioner in the T4.4
benchmark on every model, precision and backend (steady state and transient steps;
docs/dev/precision.md).
"""
const DEFAULT_PRECONDITIONER = :mc_ilu0

"""
    resolve_preconditioner(kind::Symbol) -> Symbol

`kind`, or [`DEFAULT_PRECONDITIONER`](@ref) for `:auto`.
"""
resolve_preconditioner(kind::Symbol) = kind === :auto ? DEFAULT_PRECONDITIONER : kind

"""
    precond_setup!(P::AbstractPreconditioner, solver::KrylovSolver, amat) -> P

Set up `P` from the assembled matrix `amat` and the row and column factors of `solver`
(called by `solve!` once per call, i.e. once per outer iteration).
"""
function precond_setup! end

"""
    precond_apply!(z, P::AbstractPreconditioner, r) -> z

`z = M⁻¹·r` on the device (vectors in the Krylov precision `Ts`).
"""
function precond_apply! end

# `precond_setup!` with what `solve!` knows of the matrix (T4.7): `diag` (contiguous diagonal
# of `amat` or `nothing`) and `offdiag_id` (> 0: the off-diagonal entries are those of the
# last call with the same id; see `solve!`). Preconditioners that can use them add methods.
_precond_setup!(P::AbstractPreconditioner, solver, amat, diag, offdiag_id) = precond_setup!(P, solver, amat)

# `z = M⁻¹·r`, and `r·z` (Float64 where the backend has it) when the preconditioner fuses
# it into its last kernel (with `reducer`), else `nothing` (the solver then reduces).
# z = M⁻¹·r, and when the preconditioner fuses it, r·z into `reducer` (stage 1 and the
# finalisation with `post`, `sa`, `ss`; not synchronised): returns whether it did.
_precond_apply_dot!(z, P::AbstractPreconditioner, r, reducer, post, sa, ss) = (precond_apply!(z, P, r); false)

# Called after each linear solve with its `KrylovResult` (e.g. AMG's rebuild decision).
_precond_record!(P::AbstractPreconditioner, result) = nothing

# Whether a failed solve should be repeated after `precond_setup!` again (AMG with a
# hierarchy from an earlier matrix); the method prepares the rebuild.
_precond_retry!(P::AbstractPreconditioner) = false

"""
    JacobiPreconditioner(minv)

Jacobi (`ims_base_pcjac`): `minv` is the inverse diagonal of the scaled operator, in `Tp`.
Its vector is the `minv` of the `KrylovSolver`, which the fused CG and BiCGSTAB kernels
read directly.
"""
struct JacobiPreconditioner{VP <: AbstractVector} <: AbstractPreconditioner
    minv::VP
end

Adapt.@adapt_structure JacobiPreconditioner

function precond_setup!(P::JacobiPreconditioner, solver, amat)
    csr = solver.csr
    _map!(get_backend(P.minv), _setup_jacobi, Int(csr.nodes), P.minv, solver.rowfac, solver.colfac, amat, csr.ia)
    return P
end

@inline function _jacobi_apply(n, z, minv, r)
    @inbounds z[n] = _precond(minv, r[n], n)
    return nothing
end

function precond_apply!(z, P::JacobiPreconditioner, r)
    _map!(get_backend(z), _jacobi_apply, length(z), z, P.minv, r)
    return z
end

# Scaled operator values in Tp: vals[ii] = rowfac_n · a_nm · colfac_m for the entries of
# row n (shared by Chebyshev and the multicolour preconditioners; `dst` = position of the
# row's first entry in `vals`).
@inline function _scaled_row!(vals, dst, n, amat, ia, ja, rowfac, colfac)
    Tp = eltype(vals)
    Ts = eltype(rowfac)
    @inbounds begin
        rf = rowfac[n]
        i0 = ia[n]
        for ii in i0:(ia[n + 1] - Int32(1))
            vals[dst + (ii - i0)] = Tp(rf * Ts(amat[ii]) * colfac[ja[ii]])
        end
    end
    return nothing
end

"""
    build_preconditioner(kind, backend, csr, minv; Ts, Tp, colors = nothing, options...)
        -> AbstractPreconditioner

The preconditioner `kind` (one of `PRECONDITIONERS`) for the CSR structure `csr` on
`backend`; `minv` is the solver's Jacobi vector, `colors` an optional host colouring (one
colour per node, `1:ncolors`) for the multicolour preconditioners (otherwise greedy).
`options` (`precond_options` of `KrylovSettings`) are the keywords of its constructor:
none for `:jacobi`; `degree`, `ratio`, `lanczos`, `safety` for `:chebyshev`
(`ChebyshevPreconditioner`); `relax` for `:mc_sgs`/`:mc_ilu0`
(`MulticolorPreconditioner`); those of `AMGOptions` for `:amg`.
"""
function build_preconditioner(kind::Symbol, backend::Backend, csr, minv; Ts::Type, Tp::Type, colors = nothing, options...)
    if kind === :jacobi
        isempty(options) || throw(ArgumentError("the Jacobi preconditioner has no options (got $(keys(options)))"))
        return JacobiPreconditioner(minv)
    end
    kind === :chebyshev && return ChebyshevPreconditioner(backend, csr; Ts, Tp, options...)
    kind === :mc_sgs && return MulticolorPreconditioner(backend, csr, :sgs; Ts, Tp, colors, options...)
    kind === :mc_ilu0 && return MulticolorPreconditioner(backend, csr, :ilu0; Ts, Tp, colors, options...)
    kind === :amg && return AMGPreconditioner(backend, Int(csr.nodes), Ts, Tp; options...)
    throw(ArgumentError("unknown preconditioner $kind (expected one of $(PRECONDITIONERS))"))
end
