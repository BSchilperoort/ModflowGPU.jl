# Preconditioner interface of the Krylov solvers (src/solver/krylov.jl; T4.2, T4.3).
#
# A preconditioner `P <: AbstractPreconditioner` lives in the `KrylovSolver` (`precond`)
# and is used through two generic functions:
#
# - `precond_setup!(P, solver, amat)`: once per linear solve (once per outer iteration),
#   after the row and column factors (`solver.rowfac`, `solver.colfac`) and the Jacobi
#   inverse diagonal (`solver.minv`) of the scaled operator have been set up from `amat`.
#   The operator being preconditioned is the scaled (for CG sign-adjusted) matrix
#   `Â = diag(rowfac)·A·diag(colfac)` (`_op_row`).
# - `precond_apply!(z, P, r)`: `z = M⁻¹·r` for whole device vectors `z`, `r` (in `Ts`,
#   the Krylov type; the preconditioner's own data in `Tp`). `z` and `r` do not alias.
#
# `JacobiPreconditioner` is the default: its inverse diagonal is `solver.minv`, and CG and
# BiCGSTAB apply it fused into their vector kernels (`_precond`), so `precond_apply!` is
# never called for it. Every other preconditioner is applied with `precond_apply!` (one
# extra dot product per CG iteration). A preconditioner for CG must be symmetric positive
# definite (for the sign-adjusted operator).
#
# Selection: `KrylovSettings(; preconditioner)` (`:jacobi`, `:amg`; T4.2 adds `:chebyshev`,
# `:mc_sgs`, `:mc_ilu0`), with `precond_options` passed to the preconditioner's
# constructor; `make_preconditioner` builds it.

"""
    AbstractPreconditioner

Supertype of the preconditioners of `KrylovSolver` (see `precond_setup!`,
`precond_apply!` and the header of src/solver/precond/precond.jl).
"""
abstract type AbstractPreconditioner end

"""
    JacobiPreconditioner()

Jacobi (inverse diagonal of the scaled operator, `solver.minv`, IMS `ims_base_pcjac`),
applied fused into the CG and BiCGSTAB kernels. The default.
"""
struct JacobiPreconditioner <: AbstractPreconditioner end

"""
    precond_setup!(P::AbstractPreconditioner, solver::KrylovSolver, amat) -> P

Set up `P` for the current matrix `amat` (once per linear solve, after the scaling and the
Jacobi diagonal of `solver`). A no-op for `JacobiPreconditioner`.
"""
precond_setup!(P::JacobiPreconditioner, solver, amat) = P

"""
    precond_apply!(z, P::AbstractPreconditioner, r) -> z

`z = M⁻¹·r` on the device (whole vectors). Not used for `JacobiPreconditioner`, which is
fused into the Krylov kernels.
"""
function precond_apply! end

# Called after each linear solve with its `KrylovResult` (e.g. AMG's rebuild decision).
_precond_record!(P::AbstractPreconditioner, result) = nothing

# Whether a failed solve should be repeated after rebuilding `P` (AMG with a hierarchy
# from an earlier matrix); the method prepares the rebuild.
_precond_retry!(P::AbstractPreconditioner) = false

# `z = M⁻¹·r`, and `r·z` (Float64) when the preconditioner fuses it into its last kernel
# (with `reducer`), else `nothing` (the solver then computes the dot product).
_precond_apply_dot!(z, P::AbstractPreconditioner, r, reducer) = (precond_apply!(z, P, r); nothing)

const PRECONDITIONERS = (:jacobi, :amg)

"""
    make_preconditioner(name, backend, csr, Ts, Tp; options...) -> AbstractPreconditioner

The preconditioner `name` (`:jacobi`, `:amg`) for the CSR structure `csr` on `backend`,
with Krylov vectors in `Ts` and preconditioner data in `Tp`; `options` go to its
constructor (`AMGOptions` for `:amg`).
"""
function make_preconditioner(name::Symbol, backend::Backend, csr, ::Type{Ts}, ::Type{Tp}; options...) where {Ts, Tp}
    if name === :jacobi
        isempty(options) || throw(ArgumentError("the Jacobi preconditioner has no options (got $(keys(options)))"))
        return JacobiPreconditioner()
    elseif name === :amg
        return AMGPreconditioner(backend, Int(csr.nodes), Ts, Tp; options...)
    end
    throw(ArgumentError("unknown preconditioner $name (expected one of $PRECONDITIONERS)"))
end
