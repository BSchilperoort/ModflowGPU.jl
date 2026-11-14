# Assembly of the GWF system in MF6's CSR layout and the flux-form residual (PLAN.md
# sections 3.3 and 3.7).
#
# Ports, for confined models (ICELLTYPE 0, no XT3D):
# - `npf_fc` / `fc_default_flow` (gwf-npf.f90) with `hcond` / `vcond`
#   (GwfConductanceUtils.f90): off-diagonal `a_nm = condsat_nm` when neither cell has
#   `ibound == 0`, diagonal `−Σ_m a_nm`;
# - `bnd_fc` (BoundaryPackage.f90) and `sto_fc` (gwf-sto.f90): the boundary and storage
#   terms, added by the packages' `add_to_system!` into the per-node vectors `bdiag` and
#   `brhs` of a `LinearSystem` (`add_terms!`);
# - the row adjustments of `sln_ls` (NumericalSolution.f90): an active row whose diagonal
#   is below 1e-15 in magnitude gets diagonal −1 and `rhs −= x`; a row with `ibound <= 0`
#   (constant head; no-flow cells are not in the reduced system) becomes an identity row
#   with `rhs = x`; with a symmetric solver (IMS CG) the entries of active rows in
#   constant-head columns are moved to the right-hand side (`rhs −= a_nm·x_m`, `a_nm = 0`).
#   See docs/mf6_notes.md (constant heads).
#
# Heads are offsets `h' = h − r` from the per-cell references `r` (section 3.7, rule 1).
# The matrix is MF6's `A`; the right-hand side is `b' = b − A·r`: for an active row
# `b'_n = brhs_n − Σ_m a_nm·Δr_nm` (the boundary and storage terms are already formulated
# with offsets), for a constant-head row `b'_n = h'_n`.
#
# The residual (rule 2) is never computed as `b' − A·h'`. Per active row,
#
#     res_n = brhs_n − bdiag_n·h'_n − Σ_m a_nm·(h'_m − h'_n + Δr_nm),
#
# which equals `(b − A·h)_n` in exact arithmetic, with the row sum accumulated with
# compensated (Neumaier) summation in `T`, so it also works on backends without Float64.

"""
    CsrStructure{VI}

Device view of the CSR structure of a `Connectivity` (PLAN.md section 3.3; MF6 layout,
diagonal first in each row): `nodes`, `nja`, and the `Int32` vectors `ia` `(nodes + 1)`,
`ja`, `jas`, `isym` `(nja)`, `idiag = ia[1:nodes]` (diagonal positions, for
`add_to_system!` into a matrix), `inode = 1:nodes` (for `add_to_system!` into a vector
of per-node diagonal terms) and `irow` `(nja)`, the row of each entry (T4.5: kernels with
one work item per entry, whose writes are coalesced, unlike those of a work item per row).
Moved to a device with `Adapt`.
"""
struct CsrStructure{VI <: AbstractVector{Int32}}
    nodes::Int32
    nja::Int32
    ia::VI
    ja::VI
    jas::VI
    isym::VI
    idiag::VI
    inode::VI
    irow::VI
end

Adapt.@adapt_structure CsrStructure

# Without `irow`: computed from `ia` (on the host).
function CsrStructure(nodes, nja, ia::VI, ja::VI, jas::VI, isym::VI, idiag::VI, inode::VI) where {VI}
    iah = Array(ia)
    irow = Vector{Int32}(undef, Int(nja))
    for n in 1:Int(nodes), ii in iah[n]:(iah[n + 1] - 1)
        irow[ii] = n
    end
    dirow = similar(ia, Int(nja))
    copyto!(dirow, irow)
    return CsrStructure{VI}(Int32(nodes), Int32(nja), ia, ja, jas, isym, idiag, inode, dirow)
end

"""
    CsrStructure(backend, conn::Connectivity) -> CsrStructure

Copy the CSR structure of `conn` to `backend`.
"""
function CsrStructure(backend::Backend, conn::Connectivity)
    dev(x) = _device_copy(backend, x)
    return CsrStructure(
        conn.nodes, conn.nja, dev(conn.ia), dev(conn.ja), dev(conn.jas), dev(conn.isym),
        dev(conn.ia[1:conn.nodes]), dev(collect(Int32(1):conn.nodes)),
    )
end

KernelAbstractions.get_backend(csr::CsrStructure) = get_backend(csr.ia)

"""
    LinearSystem{T}

The assembled system of one model on a device, in the engine precision `T`:

- `amat` `(nja)`: MF6's matrix in CSR order (negative diagonal);
- `rhs` `(nodes)`: the right-hand side `b' = b − A·r` for offset heads;
- `bdiag`, `brhs` `(nodes)`: the sums of the boundary and storage terms of each node
  (`hcof` and `rhs` of `bnd_fc`, `−rho` and `−rho·hold` of `sto_fc`), filled by
  `add_terms!` after `reset_terms!`, and used by both `assemble!` and `flux_residual!`.
"""
struct LinearSystem{T <: AbstractFloat, VT <: AbstractVector{T}}
    amat::VT
    rhs::VT
    bdiag::VT
    brhs::VT
end

Adapt.@adapt_structure LinearSystem

"""
    LinearSystem(backend, T, csr::CsrStructure) -> LinearSystem{T}

A zeroed system for the structure `csr` on `backend`.
"""
function LinearSystem(backend::Backend, ::Type{T}, csr::CsrStructure) where {T <: AbstractFloat}
    z(n) = KernelAbstractions.zeros(backend, T, n)
    return LinearSystem{T, typeof(z(0))}(z(csr.nja), z(csr.nodes), z(csr.nodes), z(csr.nodes))
end

Base.eltype(::LinearSystem{T}) where {T} = T

"""
    reset_terms!(sys::LinearSystem) -> sys

Zero the boundary and storage terms `bdiag` and `brhs` (start of a formulation).
"""
function reset_terms!(sys::LinearSystem)
    fill!(sys.bdiag, zero(eltype(sys)))
    fill!(sys.brhs, zero(eltype(sys)))
    return sys
end

"""
    add_terms!(sys, csr, pkg::BoundaryPackage) -> sys
    add_terms!(sys, csr, sto::StoragePackage, hold, ibound) -> sys

Add the terms of a boundary package (after `formulate!`) or of the storage package to
`sys.bdiag` and `sys.brhs` (`add_to_system!` with `idiag = csr.inode`).
"""
function add_terms!(sys::LinearSystem, csr::CsrStructure, pkg::BoundaryPackage)
    add_to_system!(sys.bdiag, sys.brhs, csr.inode, pkg)
    return sys
end

function add_terms!(sys::LinearSystem, csr::CsrStructure, sto::StoragePackage, hold, ibound)
    add_to_system!(sys.bdiag, sys.brhs, csr.inode, sto, hold, ibound)
    return sys
end

# Off-diagonal coefficient of entry `ii` (row `n`, column `m`): `condsat`, zero when either
# cell is inactive (`hcond`/`vcond` with ibound 0), and the reference difference r_m − r_n.
@inline function _connection(condsat, Δr, jas, ibound, n, m, ii)
    j = jas[ii]
    c = (ibound[n] != 0 && ibound[m] != 0) ? condsat[j] : zero(eltype(condsat))
    dr = m > n ? Δr[j] : -Δr[j]
    return c, dr
end

@kernel function assemble_kernel!(
        amat, rhs, @Const(ia), @Const(ja), @Const(jas), @Const(condsat), @Const(Δr), @Const(bdiag), @Const(brhs),
        @Const(h), @Const(ibound), symmetric::Bool, tiny,
    )
    n = @index(Global)
    T = eltype(amat)
    @inbounds begin
        id = ia[n]
        i2 = ia[n + 1] - Int32(1)
        if ibound[n] > 0
            d = zero(T)
            s = brhs[n]
            for ii in (id + Int32(1)):i2
                m = ja[ii]
                c, dr = _connection(condsat, Δr, jas, ibound, n, m, ii)
                d -= c
                s -= c * dr
                if symmetric && ibound[m] < 0
                    # sln_ls, symmetric matrix: constant-head column to the right-hand side.
                    s -= c * h[m]
                    amat[ii] = zero(T)
                else
                    amat[ii] = c
                end
            end
            d += bdiag[n]
            # sln_ls: small diagonal of an active cell (DEM15).
            if abs(d) < tiny
                d = -one(T)
                s -= h[n]
            end
            amat[id] = d
            rhs[n] = s
        else
            # sln_ls: Dirichlet (constant-head) row.
            amat[id] = one(T)
            for ii in (id + Int32(1)):i2
                amat[ii] = zero(T)
            end
            rhs[n] = h[n]
        end
    end
end

"""
    assemble!(sys::LinearSystem, csr::CsrStructure, npf::NpfModel, h, ibound; symmetric = false) -> sys

Assemble MF6's matrix `sys.amat` and the offset right-hand side `sys.rhs = b − A·r` (see
the header of src/solver/csr.jl) from the saturated conductances and reference
differences of `npf` (device arrays), the boundary and storage terms in `sys.bdiag` and
`sys.brhs`, the offset heads `h` (with the constant heads applied) and `ibound`, one work
item per row (`npf_fc`, `sln_ls`). `symmetric = true` (IMS `LINEAR_ACCELERATION CG`)
moves the constant-head columns of active rows to the right-hand side, as `sln_ls` does
for a symmetric solver.
"""
function assemble!(sys::LinearSystem, csr::CsrStructure, npf::NpfModel, h, ibound; symmetric::Bool = false)
    backend = get_backend(sys.amat)
    assemble_kernel!(backend, _WG)(
        sys.amat, sys.rhs, csr.ia, csr.ja, csr.jas, npf.condsat, npf.Δr, sys.bdiag, sys.brhs, h, ibound, symmetric,
        eltype(sys)(1.0e-15); ndrange = csr.nodes,
    )
    return sys
end

# Neumaier's compensated sum: add `x` to the sum `s` with compensation `c`.
@inline function _sum_add(s::T, c::T, x::T) where {T}
    t = s + x
    c += abs(s) >= abs(x) ? (s - t) + x : (x - t) + s
    return t, c
end

@kernel function flux_residual_kernel!(
        res, scale, bound, @Const(ia), @Const(ja), @Const(jas), @Const(condsat), @Const(Δr), @Const(bdiag),
        @Const(brhs), @Const(h), @Const(ibound),
    )
    n = @index(Global)
    T = eltype(res)
    @inbounds begin
        s = zero(T)
        c = zero(T)
        sc = zero(T)
        rb = zero(T)
        if ibound[n] > 0
            hn = h[n]
            for ii in (ia[n] + Int32(1)):(ia[n + 1] - Int32(1))
                m = ja[ii]
                cnm, dr = _connection(condsat, Δr, jas, ibound, n, m, ii)
                q = cnm * (h[m] - hn + dr)
                s, c = _sum_add(s, c, -q)
                sc += abs(q)
                rb += cnm * (abs(h[m]) + abs(hn) + abs(dr))
            end
            tb = bdiag[n] * hn
            s, c = _sum_add(s, c, -tb)
            s, c = _sum_add(s, c, brhs[n])
            sc += abs(tb) + abs(brhs[n])
            rb += abs(tb) + abs(brhs[n])
        end
        res[n] = s + c
        scale === nothing || (scale[n] = sc)
        bound === nothing || (bound[n] = rb)
    end
end

"""
    flux_residual!(res, sys::LinearSystem, csr::CsrStructure, npf::NpfModel, h, ibound;
                   scale = nothing, bound = nothing) -> res

The flux-form residual (PLAN.md section 3.7, rule 2) of every row, `res_n = (b − A·h)_n`
computed as `brhs_n − bdiag_n·h'_n − Σ_m a_nm·(h'_m − h'_n + Δr_nm)` with compensated
summation in `T` (see the header of src/solver/csr.jl); zero for rows with
`ibound <= 0`. With `scale` (a vector of length `nodes`) it also stores `Σ |terms|` per
row (the denominator of the universal residual test, PLAN.md section 4), and with `bound`
the magnitude sum `Σ a_nm·(|h'_m| + |h'_n| + |Δr_nm|) + |bdiag_n·h'_n| + |brhs_n|`, which
times `eps(T)` bounds the effect of rounding the heads to `T`.
"""
function flux_residual!(res, sys::LinearSystem, csr::CsrStructure, npf::NpfModel, h, ibound; scale = nothing, bound = nothing)
    backend = get_backend(res)
    flux_residual_kernel!(backend, _WG)(
        res, scale, bound, csr.ia, csr.ja, csr.jas, npf.condsat, npf.Δr, sys.bdiag, sys.brhs, h, ibound;
        ndrange = csr.nodes,
    )
    return res
end

# --- Fused formulation (T4.5): boundary and storage terms, assembly and residual in one pass
#
# One work item per row does what `reset_terms!`, `add_terms!` (every package, then
# storage), `assemble!` and `flux_residual!` do in sequence, with the same operations in the
# same order (bitwise the same results), so the row's connections are read once. The
# off-diagonal entries of `amat` depend only on `condsat`, `ibound` and `symmetric`; with
# `offdiag = false` they are not written (the caller knows they are unchanged), which halves
# the bytes moved: the row-parallel writes of `amat` cost as much as the rest of the kernel.

# bdiag/brhs of node n: the records of each package (`boundary_add_kernel!`), in package
# order, unrolled over the tuple.
@inline _add_package_terms(n, d, s, ::Tuple{}) = (d, s)
@inline function _add_package_terms(n, d, s, pkgs::Tuple)
    hcof, rhs, nodeptr, noderec = pkgs[1]
    @inbounds for k in nodeptr[n]:(nodeptr[n + 1] - Int32(1))
        i = noderec[k]
        d += hcof[i]
        s += rhs[i]
    end
    return _add_package_terms(n, d, s, Base.tail(pkgs))
end

# Storage (`storage_add_kernel!`): `nothing` in a steady-state period.
@inline _add_storage_terms(n, d, s, ::Nothing, ibound) = (d, s)
@inline function _add_storage_terms(n, d, s, sto::Tuple, ibound)
    rho, hold = sto
    @inbounds if ibound[n] > 0
        r = rho[n]
        d = d - r
        s = s - r * hold[n]
    end
    return d, s
end

@kernel function formulate_kernel!(
        amat, rhs, bdiag, brhs, res, scale, @Const(ia), @Const(ja), @Const(jas), @Const(condsat), @Const(Δr),
        @Const(h), @Const(ibound), pkgs, sto, symmetric::Bool, offdiag::Bool, tiny,
    )
    n = @index(Global)
    T = eltype(amat)
    @inbounds begin
        bd, br = _add_package_terms(n, zero(T), zero(T), pkgs)
        bd, br = _add_storage_terms(n, bd, br, sto, ibound)
        bdiag[n] = bd
        brhs[n] = br
        id = ia[n]
        i2 = ia[n + 1] - Int32(1)
        # residual (flux_residual_kernel!)
        rs = zero(T)
        rc = zero(T)
        sc = zero(T)
        if ibound[n] > 0
            d = zero(T)
            s = br
            hn = h[n]
            for ii in (id + Int32(1)):i2
                m = ja[ii]
                c, dr = _connection(condsat, Δr, jas, ibound, n, m, ii)
                d -= c
                s -= c * dr
                q = c * (h[m] - hn + dr)
                rs, rc = _sum_add(rs, rc, -q)
                sc += abs(q)
                if symmetric && ibound[m] < 0
                    s -= c * h[m]
                    offdiag && (amat[ii] = zero(T))
                else
                    offdiag && (amat[ii] = c)
                end
            end
            d += bd
            if abs(d) < tiny
                d = -one(T)
                s -= hn
            end
            amat[id] = d
            rhs[n] = s
            tb = bd * hn
            rs, rc = _sum_add(rs, rc, -tb)
            rs, rc = _sum_add(rs, rc, br)
            sc += abs(tb) + abs(br)
        else
            amat[id] = one(T)
            if offdiag
                for ii in (id + Int32(1)):i2
                    amat[ii] = zero(T)
                end
            end
            rhs[n] = h[n]
        end
        res[n] = rs + rc
        scale === nothing || (scale[n] = sc)
    end
end

"""
    formulate_system!(sys, res, csr, npf, h, ibound, packages, storage; symmetric = false,
                      offdiag = true, scale = nothing) -> sys

`reset_terms!`, `add_terms!` of the boundary packages and storage, `assemble!` and
`flux_residual!` in one kernel (T4.5), with bitwise the same results. `packages`: a tuple
of `(hcof, rhs, nodeptr, noderec)` per boundary package with records (after its
`formulate!`), in package order; `storage`: `(rho, hold)` in a transient period, otherwise
`nothing`. `offdiag = false` skips writing the off-diagonal entries of `sys.amat`; valid
only when they were written by an earlier call with the same `ibound` and `symmetric` (they
depend on nothing else).
"""
function formulate_system!(
        sys::LinearSystem, res, csr::CsrStructure, npf::NpfModel, h, ibound, packages::Tuple, storage;
        symmetric::Bool = false, offdiag::Bool = true, scale = nothing,
    )
    backend = get_backend(sys.amat)
    formulate_kernel!(backend, _WG)(
        sys.amat, sys.rhs, sys.bdiag, sys.brhs, res, scale, csr.ia, csr.ja, csr.jas, npf.condsat, npf.Δr, h, ibound,
        packages, storage, symmetric, offdiag, eltype(sys)(1.0e-15); ndrange = csr.nodes,
    )
    return sys
end

@kernel function spmv_kernel!(y, @Const(amat), @Const(ia), @Const(ja), @Const(x))
    n = @index(Global)
    T = eltype(y)
    @inbounds begin
        s = zero(T)
        for ii in ia[n]:(ia[n + 1] - Int32(1))
            s += T(amat[ii]) * T(x[ja[ii]])
        end
        y[n] = s
    end
end

"""
    spmv!(y, amat, csr::CsrStructure, x) -> y

`y = A·x` for the matrix `amat` in the CSR layout of `csr` (SPARSKIT `amux`, used by
ImsLinearBase.f90), one work item per row, accumulated in `eltype(y)`.
"""
function spmv!(y, amat, csr::CsrStructure, x)
    spmv_kernel!(get_backend(y), _WG)(y, amat, csr.ia, csr.ja, x; ndrange = csr.nodes)
    return y
end
