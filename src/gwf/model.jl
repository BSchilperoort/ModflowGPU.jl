# One GWF model on a device: the host setup (connectivity, NPF, reference heads; Float64)
# and every device array the outer iterations, flows and budgets work on (PLAN.md
# sections 3.3, 3.4 and 3.7).

"""
    GwfModel{T}

A GWF model ready to run on a device in precision `T` (state and assembly):

- host: `spec::GwfSpec`, `conn::Connectivity`, `setup::NpfSetup` (condsat, reference heads
  `r` and differences, Float64), `ibound_host` (1 active, −1 constant head), `kper`
  (stress period applied last, 0 before the first), `offdiag` (the `symmetric` setting
  the off-diagonal entries of `sys.amat` were last assembled for, `nothing` when they are
  stale: T4.5, see `formulate!`), `offdiag_version` (counts their rewrites, so a
  preconditioner can tell whether only the diagonal changed);
- device: `csr::CsrStructure`, `npf::NpfModel{T}`, `sys::LinearSystem{T}`, `packages`
  (`BoundaryPackage`s in MF6's order, `GwfSpec.boundaries`), `sto` (`StoragePackage` or
  `nothing`), `area` (cell areas, for RCH/EVT), `ibound`, the offset heads `h` and the
  heads at the start of the step `hold`, the residual `res`, the correction `δ`, `flowja`
  (`nja`).

    GwfModel(backend, T, spec::GwfSpec, nper)
"""
mutable struct GwfModel{T <: AbstractFloat, B <: Backend, C <: CsrStructure, N <: NpfModel{T}, S <: LinearSystem{T}, ST, V <: AbstractVector{T}, VI <: AbstractVector{Int32}}
    const spec::GwfSpec
    const backend::B
    const conn::Connectivity
    const setup::NpfSetup
    const csr::C
    const npf::N
    const sys::S
    const packages::Vector{BoundaryPackage}
    const sto::ST
    const area::V
    const ibound::VI
    ibound_host::Vector{Int32}
    const h::V
    const hold::V
    const res::V
    const δ::V
    const flowja::V
    kper::Int
    offdiag::Union{Nothing, Bool}
    offdiag_version::Int
end

function GwfModel(backend::Backend, ::Type{T}, spec::GwfSpec, nper::Integer) where {T <: AbstractFloat}
    spec.newton && throw(MF6UnsupportedError(spec.file, "unsupported: NEWTON in model $(spec.name) (Newton-Raphson arrives with T6.1)"))
    conn = build_connectivity(spec)
    setup = npf_setup(spec, conn)
    csr = CsrStructure(backend, conn)
    npf = NpfModel(backend, T, setup)
    sys = LinearSystem(backend, T, csr)
    packages = BoundaryPackage[BoundaryPackage(backend, T, b, conn, spec.dis) for b in spec.boundaries]
    sto = isnothing(spec.sto) ? nothing : StoragePackage(backend, T, spec.sto, conn, nper)
    z() = KernelAbstractions.zeros(backend, T, conn.nodes)
    ibh = ones(Int32, conn.nodes)
    h = initial_heads(backend, T, spec.ic, conn, setup.r)
    hold = _device_copy(backend, Array(h))
    return GwfModel(
        spec, backend, conn, setup, csr, npf, sys, packages, sto, _device_copy(backend, T.(conn.area)),
        _device_copy(backend, ibh), ibh, h, hold, z(), z(), KernelAbstractions.zeros(backend, T, conn.nja), 0,
        nothing, 0,
    )
end

# Adapt, e.g. `adapt(backend, m)`: every device array moves (and `backend` becomes the
# backend of the adapted heads); the host setup is shared with `m`.
function Adapt.adapt_structure(to, m::GwfModel)
    h = adapt(to, m.h)
    return GwfModel(
        m.spec, get_backend(h), m.conn, m.setup, adapt(to, m.csr), adapt(to, m.npf), adapt(to, m.sys),
        BoundaryPackage[adapt(to, p) for p in m.packages], adapt(to, m.sto), adapt(to, m.area), adapt(to, m.ibound),
        copy(m.ibound_host), h, adapt(to, m.hold), adapt(to, m.res), adapt(to, m.δ), adapt(to, m.flowja), m.kper,
        nothing, m.offdiag_version + 1,
    )
end

Base.eltype(::GwfModel{T}) where {T} = T
KernelAbstractions.get_backend(m::GwfModel) = m.backend
Base.show(io::IO, m::GwfModel{T}) where {T} =
    print(io, "GwfModel{", T, "}(\"", m.spec.name, "\", ", m.conn.nodes, " cells, ", length(m.packages), " stress package(s))")

"""
    apply_period!(m::GwfModel, kper) -> m

Start stress period `kper` (`bnd_rp`, `sto_rp`, `chd_rp`), once per period: the period
data of every stress package (host, Float64, then copied), the storage setting and the
IBOUND of the constant heads (copied to the device when it changed).
"""
function apply_period!(m::GwfModel, kper::Integer)
    # The IBOUND of the constant heads depends on the CHD cells only (T4.6: not rebuilt
    # when no CHD package has new cells).
    chd = m.kper == 0
    for pkg in m.packages
        st = _apply_period!(pkg, kper, m.setup.r, nothing)
        chd |= st === :structure && boundary_kind(pkg) === :CHD
    end
    isnothing(m.sto) || apply_period!(m.sto, kper)
    if chd
        ib = boundary_ibound(m.conn, m.packages)
        if ib != m.ibound_host
            m.ibound_host = ib
            copyto!(m.ibound, ib)
            m.offdiag = nothing
        end
    end
    m.kper = kper
    return m
end

"""
    start_timestep!(m::GwfModel, delt) -> m

Start a time step of length `delt` (Float64, host): `rho = SC/delt` (`set_timestep!`),
`hold ← h` (`gwf_ad`), and the constant heads applied to both (`chd_ad`).
"""
function start_timestep!(m::GwfModel, delt::Float64)
    isnothing(m.sto) || set_timestep!(m.sto, delt)
    copyto!(m.hold, m.h)
    for pkg in m.packages
        if boundary_kind(pkg) === :CHD
            apply_chd_heads!(m.h, pkg)
            apply_chd_heads!(m.hold, pkg)
        end
    end
    return m
end

"""
    formulate!(m::GwfModel; symmetric = false, scale = nothing) -> m

Formulate and assemble the system of an outer iteration from the current offset heads
(`bnd_cf`/`bnd_fc` of every stress package except CHD, `sto_fc`, `npf_fc`, `sln_ls`), then
the flux-form residual `m.res` (PLAN.md section 3.7, rule 2; with `scale`, also the
row sums `Σ |terms|`, see `flux_residual!`). Device work only: one formulation kernel per
package, then `formulate_system!` (T4.5), which rewrites the off-diagonal entries of the
matrix only when `ibound` or `symmetric` changed since they were last written.
"""
function formulate!(m::GwfModel; symmetric::Bool = false, scale = nothing)
    for pkg in m.packages
        boundary_kind(pkg) === :CHD && continue
        formulate!(pkg, m.h, m.ibound, m.area)
    end
    pkgs = Tuple((p.hcof, p.rhs, p.nodeptr, p.noderec) for p in m.packages if boundary_kind(p) !== :CHD && p.nbound > 0)
    storage = nothing
    if !isnothing(m.sto) && m.sto.transient
        _check_timestep(m.sto)
        storage = (m.sto.rho, m.hold)
    end
    offdiag = m.offdiag !== symmetric
    formulate_system!(m.sys, m.res, m.csr, m.npf, m.h, m.ibound, pkgs, storage; symmetric, offdiag, scale)
    if offdiag
        m.offdiag = symmetric
        m.offdiag_version += 1
    end
    return m
end

"""
    absolute_heads(m::GwfModel) -> Vector{Float64}

The heads per reduced node on the host in Float64, `r + h'` (PLAN.md section 3.7, rule 1:
absolute heads exist only on the host).
"""
absolute_heads(m::GwfModel) = m.setup.r .+ Float64.(Array(m.h))
