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
  stale: T4.5, see `formulate!`), `offdiag_id` (T4.7: an id, unique in the session, of
  the last write of the off-diagonal entries by `formulate!`, 0 before; the solver reuses
  preconditioner data while it is unchanged);
- device: `csr::CsrStructure`, `npf::NpfModel{T}` (with `npf.cv`, the convertible data,
  T6.1), `sys::LinearSystem{T}`, `packages`
  (`BoundaryPackage`s in MF6's order, `GwfSpec.boundaries`), `sto` (`StoragePackage` or
  `nothing`), `area` (cell areas, for RCH/EVT), `ibound`, the offset heads `h` and the
  heads at the start of the step `hold`, the residual `res`, the correction `δ`, `flowja`
  (`nja`), `wd` (T6.1: the `WetDry` state of a model with convertible cells, else
  `nothing`).

With convertible cells IBOUND changes during the outer iterations (`wetdry!`: dry cells get
0); `ibound_host` follows it.

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
    offdiag_id::Int
    const wd::Any
end

function GwfModel(backend::Backend, ::Type{T}, spec::GwfSpec, nper::Integer) where {T <: AbstractFloat}
    spec.newton && throw(MF6UnsupportedError(spec.file, "unsupported: NEWTON in model $(spec.name) (Newton-Raphson arrives with T6.1)"))
    conn = build_connectivity(spec)
    setup = npf_setup(spec, conn)
    csr = CsrStructure(backend, conn)
    npf = NpfModel(backend, T, setup; options = spec.npf, conn)
    sys = LinearSystem(backend, T, csr)
    packages = BoundaryPackage[BoundaryPackage(backend, T, b, conn, spec.dis; icelltype = setup.properties.icelltype) for b in spec.boundaries]
    sto = isnothing(spec.sto) ? nothing : StoragePackage(backend, T, spec.sto, conn, nper; r = setup.r)
    z() = KernelAbstractions.zeros(backend, T, conn.nodes)
    ibh = ones(Int32, conn.nodes)
    h = initial_heads(backend, T, spec.ic, conn, setup.r)
    hold = _device_copy(backend, Array(h))
    wd = isnothing(npf.cv) ? nothing : WetDry(backend, T, setup.properties, spec.npf, conn.nodes)
    m = GwfModel(
        spec, backend, conn, setup, csr, npf, sys, packages, sto, _device_copy(backend, T.(conn.area)),
        _device_copy(backend, ibh), ibh, h, hold, z(), z(), KernelAbstractions.zeros(backend, T, conn.nja), 0,
        nothing, 0, wd,
    )
    # `preprocess_input`: `wd(0, hnew)` with the starting heads (IBOUND without the
    # constant heads, which `chd_rp` sets later).
    isnothing(wd) || wetdry!(m, 0)
    return m
end

# Adapt, e.g. `adapt(backend, m)`: every device array moves (and `backend` becomes the
# backend of the adapted heads); the host setup is shared with `m`.
function Adapt.adapt_structure(to, m::GwfModel)
    h = adapt(to, m.h)
    return GwfModel(
        m.spec, get_backend(h), m.conn, m.setup, adapt(to, m.csr), adapt(to, m.npf), adapt(to, m.sys),
        BoundaryPackage[adapt(to, p) for p in m.packages], adapt(to, m.sto), adapt(to, m.area), adapt(to, m.ibound),
        copy(m.ibound_host), h, adapt(to, m.hold), adapt(to, m.res), adapt(to, m.δ), adapt(to, m.flowja), m.kper,
        nothing, 0, adapt(to, m.wd),
    )
end

Base.eltype(::GwfModel{T}) where {T} = T
KernelAbstractions.get_backend(m::GwfModel) = m.backend
Base.show(io::IO, m::GwfModel{T}) where {T} =
    print(io, "GwfModel{", T, "}(\"", m.spec.name, "\", ", m.conn.nodes, " cells, ", length(m.packages), " stress package(s))")

"""
    apply_period!(m::GwfModel, kper; window = nothing) -> m

Start stress period `kper` (`bnd_rp`, `sto_rp`, `chd_rp`), once per period: the period
data of every stress package (host, Float64, then copied), the storage setting and the
IBOUND of the constant heads (copied to the device when it changed). `window = (t0, t1)`
is the first time step of the period, for packages with time-series values (T6.3).
"""
function apply_period!(m::GwfModel, kper::Integer; window::Union{Nothing, NTuple{2, Float64}} = nothing)
    # The IBOUND of the constant heads depends on the CHD cells only (T4.6: not rebuilt
    # when no CHD package has new cells).
    chd = m.kper == 0
    # RCH/EVT records move to the highest active cell (convertible models only: elsewhere
    # IBOUND has no zeros, and passing it would turn off the T4.6 fast path).
    ibmove = isnothing(m.wd) ? nothing : m.ibound_host
    for pkg in m.packages
        st = _apply_period!(pkg, kper, m.setup.r, ibmove, window)
        chd |= st === :structure && boundary_kind(pkg) === :CHD
    end
    isnothing(m.sto) || apply_period!(m.sto, kper)
    if chd
        # chd_rp: the cells of the previous constant heads become active again, dry cells
        # stay dry (T6.1), the new constant-head cells get −1.
        ib = boundary_ibound(m.conn, m.packages, m.ibound_host)
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
    advance_time_series!(m::GwfModel, window) -> m

Evaluate the time series of every stress package for the time step `window = (t0, t1)`
(`[totimc, totimc + delt]`; `idm_ts_advance` at the end of `Mf6PrepareTimestep`) and send
the changed `bound` arrays to the device (T6.3). Packages without time series, or already
evaluated for `window` by `apply_period!`, are skipped.
"""
function advance_time_series!(m::GwfModel, window::NTuple{2, Float64})
    for pkg in m.packages
        advance_time_series!(pkg, window, m.setup.r)
    end
    return m
end

"""
    start_timestep!(m::GwfModel, delt) -> m

Start a time step of length `delt` (Float64, host): `rho = SC/delt` (`set_timestep!`),
`hold ← h` (`gwf_ad`; `npf_ad` with REWET: the bottom for dry wettable cells, T6.1), and
the constant heads applied to both (`chd_ad`).
"""
function start_timestep!(m::GwfModel, delt::Float64)
    isnothing(m.sto) || set_timestep!(m.sto, delt)
    copyto!(m.hold, m.h)
    # npf_ad (T6.1): the old head of a dry cell that can rewet is its bottom.
    isnothing(m.wd) || npf_ad!(m.wd, m.npf.cv, m.hold, m.ibound)
    for pkg in m.packages
        if boundary_kind(pkg) === :CHD
            apply_chd_heads!(m.h, pkg)
            apply_chd_heads!(m.hold, pkg)
        end
    end
    return m
end

# Ids of the writes of off-diagonal entries (`GwfModel.offdiag_id`), unique per session.
const _OFFDIAG_IDS = Threads.Atomic{Int}(0)

"""
    formulate!(m::GwfModel; symmetric = false, terms = true, scale = nothing) -> m

Formulate and assemble the system of an outer iteration from the current offset heads
(`bnd_cf`/`bnd_fc` of every stress package except CHD, `sto_fc`, `npf_fc`, `sln_ls`), then
the flux-form residual `m.res` (PLAN.md section 3.7, rule 2; with `scale`, also the
row sums `Σ |terms|`, see `flux_residual!`). Device work only: `formulate_system!` (T4.5;
since T4.7 it also formulates the package records), which rewrites the off-diagonal entries of the
matrix only when `ibound` or `symmetric` changed since they were last written.
`terms = false` (the outer iterations, T4.7) leaves `sys.rhs`, `sys.bdiag` and `sys.brhs`
unwritten (see `formulate_system!`).
"""
function formulate!(m::GwfModel; symmetric::Bool = false, terms::Bool = true, scale = nothing)
    # The package records are formulated in the row kernel (T4.7; `formulate!` per package
    # did it in a kernel of its own). Records without a cell keep their `hcof`/`rhs`; no
    # reader uses them (`calc_simvals!` skips them).
    pkgs = Tuple(
        (BoundaryKind{boundary_kind(p)}(), p.bound, m.area, p.hcof, p.rhs, p.nodeptr, p.noderec)
            for p in m.packages if boundary_kind(p) !== :CHD && p.nbound > 0
    )
    storage = nothing
    if !isnothing(m.sto) && m.sto.transient
        _check_timestep(m.sto)
        storage = isnothing(m.sto.conv) ? (m.sto.rho, m.hold) : (m.sto.rho, m.hold, m.sto.conv)
    end
    # Valid while the off-diagonals depend only on the constant `condsat`, `ibound` and
    # `symmetric`. Whatever changes them must set `m.offdiag = nothing`: `apply_period!`
    # (IBOUND) does; with convertible cells (T6.1) the conductances depend on the heads, so
    # the saturations are recomputed (`npf_cf!`) and the off-diagonals rewritten every time
    # (a new `offdiag_id`: the preconditioner takes the full setup path). Calling
    # `assemble!` on `m.sys` directly also leaves `m.offdiag` stale.
    if !isnothing(m.npf.cv)
        npf_cf!(m.npf.cv, m.h, m.ibound)
        m.offdiag = nothing
    end
    offdiag = m.offdiag !== symmetric
    offdiag && (m.offdiag_id = Threads.atomic_add!(_OFFDIAG_IDS, 1) + 1)
    formulate_system!(m.sys, m.res, m.csr, m.npf, m.h, m.ibound, pkgs, storage; symmetric, offdiag, terms, scale)
    m.offdiag = symmetric
    return m
end

"""
    absolute_heads(m::GwfModel) -> Vector{Float64}

The heads per reduced node on the host in Float64, `r + h'` (PLAN.md section 3.7, rule 1:
absolute heads exist only on the host).
"""
function absolute_heads(m::GwfModel)
    h = m.setup.r .+ Float64.(Array(m.h))
    if !isnothing(m.wd)
        # Dry cells (T6.1): HDRY, as MF6 stores in `x` (`sgwf_npf_wetdry`).
        for n in eachindex(h)
            m.ibound_host[n] == 0 && (h[n] = HDRY)
        end
    end
    return h
end

"""
    wetdry!(m::GwfModel, kiter) -> Bool

Wetting and drying before the formulation of outer iteration `kiter` (`npf_cf` →
`sgwf_npf_wetdry`; `kiter = 0` once at the start, `preprocess_input`), for a model with
convertible cells (T6.1; `wetdry_device!`). When IBOUND changed, `ibound_host` is updated,
the RCH/EVT records move to the highest active cell (`rch_cf`, `evt_cf`), and the
off-diagonals are marked stale. A constant-head cell going dry is an `MF6InputError`
("CONSTANT-HEAD CELL WENT DRY -- SIMULATION ABORTED"). Returns whether IBOUND changed.
"""
function wetdry!(m::GwfModel, kiter::Integer)
    isnothing(m.wd) && return false
    changed, chddry = wetdry_device!(m.wd, m.npf.cv, m.csr, m.npf.Δr, m.h, m.ibound, kiter)
    chddry && throw(MF6InputError(m.spec.npf.file, 0, "CONSTANT-HEAD CELL WENT DRY -- SIMULATION ABORTED"))
    changed || return false
    copyto!(m.ibound_host, m.ibound)
    m.offdiag = nothing
    for pkg in m.packages
        boundary_kind(pkg) in (:RCH, :EVT) && move_to_highest_active!(pkg, m.ibound_host, m.setup.r)
    end
    return true
end
