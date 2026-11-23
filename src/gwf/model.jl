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
  `nothing`; also `nothing` with the Newton formulation, which does not dry cells);
- T6.2: `newton` (the NEWTON option), `nurbot` (NEWTON UNDER_RELAXATION: per node the bottom
  of the lowest cell below it as an offset, `npf_nur`'s `ibotnode`; `-Inf` where it does
  not apply; empty without it), `vol` (cell volumes at full saturation, for the
  pseudo-transient continuation of steady-state Newton periods; empty without NEWTON) and
  `delt` (the length of the current time step, host);
- T6.5: `xt3d` (the `Xt3dNpf` of a model with the NPF XT3D option, else `nothing`) and
  `scsr`, the CSR structure of the system `sys` and the linear solver: `csr` itself, or with
  XT3D (without its RHS option) the stencil extended with the neighbours of neighbours.
  `csr` stays the connectivity (flows, `flowja`). T6.13: a model with GNC6 has the flux
  terms of its connections and ghost-node corrections in `xt3d` (src/gwf/gnc.jl), and with
  implicit GNC `scsr` is the connectivity plus GNC's entries.
- T6.6: `solution` (`nothing` for a single model; for the models of a multi-model solution
  the host `SolutionSetup`: the model is then the merged model of the solution, see
  src/gwf/exchange.jl) and `exgentry` (device Bool per CSR entry marking the exchange
  connections, or `nothing`).
- T6.11: `hfb` (the `HfbState` of a model with HFB6, else `nothing`; `apply_period!`
  applies its PERIOD blocks).
- T6.12: `tvk` (the `TvkState` of a model with TVK6, else `nothing`; `apply_period!` and
  `advance_time_series!` apply it). TVS6 is part of `sto` (`StoragePackage.tvs`).
- T6.14: `xvar` (the host `Xt3dVarSetup` of a model with XT3D or GNC6 on the head-dependent
  path, src/gwf/xt3d_var.jl: with convertible cells, NEWTON, HFB6 or TVK6; `xt3d` is then
  its `Xt3dVarNpf`; else `nothing`).
- T6.9: `conversions`, the cell conversions of `wetdry!` since the start of the time step
  (`(kiter, wet, dry)` with reduced nodes in MF6's order, one entry per call that converted
  a cell; the initial call of the constructor has `kiter` 0), for the listing
  (`write_conversions`).

With convertible cells IBOUND changes during the outer iterations (`wetdry!`: dry cells get
0); `ibound_host` follows it.

    GwfModel(backend, T, spec::GwfSpec, nper)
    GwfModel(backend, T, sol::SolutionSetup, nper)
"""
mutable struct GwfModel{T <: AbstractFloat, B <: Backend, C <: CsrStructure, N <: NpfModel{T}, S <: LinearSystem{T}, ST, V <: AbstractVector{T}, VI <: AbstractVector{Int32}, X}
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
    const newton::Bool
    const nurbot::V
    const vol::V
    delt::Float64
    const xt3d::X
    const scsr::C
    const solution::Any
    const exgentry::Any
    const conversions::Vector{Tuple{Int, Vector{Int32}, Vector{Int32}}}
    const hfb::Union{Nothing, HfbState}
    const tvk::Union{Nothing, TvkState}
    const xvar::Union{Nothing, Xt3dVarSetup}
end

function GwfModel(backend::Backend, ::Type{T}, spec::GwfSpec, nper::Integer) where {T <: AbstractFloat}
    newton = spec.newton
    conn = build_connectivity(spec)
    setup = npf_setup(spec, conn)
    csr = CsrStructure(backend, conn)
    hfb = hfb_state(spec)
    npf = NpfModel(backend, T, setup; options = spec.npf, conn, newton, hfb = !isnothing(hfb))
    # T6.5: XT3D (confined models; checked by `npf_setup`).
    # T6.13: GNC6 as flux terms on the XT3D kernels (src/gwf/gnc.jl; XT3D with GNC is an error).
    # T6.14: with convertible cells, NEWTON, HFB6 or TVK6 the head-dependent path
    # (src/gwf/xt3d_var.jl).
    check_gnc(spec, setup.properties.icelltype)
    xvar = nothing
    xs = nothing
    if xt3d_variable(spec, setup.properties.icelltype)
        xvar = spec.npf.xt3d ? xt3d_var_setup(conn, setup.properties, setup.r; rhs = spec.npf.xt3d_rhs, newton) :
            gnc_var_setup(spec, conn, setup)
        xt3d = Xt3dVarNpf(backend, T, xvar)
    else
        xs = spec.npf.xt3d ? xt3d_setup(conn, setup.properties, setup.r; rhs = spec.npf.xt3d_rhs) : gnc_setup(spec, conn, setup)
        xt3d = isnothing(xs) ? nothing : Xt3dNpf(backend, T, xs)
    end
    sx = isnothing(xvar) ? xs : xvar
    scsr = (isnothing(sx) || sx.rhs) ? csr : CsrStructure(backend, sx)
    sys = LinearSystem(backend, T, scsr)
    packages = BoundaryPackage[BoundaryPackage(backend, T, b, conn, spec.dis; icelltype = setup.properties.icelltype) for b in spec.boundaries]
    sto = isnothing(spec.sto) ? nothing : StoragePackage(backend, T, spec.sto, conn, nper; r = setup.r, newton, dis = spec.dis)
    tvk = tvk_state(spec, conn, setup)
    z(k = conn.nodes) = KernelAbstractions.zeros(backend, T, k)
    ibh = ones(Int32, conn.nodes)
    h = initial_heads(backend, T, spec.ic, conn, setup.r)
    hold = _device_copy(backend, Array(h))
    # The Newton formulation does not dry or rewet cells (`npf_cf`, `preprocess_input`).
    wd = (isnothing(npf.cv) || newton) ? nothing : WetDry(backend, T, setup.properties, spec.npf, conn.nodes)
    nurbot = newton && spec.newton_under_relaxation ? _device_copy(backend, T.(newton_ur_bottoms(conn, setup))) : z(0)
    # gwf_ptc: `get_cell_volume(n, top(n))`, the volume at full saturation.
    vol = newton ? _device_copy(backend, T.(conn.area .* (conn.top .- conn.bot))) : z(0)
    m = GwfModel(
        spec, backend, conn, setup, csr, npf, sys, packages, sto, _device_copy(backend, T.(conn.area)),
        _device_copy(backend, ibh), ibh, h, hold, z(), z(), KernelAbstractions.zeros(backend, T, conn.nja), 0,
        nothing, 0, wd, newton, nurbot, vol, 0.0, xt3d, scsr, nothing, nothing, Tuple{Int, Vector{Int32}, Vector{Int32}}[], hfb, tvk,
        xvar,
    )
    # `preprocess_input`: `wd(0, hnew)` with the starting heads (IBOUND without the
    # constant heads, which `chd_rp` sets later).
    isnothing(wd) || wetdry!(m, 0)
    return m
end

"""
    newton_ur_bottoms(conn::Connectivity, setup::NpfSetup) -> Vector{Float64}

The bottom NEWTON UNDER_RELAXATION uses (gwf-npf.f90 `npf_nur`, with the `ibotnode` search
of `preprocess_input`) per reduced node, as an offset from the node's reference head: the
bottom of the cell reached by following, from the node, the vertical connection to the
higher-numbered cell with the lowest bottom below the current minimum, as long as there is
one. `-Inf` (never applied) unless the node and that cell are convertible. Host, Float64.
"""
function newton_ur_bottoms(conn::Connectivity, setup::NpfSetup)
    ict = setup.properties.icelltype
    out = fill(-Inf, Int(conn.nodes))
    for n in 1:conn.nodes
        minbot = conn.bot[n]
        nn = n
        while true
            nextn = 0
            for ii in (conn.ia[nn] + 1):(conn.ia[nn + 1] - 1)
                m = conn.ja[ii]
                if conn.ihc[conn.jas[ii]] == 0 && m > nn && conn.bot[m] < minbot
                    nextn = m
                    minbot = conn.bot[m]
                end
            end
            nextn > 0 || break
            nn = nextn
        end
        (ict[n] > 0 && ict[nn] > 0) && (out[n] = conn.bot[nn] - setup.r[n])
    end
    return out
end

# Adapt, e.g. `adapt(backend, m)`: every device array moves (and `backend` becomes the
# backend of the adapted heads); the host setup is shared with `m`.
function Adapt.adapt_structure(to, m::GwfModel)
    h = adapt(to, m.h)
    csr = adapt(to, m.csr)
    scsr = m.scsr === m.csr ? csr : adapt(to, m.scsr)
    return GwfModel(
        m.spec, get_backend(h), m.conn, m.setup, csr, adapt(to, m.npf), adapt(to, m.sys),
        BoundaryPackage[adapt(to, p) for p in m.packages], adapt(to, m.sto), adapt(to, m.area), adapt(to, m.ibound),
        copy(m.ibound_host), h, adapt(to, m.hold), adapt(to, m.res), adapt(to, m.δ), adapt(to, m.flowja), m.kper,
        nothing, 0, adapt(to, m.wd), m.newton, adapt(to, m.nurbot), adapt(to, m.vol), m.delt, adapt(to, m.xt3d), scsr,
        m.solution, adapt(to, m.exgentry), copy(m.conversions), isnothing(m.hfb) ? nothing : copy(m.hfb),
        isnothing(m.tvk) ? nothing : copy(m.tvk), deepcopy(m.xvar),
    )
end

# T6.6: the merged model of a multi-model solution (src/gwf/exchange.jl): the merged
# connectivity and NPF setup, every model's stress packages (cells in the model's user
# numbering, `uoffset`), the merged storage and starting heads. Confined models only
# (`solution_setup` checks), so there is no convertible or Newton data.
function GwfModel(backend::Backend, ::Type{T}, sol::SolutionSetup, nper::Integer) where {T <: AbstractFloat}
    conn = sol.conn
    setup = sol.setup
    csr = CsrStructure(backend, conn)
    npf = NpfModel(backend, T, setup)
    xs = sol.xt3d
    xt3d = isnothing(xs) ? nothing : Xt3dNpf(backend, T, xs)
    scsr = (isnothing(xs) || xs.rhs) ? csr : CsrStructure(backend, xs)
    sys = LinearSystem(backend, T, scsr)
    ict = setup.properties.icelltype
    packages = BoundaryPackage[
        BoundaryPackage(backend, T, b, conn, part.spec.dis; icelltype = ict, uoffset = part.uoffset)
            for part in sol.models for b in part.spec.boundaries
    ]
    sto = isnothing(sol.sto) ? nothing : StoragePackage(backend, T, sol.sto, conn, nper; r = setup.r)
    z(k = conn.nodes) = KernelAbstractions.zeros(backend, T, k)
    ibh = ones(Int32, conn.nodes)
    h = initial_heads(backend, T, sol.ic, conn, setup.r)
    hold = _device_copy(backend, Array(h))
    spec = first(sol.models).spec
    return GwfModel(
        spec, backend, conn, setup, csr, npf, sys, packages, sto, _device_copy(backend, T.(conn.area)),
        _device_copy(backend, ibh), ibh, h, hold, z(), z(), KernelAbstractions.zeros(backend, T, conn.nja), 0,
        nothing, 0, nothing, false, z(0), z(0), 0.0, xt3d, scsr, sol, _device_copy(backend, sol.exgentry),
        Tuple{Int, Vector{Int32}, Vector{Int32}}[], nothing, nothing, nothing,
    )
end


Base.eltype(::GwfModel{T}) where {T} = T
KernelAbstractions.get_backend(m::GwfModel) = m.backend
function Base.show(io::IO, m::GwfModel{T}) where {T}
    name = isnothing(m.solution) ? m.spec.name : join((p.spec.name for p in m.solution.models), "+")
    return print(io, "GwfModel{", T, "}(\"", name, "\", ", m.conn.nodes, " cells, ", length(m.packages), " stress package(s))")
end

"""
    apply_period!(m::GwfModel, kper; window = nothing) -> m

Start stress period `kper` (`bnd_rp`, `sto_rp`, `chd_rp`), once per period: the period
data of every stress package (host, Float64, then copied), the storage setting, the
barriers of HFB6 (T6.11, `hfb_rp!`) and the IBOUND of the constant heads (copied to the
device when it changed). `window = (t0, t1)`
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
    # T6.12: TVK6 and TVS6 without time series: the block of the period (`npf_rp` → `npf_ad`,
    # `sto_rp`); new saturated conductances make the off-diagonals stale.
    (!isnothing(m.tvk) && tvk_rp!(m.tvk, m.npf, kper; window)) && (m.offdiag = nothing)
    _xvar_tvk!(m)
    isnothing(m.sto) || tvs_rp!(m.sto, kper; window)
    # T6.11: HFB6 after the stress packages (`gwf_rp`: npf, hfb, oc, sto, then the packages;
    # independent of each other). Changed saturated conductances make the off-diagonals stale.
    if !isnothing(m.hfb)
        old = copy(m.hfb.xjs)
        if hfb_rp!(m.hfb, m.npf, m.conn, m.setup, kper)
            m.offdiag = nothing
            # T6.14: barriers on the XT3D coefficients (`xt3d_fhfb`).
            m.hfb.xt3d && xvar_hfb!(m.xvar, m.xt3d, old, m.hfb.xjs, m.hfb.xchr)
        end
    end
    if chd
        # chd_rp: the cells of the previous constant heads become active again, dry cells
        # stay dry (T6.1), the new constant-head cells get −1.
        ib = boundary_ibound(m.conn, m.packages, m.ibound_host)
        if ib != m.ibound_host
            m.ibound_host = ib
            copyto!(m.ibound, ib)
            m.offdiag = nothing
            _xvar_ibound!(m)
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
evaluated for `window` by `apply_period!`, are skipped. T6.12: TVK6 and TVS6 with a TS6
file are applied for the step.
"""
function advance_time_series!(m::GwfModel, window::NTuple{2, Float64})
    # T6.12: TVK6 and TVS6 with time series (`npf_ad`, `sto_ad` → TvBase.f90 `ad`).
    (!isnothing(m.tvk) && tvk_ad!(m.tvk, m.npf, window)) && (m.offdiag = nothing)
    _xvar_tvk!(m)
    isnothing(m.sto) || tvs_ad!(m.sto, window)
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
    m.delt = delt
    empty!(m.conversions)
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
    formulate!(m::GwfModel; symmetric = false, terms = true, scale = nothing, jac = m.newton) -> m

Formulate and assemble the system of an outer iteration from the current offset heads
(`bnd_cf`/`bnd_fc` of every stress package except CHD, `sto_fc`, `npf_fc`, `sln_ls`), then
the flux-form residual `m.res` (PLAN.md section 3.7, rule 2; with `scale`, also the
row sums `Σ |terms|`, see `flux_residual!`). Device work only: `formulate_system!` (T4.5;
since T4.7 it also formulates the package records), which rewrites the off-diagonal entries of the
matrix only when `ibound` or `symmetric` changed since they were last written.
`terms = false` (the outer iterations, T4.7) leaves `sys.rhs`, `sys.bdiag` and `sys.brhs`
unwritten (see `formulate_system!`). With the Newton formulation (T6.2) `jac` adds the
Newton–Raphson terms (`sln_buildsystem(kiter, inewton = 1)`; `jac = false` is the system
backtracking evaluates, `inewton = 0`).
"""
function formulate!(m::GwfModel; symmetric::Bool = false, terms::Bool = true, scale = nothing, jac::Bool = m.newton)
    # The package records are formulated in the row kernel (T4.7; `formulate!` per package
    # did it in a kernel of its own). Records without a cell keep their `hcof`/`rhs`; no
    # reader uses them (`calc_simvals!` skips them).
    pkgs = Tuple(
        (record_kind(p, m.npf.cv; newton = m.newton), p.bound, m.area, p.hcof, p.rhs, p.nodeptr, p.noderec)
            for p in m.packages if boundary_kind(p) !== :CHD && p.nbound > 0
    )
    storage = nothing
    if !isnothing(m.sto) && m.sto.transient
        _check_timestep(m.sto)
        storage = storage_args(m.sto, m.hold)
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
    # T6.14: on the head-dependent XT3D/GNC path the matrix is assembled every time.
    isnothing(m.xvar) || (m.offdiag = nothing)
    offdiag = m.offdiag !== symmetric
    offdiag && (m.offdiag_id = Threads.atomic_add!(_OFFDIAG_IDS, 1) + 1)
    if isnothing(m.xt3d)
        formulate_system!(m.sys, m.res, m.csr, m.npf, m.h, m.ibound, pkgs, storage; symmetric, offdiag, terms, scale, jac = jac && m.newton)
    elseif !isnothing(m.xvar)
        xvar_formulate_system!(m.sys, m.res, m.csr, m.scsr, m.xt3d, m.npf, m.h, m.ibound, pkgs, storage; offdiag, terms, scale, jac = jac && m.newton)
    else
        # T6.5: XT3D (no CG, no Newton: `initialize`, `check_xt3d`).
        xt3d_formulate_system!(m.sys, m.res, m.csr, m.scsr, m.xt3d, m.h, m.ibound, pkgs, storage; offdiag, terms, scale)
    end
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
    changed, chddry, flags = wetdry_device!(m.wd, m.npf.cv, m.csr, m.npf.Δr, m.h, m.ibound, kiter)
    chddry && throw(MF6InputError(m.spec.npf.file, 0, "CONSTANT-HEAD CELL WENT DRY -- SIMULATION ABORTED"))
    changed || return false
    old = copy(m.ibound_host)
    copyto!(m.ibound_host, m.ibound)
    # T6.9 (`sgwf_npf_wdmsg`): the cells rewetted (in node order), then the cells dried,
    # among them those rewetted and dried again in this call.
    wet = isnothing(flags) ? Int32[] : Int32.(findall(!=(0), Array(flags)))
    dry = Int32[n for n in eachindex(old) if m.ibound_host[n] == 0 && (old[n] != 0 || insorted(n, wet))]
    (isempty(wet) && isempty(dry)) || push!(m.conversions, (Int(kiter), wet, dry))
    m.offdiag = nothing
    for pkg in m.packages
        boundary_kind(pkg) in (:RCH, :EVT) && move_to_highest_active!(pkg, m.ibound_host, m.setup.r)
    end
    _xvar_ibound!(m)
    return true
end

# T6.14: the NPF properties in force (TVK6 changes K).
_current_properties(m::GwfModel) = isnothing(m.tvk) ? m.setup.properties : m.tvk.props

# T6.14: XT3D terms after IBOUND changed (`xt3d_load`: inactive neighbours leave a stencil).
function _xvar_ibound!(m::GwfModel)
    isnothing(m.xvar) && return nothing
    xvar_ibound!(m.xvar, m.xt3d, m.conn, _current_properties(m), m.ibound_host) && (m.offdiag = nothing)
    return nothing
end

# T6.14: XT3D terms of the cells TVK6 changed (`npf_ad` → `xt3d_fcpc`).
function _xvar_tvk!(m::GwfModel)
    isnothing(m.tvk) && return nothing
    if !isnothing(m.xvar) && !isempty(m.tvk.nodes)
        xvar_recompute!(m.xvar, m.xt3d, m.conn, m.tvk.props, unique(m.tvk.nodes)) && (m.offdiag = nothing)
    end
    empty!(m.tvk.nodes)
    return nothing
end

# T6.6: a single model is its own view (`ModelView`, src/gwf/exchange.jl).
_vspec(m::GwfModel) = m.spec
_vconn(m::GwfModel) = m.conn
_vrange(m::GwfModel) = 1:Int(m.conn.nodes)
_voffset(::GwfModel) = 0
_vpackages(m::GwfModel) = eachindex(m.packages)
_vmodel(m::GwfModel) = m
_vsetup(m::GwfModel) = m.setup
_vname(m::GwfModel) = m.spec.name

_vspec(v::ModelView) = v.part.spec
_vconn(v::ModelView) = v.part.conn
_vrange(v::ModelView) = (Int(v.part.offset) + 1):(Int(v.part.offset) + Int(v.part.conn.nodes))
_voffset(v::ModelView) = Int(v.part.offset)
_vpackages(v::ModelView) = v.part.packages
_vmodel(v::ModelView) = v.model
_vsetup(v::ModelView) = v.part.setup
_vname(v::ModelView) = v.part.spec.name
Base.eltype(v::ModelView) = eltype(v.model)
KernelAbstractions.get_backend(v::ModelView) = v.model.backend

"""
    model_views(m::GwfModel) -> Vector

The models of `m` (T6.6): `[m]` for a single model, the `ModelView`s of the merged model of
a multi-model solution.
"""
model_views(m::GwfModel) = isnothing(m.solution) ? Any[m] : Any[ModelView(m, k, p) for (k, p) in enumerate(m.solution.models)]

"""
    absolute_heads(v::ModelView) -> Vector{Float64}

The heads of one model of a multi-model solution (its reduced nodes; confined, so no dry
cells), `r + h'` on the host in Float64.
"""
function absolute_heads(v::ModelView)
    rg = _vrange(v)
    return v.part.setup.r .+ Float64.(Array(view(v.model.h, rg)))
end
