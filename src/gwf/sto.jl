# Storage (gwf-sto.f90, GwfStorageUtils.f90), confined cells only.
#
# For a confined cell (ICONVERT 0) MF6's default storage formulation reduces to
# (`fc_default_sto`, `SsTerms` with snnew = snold = 1):
#
#     sc1 = SsCapacity = ss · (top − bot) · area       (STORAGECOEFFICIENT: ss · area)
#     rho = sc1 · (1 / delt)
#     amat[n,n] += −rho,   rhs[n] += −rho · hold[n]
#     rate = aterm · hnew − rhsterm = rho · (hold − hnew)   (`cq_default_sto`, STO-SS)
#
# Only the head difference enters, so heads as offsets from the cell references (PLAN.md
# section 3.7, rule 1) carry over: `rhs` holds `−rho·h'old`. `rho` is computed on the host
# in Float64 (`delt` stays Float64 on the host, rule 4) and cast to `T` once per time step
# length (`set_timestep!`). SS_CONFINED_ONLY and the original-SS option only change the
# terms of convertible cells, so with confined cells they have no effect. Convertible
# storage (ICONVERT /= 0 in a transient period: SY, `sQuadraticSaturation`) comes with
# T6.1 and is rejected here; TVS (`integratechanges`) is rejected as well.

"""
    storage_capacity(sto::StoSpec, conn::Connectivity) -> Vector{Float64}

The primary storage capacity per reduced node (`SsCapacity`, GwfStorageUtils.f90):
`ss · (top − bot) · area`, or `ss · area` with STORAGECOEFFICIENT, in Float64.
"""
function storage_capacity(sto::StoSpec, conn::Connectivity)
    length(sto.ss) == conn.nodesuser ||
        throw(ArgumentError("storage_capacity: length(ss) = $(length(sto.ss)), expected $(conn.nodesuser) user nodes"))
    sc = Vector{Float64}(undef, conn.nodes)
    for n in 1:conn.nodes
        thick = sto.storagecoefficient ? 1.0 : conn.top[n] - conn.bot[n]
        sc[n] = sto.ss[conn.nodeuser[n]] * thick * conn.area[n]
    end
    return sc
end

"""
    check_storage_spec(sto::StoSpec, conn::Connectivity, nper)

Throw `MF6UnsupportedError` for storage the engine does not support yet: convertible
cells (ICONVERT /= 0 in a reduced node) when any of the `nper` stress periods is
transient (T6.1), and TVS6 (T6.x). Without transient periods convertible cells have no
storage term, so they are accepted.
"""
function check_storage_spec(sto::StoSpec, conn::Connectivity, nper::Integer)
    isempty(sto.tvs_files) ||
        throw(MF6UnsupportedError(sto.file, "unsupported: TVS6 time-varying storage in STO6 (not implemented)"))
    if any(k -> is_transient(sto, k), 1:nper)
        for n in 1:conn.nodes
            nu = conn.nodeuser[n]
            sto.iconvert[nu] != 0 && throw(
                MF6UnsupportedError(
                    sto.file,
                    "unsupported: convertible storage (STO6 ICONVERT = $(sto.iconvert[nu]) in user node $nu with a transient " *
                        "stress period); convertible cells arrive with T6.1",
                ),
            )
        end
    end
    return nothing
end

"""
    StoragePackage{T}(backend, sto::StoSpec, conn::Connectivity, nper)
    StoragePackage(backend, T, sto, conn, nper)

Confined storage (STO6) on the device in precision `T`, after `check_storage_spec`.

Device arrays by reduced node: `rho` (`SsCapacity / delt`, computed in Float64 and cast to
`T` by `set_timestep!`) and `strgss` (STO-SS rate per cell, `calc_storage_rates!`).
Host fields: `spec`, `conn`, `sc` (`storage_capacity`, Float64), `delt` (time step length
of `rho`, 0 before the first `set_timestep!`), `kper` (period of the last
`apply_period!`, 0 before) and `transient` (whether period `kper` is transient; `sto_rp`'s
`iss == 0`).
"""
mutable struct StoragePackage{T <: AbstractFloat, VT <: AbstractVector{T}}
    const spec::StoSpec
    const conn::Connectivity
    const sc::Vector{Float64}
    const rho::VT
    const strgss::VT
    delt::Float64
    kper::Int
    transient::Bool
end

function StoragePackage{T}(backend::Backend, sto::StoSpec, conn::Connectivity, nper::Integer) where {T <: AbstractFloat}
    check_storage_spec(sto, conn, nper)
    z() = KernelAbstractions.zeros(backend, T, conn.nodes)
    return StoragePackage(sto, conn, storage_capacity(sto, conn), z(), z(), 0.0, 0, false)
end

StoragePackage(backend::Backend, ::Type{T}, sto::StoSpec, conn::Connectivity, nper::Integer) where {T} =
    StoragePackage{T}(backend, sto, conn, nper)

# Adapt: `rho` and `strgss` move, the host fields are kept.
Adapt.adapt_structure(to, s::StoragePackage) =
    StoragePackage(s.spec, s.conn, s.sc, adapt(to, s.rho), adapt(to, s.strgss), s.delt, s.kper, s.transient)

Base.eltype(::StoragePackage{T}) where {T} = T
Base.show(io::IO, s::StoragePackage{T}) where {T} =
    print(io, "StoragePackage{", T, "}(", s.conn.nodes, " cells, period ", s.kper, s.transient ? " transient" : " steady state", ")")

"""
    apply_period!(sto::StoragePackage, kper) -> Bool

Set whether stress period `kper` is transient (`sto_rp`: the last STO PERIOD block at or
before `kper`; transient before the first block, `is_transient`). Returns whether the
setting changed.
"""
function apply_period!(sto::StoragePackage, kper::Integer)
    tr = is_transient(sto.spec, kper)
    changed = sto.kper == 0 || tr != sto.transient
    sto.kper = kper
    sto.transient = tr
    return changed
end

"""
    set_timestep!(sto::StoragePackage, delt) -> sto

Set the time step length `delt` (Float64, host): recompute `rho = sc · (1 / delt)` in
Float64 (`fc_default_sto`: `tled = DONE / delt`, `rho1 = sc1 * tled`), cast to `T` and copy
to the device. Nothing is done when `delt` did not change. A zero `delt` is an error in a
transient period (`sto_fc`).
"""
function set_timestep!(sto::StoragePackage{T}, delt::Real) where {T}
    delt = Float64(delt)
    if delt == 0
        sto.transient &&
            throw(ArgumentError("DETECTED TIME STEP LENGTH OF ZERO.  GWF STORAGE PACKAGE CANNOT BE USED UNLESS DELT IS NON-ZERO."))
        return sto
    end
    delt == sto.delt && return sto
    tled = 1.0 / delt
    copyto!(sto.rho, T[T(s * tled) for s in sto.sc])
    sto.delt = delt
    return sto
end

function _check_timestep(sto::StoragePackage)
    sto.delt > 0 || throw(ArgumentError("StoragePackage: call set_timestep! before formulating storage"))
    return nothing
end

@kernel function storage_add_kernel!(amat, b, @Const(idiag), @Const(rho), @Const(hold), @Const(ibound))
    n = @index(Global)
    @inbounds begin
        # fc_default_sto / SsTerms (ICONVERT 0): aterm = −rho, rhsterm = −rho·hold.
        if ibound[n] > 0
            r = rho[n]
            id = idiag[n]
            amat[id] = amat[id] - r
            b[n] = b[n] - r * hold[n]
        end
    end
end

"""
    add_to_system!(amat, b, idiag, sto::StoragePackage, hold, ibound) -> amat

Add the storage terms of a transient period to the diagonal `amat[idiag[n]]` and to `b`
(`sto_fc`, `fc_default_sto`): `−rho[n]` and `−rho[n]·hold[n]` for every cell with
`ibound[n] > 0`, one work item per node. `hold` holds the offset heads at the start of
the time step. Nothing is added in a steady-state period.
"""
function add_to_system!(amat, b, idiag, sto::StoragePackage, hold, ibound)
    sto.transient || return amat
    _check_timestep(sto)
    backend = get_backend(b)
    storage_add_kernel!(backend, _WG)(amat, b, idiag, sto.rho, hold, ibound; ndrange = sto.conn.nodes)
    return amat
end

@kernel function storage_rate_kernel!(strgss, @Const(rho), @Const(hnew), @Const(hold), @Const(ibound))
    n = @index(Global)
    @inbounds begin
        q = zero(eltype(strgss))
        # cq_default_sto / SsTerms: rate = aterm·hnew − rhsterm = rho·(hold − hnew).
        if ibound[n] > 0
            q = rho[n] * (hold[n] - hnew[n])
        end
        strgss[n] = q
    end
end

"""
    calc_storage_rates!(sto::StoragePackage, hnew, hold, ibound) -> sto

The STO-SS rate per cell into the model, `strgss[n] = rho[n]·(hold[n] − hnew[n])`
(`sto_cq`, `cq_default_sto`); zero for cells with `ibound[n] <= 0` and in a steady-state
period. Positive rates are water released from storage.
"""
function calc_storage_rates!(sto::StoragePackage, hnew, hold, ibound)
    if !sto.transient
        fill!(sto.strgss, zero(eltype(sto)))
        return sto
    end
    _check_timestep(sto)
    backend = get_backend(sto.strgss)
    storage_rate_kernel!(backend, _WG)(sto.strgss, sto.rho, hnew, hold, ibound; ndrange = sto.conn.nodes)
    return sto
end

@kernel function storage_flowja_kernel!(flowja, @Const(idiag), @Const(strgss))
    n = @index(Global)
    @inbounds flowja[idiag[n]] = flowja[idiag[n]] + strgss[n]
end

"""
    add_storage_flowja!(flowja, idiag, sto::StoragePackage) -> flowja

Add the storage rates to the diagonal positions `idiag[n]` (`ia[n]`) of `flowja`
(`cq_default_sto`), after `calc_storage_rates!`.
"""
function add_storage_flowja!(flowja, idiag, sto::StoragePackage)
    backend = get_backend(flowja)
    storage_flowja_kernel!(backend, _WG)(flowja, idiag, sto.strgss; ndrange = sto.conn.nodes)
    return flowja
end

"""
    storage_budget(sto::StoragePackage) -> (; rin, rout)

The STO-SS budget entry (`sto_bd`, `rate_accumulator`): the sum of the positive rates
(`rin`) and of the negated negative rates (`rout`) of `strgss`, accumulated in Float64 on
the host (PLAN.md section 3.7, rule 4). The device reduction arrives with the budgets of
T2.8.
"""
function storage_budget(sto::StoragePackage)
    rin = 0.0
    rout = 0.0
    for q in Array(sto.strgss)
        q = Float64(q)
        q < 0 ? (rout -= q) : (rin += q)
    end
    return (; rin, rout)
end
