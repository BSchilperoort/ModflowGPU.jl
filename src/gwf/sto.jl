# Storage (gwf-sto.f90, GwfStorageUtils.f90).
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
# length (`set_timestep!`).
#
# Convertible cells (ICONVERT /= 0, T6.1): `SsTerms` with the saturations `sn` of the old
# and new heads (`sQuadraticSaturation` with SATOMEGA 0, i.e. clamped linear) and, with
# SS_CONFINED_ONLY, the confined-only variant; `SyTerms` with `rho2 = sy · area / delt`
# (STO-SY). In offsets the right-hand sides read `rhsterm − aterm·r`, which is the same
# formula with `top`, `bot`, `hold` and the mid-saturation elevations `z` as offsets from
# the cell reference (`top' = top − r`, `bot' = bot − r`, computed on the host in Float64).
# TVS (`integratechanges`) and the developer option of the original SS formulation are
# not supported.

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

Throw `MF6UnsupportedError` for storage the engine does not support yet: TVS6 (T6.x).
Convertible cells are supported since T6.1.
"""
function check_storage_spec(sto::StoSpec, conn::Connectivity, nper::Integer)
    isempty(sto.tvs_files) ||
        throw(MF6UnsupportedError(sto.file, "unsupported: TVS6 time-varying storage in STO6 (not implemented)"))
    return nothing
end

"""
    ConvertibleStorage{T}

Device data of the convertible cells of STO6 (T6.1): `iconvert` (`Int32` by node),
`topoff`, `botoff` (`top − r`, `bot − r`), `thk` (`top − bot`), `rho2` (`sy · area / delt`,
set by `set_timestep!`) and the option `iconf_ss` (SS_CONFINED_ONLY).
"""
struct ConvertibleStorage{T <: AbstractFloat, V <: AbstractVector{T}, VI <: AbstractVector{Int32}}
    iconvert::VI
    topoff::V
    botoff::V
    thk::V
    rho2::V
    iconf_ss::Bool
end

Adapt.@adapt_structure ConvertibleStorage

"""
    StoragePackage{T}(backend, sto::StoSpec, conn::Connectivity, nper; r = nothing)
    StoragePackage(backend, T, sto, conn, nper; r = nothing)

Storage (STO6) on the device in precision `T`, after `check_storage_spec`.

Device arrays by reduced node: `rho` (`SsCapacity / delt`, computed in Float64 and cast to
`T` by `set_timestep!`), `strgss` (STO-SS rate per cell, `calc_storage_rates!`) and
`strgsy` (STO-SY, zeros without convertible cells); `conv`, a `ConvertibleStorage` when a
cell has ICONVERT /= 0 (needs the reference heads `r`), else `nothing`.
Host fields: `spec`, `conn`, `sc` (`storage_capacity`, Float64), `sc2` (`sy · area`, empty
without convertible cells), `delt` (time step length of `rho`, 0 before the first
`set_timestep!`), `kper` (period of the last `apply_period!`, 0 before) and `transient`
(whether period `kper` is transient; `sto_rp`'s `iss == 0`).
"""
mutable struct StoragePackage{T <: AbstractFloat, VT <: AbstractVector{T}, CS}
    const spec::StoSpec
    const conn::Connectivity
    const sc::Vector{Float64}
    const sc2::Vector{Float64}
    const rho::VT
    const strgss::VT
    const strgsy::VT
    const conv::CS
    delt::Float64
    kper::Int
    transient::Bool
end

function StoragePackage{T}(
        backend::Backend, sto::StoSpec, conn::Connectivity, nper::Integer; r::Union{Nothing, AbstractVector{Float64}} = nothing,
    ) where {T <: AbstractFloat}
    check_storage_spec(sto, conn, nper)
    z(k) = KernelAbstractions.zeros(backend, T, k)
    nodes = Int(conn.nodes)
    iconvert = Int32[sto.iconvert[conn.nodeuser[n]] != 0 ? 1 : 0 for n in 1:nodes]
    conv = nothing
    sc2 = Float64[]
    if any(!=(0), iconvert)
        isnothing(r) && throw(ArgumentError("StoragePackage: convertible cells (ICONVERT /= 0) need the reference heads `r`"))
        isnothing(sto.sy) && throw(MF6InputError(sto.file, 0, "Error in GRIDDATA block: SY not found."))
        top, bot = conn.top, conn.bot
        dev(x) = _device_copy(backend, x)
        conv = ConvertibleStorage(
            dev(iconvert), dev(T.(top .- r)), dev(T.(bot .- r)), dev(T.(top .- bot)), z(nodes), sto.ss_confined_only,
        )
        sc2 = Float64[sto.sy[conn.nodeuser[n]] * conn.area[n] for n in 1:nodes]  # SyCapacity
    end
    return StoragePackage(sto, conn, storage_capacity(sto, conn), sc2, z(nodes), z(nodes), z(nodes), conv, 0.0, 0, false)
end

StoragePackage(backend::Backend, ::Type{T}, sto::StoSpec, conn::Connectivity, nper::Integer; kw...) where {T} =
    StoragePackage{T}(backend, sto, conn, nper; kw...)

# Adapt: the device arrays move, the host fields are kept.
Adapt.adapt_structure(to, s::StoragePackage) = StoragePackage(
    s.spec, s.conn, s.sc, s.sc2, adapt(to, s.rho), adapt(to, s.strgss), adapt(to, s.strgsy), adapt(to, s.conv), s.delt, s.kper,
    s.transient,
)

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
    isnothing(sto.conv) || copyto!(sto.conv.rho2, T[T(s * tled) for s in sto.sc2])  # SyCapacity · tled
    sto.delt = delt
    return sto
end

function _check_timestep(sto::StoragePackage)
    sto.delt > 0 || throw(ArgumentError("StoragePackage: call set_timestep! before formulating storage"))
    return nothing
end

# Saturation of a convertible storage cell (`sQuadraticSaturation` with SATOMEGA 0): 0 below
# the bottom, 1 above the top, `(h − bot)/(top − bot)` in between; offsets. Device-safe.
@inline function storage_saturation(h::T, topoff::T, botoff::T, thk::T) where {T}
    thk > zero(T) || return h < botoff ? zero(T) : one(T)
    h < botoff && return zero(T)
    h > topoff && return one(T)
    return (h - botoff) / thk
end

"""
    storage_terms(rho, hold, conv, h, n) -> (ss_aterm, ss_rhs, sy_aterm, sy_rhs)

The storage terms of active cell `n` (`fc_default_sto`: `SsTerms`, and `SyTerms` for a
convertible cell), with the right-hand sides for offset heads (see the header of
src/gwf/sto.jl). `conv` is `nothing` (confined storage) or a `ConvertibleStorage`.
Device-safe.
"""
@inline function storage_terms(rho, hold, ::Nothing, h, n)
    @inbounds r = rho[n]
    T = eltype(rho)
    return -r, -r * hold[n], zero(T), zero(T)
end

@inline function storage_terms(rho, hold, cs::ConvertibleStorage{T}, h, n) where {T}
    @inbounds begin
        r = rho[n]
        cs.iconvert[n] == Int32(0) && return -r, -r * hold[n], zero(T), zero(T)
        tp, bt, tthk = cs.topoff[n], cs.botoff[n], cs.thk[n]
        ho = hold[n]
        snold = storage_saturation(ho, tp, bt, tthk)
        snnew = storage_saturation(h[n], tp, bt, tthk)
        # SsTerms (iorig_ss = 0; rho1old = rho1)
        aterm = -r * snnew
        if !cs.iconf_ss
            zold = bt + T(0.5) * tthk * snold
            znew = bt + T(0.5) * tthk * snnew
            rhsterm = -r * snold * (ho - zold) - r * snnew * znew
        else
            rhsterm = zero(T)
            snold == one(T) && (rhsterm = rhsterm - r * (ho - tp))
            if snnew == one(T)
                rhsterm = rhsterm - r * tp
            else
                aterm = zero(T)
            end
        end
        # SyTerms (rho2old = rho2)
        rho2 = cs.rho2[n]
        if snnew < one(T)
            if snnew > zero(T)
                ya = -rho2
                yr = -rho2 * tthk * snold - rho2 * bt
            else
                ya = zero(T)
                yr = tthk * (zero(T) - rho2 * snold)
            end
        else
            ya = zero(T)
            yr = tthk * (rho2 * snnew - rho2 * snold)
        end
        return aterm, rhsterm, ya, yr
    end
end

@kernel function storage_add_kernel!(amat, b, @Const(idiag), @Const(rho), @Const(hold), @Const(ibound), conv, @Const(h))
    n = @index(Global)
    @inbounds begin
        # fc_default_sto / SsTerms (ICONVERT 0): aterm = −rho, rhsterm = −rho·hold.
        if ibound[n] > 0
            a1, r1, a2, r2 = storage_terms(rho, hold, conv, h, n)
            id = idiag[n]
            # MF6 adds the SS terms, then the SY terms.
            d = amat[id] + a1
            s = b[n] + r1
            if conv !== nothing
                d += a2
                s += r2
            end
            amat[id] = d
            b[n] = s
        end
    end
end

"""
    add_to_system!(amat, b, idiag, sto::StoragePackage, hold, ibound; h = hold) -> amat

Add the storage terms of a transient period to the diagonal `amat[idiag[n]]` and to `b`
(`sto_fc`, `fc_default_sto`): `−rho[n]` and `−rho[n]·hold[n]` for every confined cell with
`ibound[n] > 0`, the `SsTerms` and `SyTerms` of convertible cells with the current offset
heads `h` (T6.1), one work item per node. `hold` holds the offset heads at the start of
the time step. Nothing is added in a steady-state period.
"""
function add_to_system!(amat, b, idiag, sto::StoragePackage, hold, ibound; h = hold)
    sto.transient || return amat
    _check_timestep(sto)
    backend = get_backend(b)
    storage_add_kernel!(backend, _WG)(amat, b, idiag, sto.rho, hold, ibound, sto.conv, h; ndrange = sto.conn.nodes)
    return amat
end

@kernel function storage_rate_kernel!(strgss, strgsy, @Const(rho), @Const(hnew), @Const(hold), @Const(ibound), conv)
    n = @index(Global)
    T = eltype(strgss)
    @inbounds begin
        q = zero(T)
        qy = zero(T)
        # cq_default_sto / SsTerms: rate = aterm·hnew − rhsterm = rho·(hold − hnew).
        if ibound[n] > 0
            if conv === nothing || conv.iconvert[n] == Int32(0)
                q = rho[n] * (hold[n] - hnew[n])
            else
                a1, r1, _, _ = storage_terms(rho, hold, conv, hnew, n)
                q = a1 * hnew[n] - r1
                # SyTerms rate: rho2·thk·snold − rho2·thk·snnew
                tp, bt, tthk = conv.topoff[n], conv.botoff[n], conv.thk[n]
                rho2 = conv.rho2[n]
                snold = storage_saturation(hold[n], tp, bt, tthk)
                snnew = storage_saturation(hnew[n], tp, bt, tthk)
                qy = rho2 * tthk * snold - rho2 * tthk * snnew
            end
        end
        strgss[n] = q
        strgsy[n] = qy
    end
end

"""
    calc_storage_rates!(sto::StoragePackage, hnew, hold, ibound) -> sto

The STO-SS and STO-SY rates per cell into the model (`sto_cq`, `cq_default_sto`):
`strgss[n] = rho[n]·(hold[n] − hnew[n])` for confined cells, `aterm·h − rhsterm` of `SsTerms`
and the `SyTerms` rate (`strgsy`) for convertible ones; zero for cells with `ibound[n] <= 0`
and in a steady-state period. Positive rates are water released from storage.
"""
function calc_storage_rates!(sto::StoragePackage, hnew, hold, ibound)
    if !sto.transient
        fill!(sto.strgss, zero(eltype(sto)))
        fill!(sto.strgsy, zero(eltype(sto)))
        return sto
    end
    _check_timestep(sto)
    backend = get_backend(sto.strgss)
    storage_rate_kernel!(backend, _WG)(sto.strgss, sto.strgsy, sto.rho, hnew, hold, ibound, sto.conv; ndrange = sto.conn.nodes)
    return sto
end

@kernel function storage_flowja_kernel!(flowja, @Const(idiag), @Const(strgss), @Const(strgsy), conv::Bool)
    n = @index(Global)
    @inbounds if conv
        flowja[idiag[n]] = flowja[idiag[n]] + strgss[n] + strgsy[n]
    else
        flowja[idiag[n]] = flowja[idiag[n]] + strgss[n]
    end
end

"""
    add_storage_flowja!(flowja, idiag, sto::StoragePackage) -> flowja

Add the storage rates (STO-SS, and STO-SY with convertible cells) to the diagonal positions
`idiag[n]` (`ia[n]`) of `flowja` (`cq_default_sto`), after `calc_storage_rates!`.
"""
function add_storage_flowja!(flowja, idiag, sto::StoragePackage)
    backend = get_backend(flowja)
    storage_flowja_kernel!(backend, _WG)(flowja, idiag, sto.strgss, sto.strgsy, !isnothing(sto.conv); ndrange = sto.conn.nodes)
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
