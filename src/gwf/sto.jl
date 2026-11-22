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
# The developer option of the original SS formulation is not supported.
#
# TVS6 (T6.12; gwf-sto.f90 `save_old_ss_sy`, `fc_default_sto`/`cq_default_sto` with
# `integratechanges`, GwfStorageUtils.f90 `SsTerms`/`SyTerms` with `rho1old`, `rho2old`):
# SS and SY change at the start of a time step (`TvsStorage`, the settings from
# src/gwf/tv.jl); in a step whose values differ from those of the previous step, the old
# capacities enter the terms of the previous heads (`rhoold`, `rho2old`: `StorageChange`).
# For a confined cell `−rho1old·hold` and `rho1·h` no longer cancel the reference head:
# in offsets the right-hand side gains `sdr = (rho1 − rho1old)·r` (computed on the host in
# Float64). For a convertible cell the terms are differences of elevations within the cell
# and need no such term. In the other steps the formulation is the one without TVS
# (bitwise), and so is every step of a model without TVS.
#
# Newton–Raphson (T6.2; `sto_fn`, `fn_default_sto`): the saturations are
# `sQuadraticSaturation` with SATOMEGA 1e-6 (`newton_saturation`), and the Jacobian adds
# `−rho1·S'·(h − bot) + rho1·thk·sn·S'` (SS, unless SS_CONFINED_ONLY) and, for `0 < sn < 1`,
# `rho2 − rho2·thk·S'` (SY) to the diagonal. The SY right-hand side of `fn_default_sto`
# changes the residual too: with the Jacobian the SY term is `rho2·thk·(sn − snold)`, the
# rate of `cq_default_sto` (`storage_terms` returns it as a right-hand side with a zero
# diagonal, and `storage_jacobian` the diagonal), without it (the formulation of
# backtracking, `sln_buildsystem(inewton = 0)`) it is MF6's `SyTerms`. Saturated
# thicknesses `thk·sn` are computed without the division (`newton_satthk`), so the
# differences of nearly full cells do not cancel in Float32.

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

Throw `MF6UnsupportedError` for storage the engine does not support. Convertible cells are
supported since T6.1, TVS6 since T6.12 (a `StoSpec` with `tvs_files` but no `tvs`, read
elsewhere than by `load_simulation`, is rejected).
"""
function check_storage_spec(sto::StoSpec, conn::Connectivity, nper::Integer)
    (isempty(sto.tvs_files) || !isnothing(sto.tvs)) ||
        throw(MF6UnsupportedError(sto.file, "unsupported: TVS6 file names without the TVS6 input (`StoSpec.tvs`)"))
    return nothing
end

"""
    StorageChange{V}

Device data of the storage-change integration of TVS6 in a time step whose storage
properties changed (T6.12): per node `rhoold` (`SsCapacity` of the previous step's SS over
`delt`), `rho2old` (`SyCapacity` of the previous SY over `delt`; empty without convertible
cells) and `sdr` (`(rho1 − rho1old)·r` for confined cells, 0 elsewhere).
"""
struct StorageChange{V <: AbstractVector}
    rhoold::V
    rho2old::V
    sdr::V
end

Adapt.@adapt_structure StorageChange

"""
    TvsStorage

TVS6 of a storage package (T6.12): the settings (`source`, a `TvSource`), the grid
(`dis`, for messages), `integrate` (`integratechanges`: no DISABLE_STORAGE_CHANGE_INTEGRATION),
`iusesy` (a cell is convertible), host Float64 by reduced node: `iconvert`, the current
`ss` and `sy`, the capacities of the previous time step `scprev`, `sc2prev`, the reference
heads `r`; the device `old` (`StorageChange`) and `active` (the current step integrates a
change: the formulation uses `old`).
"""
mutable struct TvsStorage{S <: StorageChange}
    const source::TvSource
    const dis::Union{Nothing, AbstractDisSpec}
    const integrate::Bool
    const iusesy::Bool
    const iconvert::Vector{Int32}
    const ss::Vector{Float64}
    const sy::Vector{Float64}
    const scprev::Vector{Float64}
    const sc2prev::Vector{Float64}
    const r::Vector{Float64}
    const old::S
    active::Bool
end

Adapt.adapt_structure(to, t::TvsStorage) = TvsStorage(
    copy(t.source), t.dis, t.integrate, t.iusesy, t.iconvert, copy(t.ss), copy(t.sy), copy(t.scprev), copy(t.sc2prev), t.r,
    adapt(to, t.old), t.active,
)

"""
    ConvertibleStorage{T}

Device data of the convertible cells of STO6 (T6.1): `iconvert` (`Int32` by node),
`topoff`, `botoff` (`top − r`, `bot − r`), `thk` (`top − bot`), `rho2` (`sy · area / delt`,
set by `set_timestep!`), the option `iconf_ss` (SS_CONFINED_ONLY) and `newton` (T6.2: the
model's NEWTON option; quadratic saturations and the Jacobian terms).
"""
struct ConvertibleStorage{T <: AbstractFloat, V <: AbstractVector{T}, VI <: AbstractVector{Int32}}
    iconvert::VI
    topoff::V
    botoff::V
    thk::V
    rho2::V
    iconf_ss::Bool
    newton::Bool
end

Adapt.@adapt_structure ConvertibleStorage

"""
    StoragePackage{T}(backend, sto::StoSpec, conn::Connectivity, nper; r = nothing)
    StoragePackage(backend, T, sto, conn, nper; r = nothing)

Storage (STO6) on the device in precision `T`, after `check_storage_spec`.

Device arrays by reduced node: `rho` (`SsCapacity / delt`, computed in Float64 and cast to
`T` by `set_timestep!`), `strgss` (STO-SS rate per cell, `calc_storage_rates!`) and
`strgsy` (STO-SY; empty without convertible cells); `conv`, a `ConvertibleStorage` when a
cell has ICONVERT /= 0 (needs the reference heads `r`; `newton`: the model's NEWTON
option, T6.2), else `nothing`.
Host fields: `spec`, `conn`, `sc` (`storage_capacity`, Float64), `sc2` (`sy · area`, empty
without convertible cells), `delt` (time step length of `rho`, 0 before the first
`set_timestep!`), `kper` (period of the last `apply_period!`, 0 before) and `transient`
(whether period `kper` is transient; `sto_rp`'s `iss == 0`). T6.12: `tvs`, the
`TvsStorage` of a package with TVS6 (`sc`, `sc2` then change with SS and SY; needs `r`,
and `dis` for MF6's messages), else `nothing`.
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
    const tvs::Union{Nothing, TvsStorage}
end

function StoragePackage{T}(
        backend::Backend, sto::StoSpec, conn::Connectivity, nper::Integer; r::Union{Nothing, AbstractVector{Float64}} = nothing,
        newton::Bool = false, dis::Union{Nothing, AbstractDisSpec} = nothing,
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
            dev(iconvert), dev(T.(top .- r)), dev(T.(bot .- r)), dev(T.(top .- bot)), z(nodes), sto.ss_confined_only, newton,
        )
        sc2 = Float64[sto.sy[conn.nodeuser[n]] * conn.area[n] for n in 1:nodes]  # SyCapacity
    end
    sc = storage_capacity(sto, conn)
    tvs = nothing
    if !isnothing(sto.tvs)
        isnothing(r) && throw(ArgumentError("StoragePackage: TVS6 needs the reference heads `r`"))
        old = StorageChange(z(nodes), z(isnothing(conv) ? 0 : nodes), z(nodes))
        tvs = TvsStorage(
            TvSource(sto.tvs), dis, !sto.tvs.disable_storage_change_integration, !isnothing(conv), iconvert,
            Float64[sto.ss[conn.nodeuser[n]] for n in 1:nodes],
            isnothing(sto.sy) ? Float64[] : Float64[sto.sy[conn.nodeuser[n]] for n in 1:nodes], copy(sc), copy(sc2), copy(r), old, false,
        )
    end
    return StoragePackage(sto, conn, sc, sc2, z(nodes), z(nodes), z(isnothing(conv) ? 0 : nodes), conv, 0.0, 0, false, tvs)
end

StoragePackage(backend::Backend, ::Type{T}, sto::StoSpec, conn::Connectivity, nper::Integer; kw...) where {T} =
    StoragePackage{T}(backend, sto, conn, nper; kw...)

# Adapt: the device arrays move, the host fields are kept (copied with TVS6, which changes
# `sc` and `sc2`).
Adapt.adapt_structure(to, s::StoragePackage) = StoragePackage(
    s.spec, s.conn, isnothing(s.tvs) ? s.sc : copy(s.sc), isnothing(s.tvs) ? s.sc2 : copy(s.sc2), adapt(to, s.rho),
    adapt(to, s.strgss), adapt(to, s.strgsy), adapt(to, s.conv), s.delt, s.kper, s.transient, adapt(to, s.tvs),
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
        if !isnothing(sto.tvs)
            # T6.12: the step's values become the old ones; `rho` is rebuilt at the next step.
            _tvs_roll!(sto.tvs, sto)
            sto.delt = 0.0
        end
        return sto
    end
    isnothing(sto.tvs) || return _tvs_timestep!(sto, delt)
    delt == sto.delt && return sto
    tled = 1.0 / delt
    copyto!(sto.rho, T[T(s * tled) for s in sto.sc])
    isnothing(sto.conv) || copyto!(sto.conv.rho2, T[T(s * tled) for s in sto.sc2])  # SyCapacity · tled
    sto.delt = delt
    return sto
end

# T6.12: `dst[idx[i]] = val[i]`.
@kernel function tv_scatter_kernel!(dst, @Const(idx), @Const(val))
    i = @index(Global)
    @inbounds dst[idx[i]] = val[i]
end

function _tv_scatter!(dst::AbstractVector{T}, idx::Vector{Int32}, val::Vector{T}) where {T}
    isempty(idx) && return dst
    backend = get_backend(dst)
    tv_scatter_kernel!(backend, _WG)(dst, _device_copy(backend, idx), _device_copy(backend, val); ndrange = length(idx))
    return dst
end

# The values of this step become those of the previous step (`save_old_ss_sy`).
function _tvs_roll!(tv::TvsStorage, sto::StoragePackage)
    copyto!(tv.scprev, sto.sc)
    copyto!(tv.sc2prev, sto.sc2)
    tv.active = false
    return tv
end

# set_timestep! with TVS6 (T6.12): `rho` (and `rho2`) for the current SS and SY (in full when
# `delt` changed, else only the changed cells), and, when SS or SY of a cell differs from
# the previous step and the changes are integrated, the old capacities (`StorageChange`:
# `rhoold` = `rho` but at the changed cells; `sdr` at the changed confined cells).
function _tvs_timestep!(sto::StoragePackage{T}, delt::Float64) where {T}
    tv = sto.tvs
    sc, sc2 = sto.sc, sto.sc2
    tled = 1.0 / delt
    conv = sto.conv
    diff = Int32[n for n in eachindex(sc) if sc[n] !== tv.scprev[n] || (!isempty(sc2) && sc2[n] !== tv.sc2prev[n])]
    if delt != sto.delt
        copyto!(sto.rho, T[T(s * tled) for s in sc])
        isnothing(conv) || copyto!(conv.rho2, T[T(s * tled) for s in sc2])
        sto.delt = delt
    else
        _tv_scatter!(sto.rho, diff, T[T(sc[n] * tled) for n in diff])
        isnothing(conv) || _tv_scatter!(conv.rho2, diff, T[T(sc2[n] * tled) for n in diff])
    end
    tv.active = tv.integrate && !isempty(diff)
    if tv.active
        old = tv.old
        copyto!(old.rhoold, sto.rho)
        _tv_scatter!(old.rhoold, diff, T[T(tv.scprev[n] * tled) for n in diff])
        if !isnothing(conv)
            copyto!(old.rho2old, conv.rho2)
            _tv_scatter!(old.rho2old, diff, T[T(tv.sc2prev[n] * tled) for n in diff])
        end
        fill!(old.sdr, zero(T))
        # Confined cells: `−rho1old·(h'old + r) + rho1·r` in offsets.
        _tv_scatter!(old.sdr, diff, T[tv.iconvert[n] == 0 ? T((sc[n] * tled - tv.scprev[n] * tled) * tv.r[n]) : zero(T) for n in diff])
    end
    copyto!(tv.scprev, sc)
    copyto!(tv.sc2prev, sc2)
    return sto
end

"""
    tvs_sync!(sto::StoragePackage, window) -> sto

Apply the TVS6 settings in force to every tracked cell (TvBase.f90 `sync_node_changes`,
gwf-tvs.f90 `tvs_apply_row_changes`, `tvs_validate_change`; `window` is the time step, for
the time series) and recompute the capacities `sc` (`SsCapacity`) and `sc2` (`SyCapacity`)
of those cells on the host; `set_timestep!` sends them to the device. MF6's errors (a cell
outside the active domain, SS or SY < 0, SY in a model without convertible cells) are one
`MF6InputError`.
"""
function tvs_sync!(sto::StoragePackage, window)
    tv = sto.tvs
    src = tv.source
    conn = sto.conn
    file = isempty(src.spec.file) ? "TVS" : src.spec.file
    cell(n) = isnothing(tv.dis) ? "($(conn.nodeuser[n]))" : _tv_cell(tv.dis, conn.nodeuser[n])
    errors = String[]
    serr(name, n, v) = push!(errors, rstrip(" TVS changed storage property $name is < 0 for cell $(cell(n)) " * _fortran_1pg(v; w = 15, d = 6)))
    for nu in src.tracked
        n = conn.nodereduced[nu]
        ss = tv_value(src, nu, :SS, window)
        sy = tv_value(src, nu, :SY, window)
        if n <= 0
            for v in (ss, sy)
                isnothing(v) || push!(errors, isnothing(tv.dis) ? "CELLID ($nu) is not in the active model domain." : _tv_inactive(tv.dis, nu))
            end
            continue
        end
        if !isnothing(ss)
            tv.ss[n] = ss
            ss < 0.0 && serr("SS", n, ss)
            thick = sto.spec.storagecoefficient ? 1.0 : conn.top[n] - conn.bot[n]
            sto.sc[n] = tv.ss[n] * thick * conn.area[n]  # SsCapacity, as storage_capacity
        end
        if !isnothing(sy)
            if !tv.iusesy
                push!(errors, " TVS cannot change SY for cell $(cell(n)) because SY is unused in this model (all ICONVERT flags are 0).")
            else
                tv.sy[n] = sy
                sy < 0.0 && serr("SY", n, sy)
                sto.sc2[n] = tv.sy[n] * conn.area[n]  # SyCapacity
            end
        end
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    return sto
end

"""
    tvs_rp!(sto::StoragePackage, kper; window = nothing) -> sto

Stress period `kper` starts (gwf-sto.f90 `sto_rp` → TvBase.f90 `rp`): the period's TVS6
block, applied now without a TS6 file. Nothing without TVS6.
"""
function tvs_rp!(sto::StoragePackage, kper::Integer; window = nothing)
    isnothing(sto.tvs) && return sto
    tv_rp!(sto.tvs.source, kper) && tvs_sync!(sto, window)
    return sto
end

"""
    tvs_ad!(sto::StoragePackage, window) -> sto

A time step starts (gwf-sto.f90 `sto_ad` → TvBase.f90 `ad`): with a TS6 file every tracked
cell is applied with the series evaluated over `window`.
"""
function tvs_ad!(sto::StoragePackage, window)
    tv = sto.tvs
    (isnothing(tv) || !tv_ts_active(tv.source) || isempty(tv.source.tracked)) && return sto
    return tvs_sync!(sto, window)
end

"""
    storage_args(sto::StoragePackage, hold) -> Tuple

The storage argument of `formulate_system!` in a transient period: `(rho, hold)`,
`(rho, hold, conv)` with convertible cells, and (T6.12) `(rho, hold, conv, old)` in a time
step whose TVS6 changes are integrated (`conv` may be `nothing`).
"""
function storage_args(sto::StoragePackage, hold)
    (isnothing(sto.tvs) || !sto.tvs.active) || return (sto.rho, hold, sto.conv, sto.tvs.old)
    return isnothing(sto.conv) ? (sto.rho, hold) : (sto.rho, hold, sto.conv)
end

_storage_change(sto::StoragePackage) = (isnothing(sto.tvs) || !sto.tvs.active) ? nothing : sto.tvs.old

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
    newton_satthk(top, bot, thk, x) -> thk·S

The saturated thickness `thk·newton_saturation(top, bot, x)` (`tthk·sn` of
`fn_default_sto`/`SyTerms` with SATOMEGA 1e-6, T6.2), computed without dividing by `thk`
where the saturation is linear or quadratic, so differences of two of them keep the
precision of the head difference. Offsets; device-safe.
"""
@inline function newton_satthk(tp::T, bt::T, thk::T, x::T) where {T}
    teps = T(1.0e-6)
    thk > zero(T) || return x < bt ? zero(T) : thk
    br = x < bt ? zero(T) : (x > tp ? one(T) : (x - bt) / thk)
    av = one(T) / (one(T) - teps)
    if br < teps
        w = x < bt ? zero(T) : x - bt
        return av * T(0.5) * (w * w) / (teps * thk)
    elseif br < one(T) - teps
        return av * (x - bt) + T(0.5) * thk * (one(T) - av)
    elseif br < one(T)
        wi = tp - x
        return thk - av * T(0.5) * (wi * wi) / (teps * thk)
    end
    return thk
end

"""
    storage_terms(rho, hold, conv, h, n[, jac]) -> (ss_aterm, ss_rhs, sy_aterm, sy_rhs)

The storage terms of active cell `n` (`fc_default_sto`: `SsTerms`, and `SyTerms` for a
convertible cell), with the right-hand sides for offset heads (see the header of
src/gwf/sto.jl). `conv` is `nothing` (confined storage) or a `ConvertibleStorage`. With the
Newton formulation (T6.2, `conv.newton`) the saturations are quadratic, and with `jac`
(the Jacobian is added, `storage_jacobian`) the SY terms are those with `fn_default_sto`'s
right-hand side: a zero diagonal and `rho2·(thk·sn − thk·snold)`. Device-safe.
"""
@inline function storage_terms(rho, hold, ::Nothing, h, n, jac::Bool = true)
    @inbounds r = rho[n]
    T = eltype(rho)
    return -r, -r * hold[n], zero(T), zero(T)
end

@inline function storage_terms(rho, hold, cs::ConvertibleStorage{T}, h, n, jac::Bool = true) where {T}
    @inbounds begin
        r = rho[n]
        cs.iconvert[n] == Int32(0) && return -r, -r * hold[n], zero(T), zero(T)
        cs.newton && return _newton_storage_terms(r, hold, cs, h, n, jac)
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
        # SyTerms (rho2old = rho2), with `thk·sn = clamp(h, bot, top) − bot` (T6.1: MF6's
        # `rho2·thk·snold − rho2·bot` etc. cancel in Float32 when sn is near 1; the same terms
        # from the clamped old head do not).
        rho2 = cs.rho2[n]
        hoc = clamp(ho, bt, tp)
        if snnew < one(T)
            if snnew > zero(T)
                ya = -rho2
                yr = -rho2 * hoc
            else
                ya = zero(T)
                yr = -rho2 * (hoc - bt)
            end
        else
            ya = zero(T)
            yr = rho2 * (tp - hoc)
        end
        return aterm, rhsterm, ya, yr
    end
end

# The Newton formulation of `storage_terms` (T6.2): SsTerms with the quadratic saturations;
# SyTerms as MF6's (`jac = false`) or with `fn_default_sto`'s right-hand side (`jac`).
@inline function _newton_storage_terms(r::T, hold, cs::ConvertibleStorage{T}, h, n, jac::Bool) where {T}
    @inbounds begin
        tp, bt, tthk = cs.topoff[n], cs.botoff[n], cs.thk[n]
        ho = hold[n]
        hn = h[n]
        snold = newton_saturation(tp, bt, ho)
        snnew = newton_saturation(tp, bt, hn)
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
        rho2 = cs.rho2[n]
        sold = newton_satthk(tp, bt, tthk, ho)
        if jac || !(zero(T) < snnew < one(T))
            ya = zero(T)
            yr = rho2 * (newton_satthk(tp, bt, tthk, hn) - sold)
        else
            ya = -rho2
            yr = -rho2 * (sold + bt)
        end
        return aterm, rhsterm, ya, yr
    end
end

"""
    storage_terms(rho, hold, conv, h, n, jac, old::StorageChange) -> (ss_aterm, ss_rhs, sy_aterm, sy_rhs)

`storage_terms` in a time step that integrates TVS6 changes (T6.12; `fc_default_sto` with
`integratechanges`: `SsTerms` with `rho1old`, `SyTerms` with `rho2old`): the terms of the
previous heads use the old capacities `old.rhoold`, `old.rho2old`, and a confined cell's
right-hand side gets `old.sdr` (see the header of src/gwf/sto.jl). `conv` is `nothing` or a
`ConvertibleStorage`. Device-safe.
"""
@inline function storage_terms(rho, hold, conv, h, n, jac::Bool, old::StorageChange)
    @inbounds begin
        T = eltype(rho)
        r = rho[n]
        ro = old.rhoold[n]
        (conv === nothing || conv.iconvert[n] == Int32(0)) && return -r, -ro * hold[n] + old.sdr[n], zero(T), zero(T)
        tp, bt, tthk = conv.topoff[n], conv.botoff[n], conv.thk[n]
        ho = hold[n]
        hn = h[n]
        if conv.newton
            snold = newton_saturation(tp, bt, ho)
            snnew = newton_saturation(tp, bt, hn)
        else
            snold = storage_saturation(ho, tp, bt, tthk)
            snnew = storage_saturation(hn, tp, bt, tthk)
        end
        # SsTerms (iorig_ss = 0)
        aterm = -r * snnew
        if !conv.iconf_ss
            zold = bt + T(0.5) * tthk * snold
            znew = bt + T(0.5) * tthk * snnew
            rhsterm = -ro * snold * (ho - zold) - r * snnew * znew
        else
            rhsterm = zero(T)
            snold == one(T) && (rhsterm = rhsterm - ro * (ho - tp))
            if snnew == one(T)
                rhsterm = rhsterm - r * tp
            else
                aterm = zero(T)
            end
        end
        # SyTerms with `thk·sn` as in `storage_terms` (clamped heads; Newton: `newton_satthk`).
        rho2 = conv.rho2[n]
        ro2 = old.rho2old[n]
        if conv.newton
            sold = newton_satthk(tp, bt, tthk, ho)
            if jac || !(zero(T) < snnew < one(T))
                ya = zero(T)
                yr = rho2 * newton_satthk(tp, bt, tthk, hn) - ro2 * sold
            else
                ya = -rho2
                yr = -ro2 * sold - rho2 * bt
            end
        else
            sold = clamp(ho, bt, tp) - bt
            if snnew < one(T)
                if snnew > zero(T)
                    ya = -rho2
                    yr = -ro2 * sold - rho2 * bt
                else
                    ya = zero(T)
                    yr = -ro2 * sold
                end
            else
                ya = zero(T)
                yr = rho2 * tthk - ro2 * sold
            end
        end
        return aterm, rhsterm, ya, yr
    end
end

"""
    storage_jacobian(rho, conv, h, n) -> J

The Newton–Raphson term of active cell `n` on the diagonal (`fn_default_sto`, T6.2; the
right-hand side gets `J·h'`, so the residual does not change): 0 without the Newton
formulation or for a confined cell. Device-safe.
"""
@inline storage_jacobian(rho, ::Nothing, h, n) = zero(eltype(rho))
@inline function storage_jacobian(rho, cs::ConvertibleStorage{T}, h, n) where {T}
    @inbounds begin
        (cs.newton && cs.iconvert[n] != Int32(0)) || return zero(T)
        tp, bt, tthk = cs.topoff[n], cs.botoff[n], cs.thk[n]
        hn = h[n]
        snnew = newton_saturation(tp, bt, hn)
        derv = newton_saturation_derivative(tp, bt, hn)
        r = rho[n]
        j = zero(T)
        cs.iconf_ss || (j = -r * derv * (hn - bt) + r * tthk * snnew * derv)
        (zero(T) < snnew < one(T)) && (j = j - cs.rho2[n] * tthk * derv)
        return j
    end
end

@kernel function storage_add_kernel!(amat, b, @Const(idiag), @Const(rho), @Const(hold), @Const(ibound), conv, @Const(h), old)
    n = @index(Global)
    @inbounds begin
        # fc_default_sto / SsTerms (ICONVERT 0): aterm = −rho, rhsterm = −rho·hold.
        if ibound[n] > 0
            a1, r1, a2, r2 = old === nothing ? storage_terms(rho, hold, conv, h, n) : storage_terms(rho, hold, conv, h, n, true, old)
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
    storage_add_kernel!(backend, _WG)(amat, b, idiag, sto.rho, hold, ibound, sto.conv, h, _storage_change(sto); ndrange = sto.conn.nodes)
    return amat
end

@kernel function storage_rate_kernel!(strgss, strgsy, @Const(rho), @Const(hnew), @Const(hold), @Const(ibound), conv, old)
    n = @index(Global)
    T = eltype(strgss)
    @inbounds begin
        q = zero(T)
        qy = zero(T)
        # cq_default_sto / SsTerms: rate = aterm·hnew − rhsterm = rho·(hold − hnew).
        if ibound[n] > 0
            if old !== nothing
                # T6.12: a step integrating TVS6 changes (`rho1old`, `rho2old`).
                a1, r1, _, _ = storage_terms(rho, hold, conv, hnew, n, true, old)
                q = a1 * hnew[n] - r1
                if conv !== nothing && conv.iconvert[n] != Int32(0)
                    tp, bt = conv.topoff[n], conv.botoff[n]
                    if conv.newton
                        tk = conv.thk[n]
                        qy = old.rho2old[n] * newton_satthk(tp, bt, tk, hold[n]) - conv.rho2[n] * newton_satthk(tp, bt, tk, hnew[n])
                    else
                        qy = old.rho2old[n] * (clamp(hold[n], bt, tp) - bt) - conv.rho2[n] * (clamp(hnew[n], bt, tp) - bt)
                    end
                end
            elseif conv === nothing || conv.iconvert[n] == Int32(0)
                q = rho[n] * (hold[n] - hnew[n])
            else
                a1, r1, _, _ = storage_terms(rho, hold, conv, hnew, n)
                q = a1 * hnew[n] - r1
                # SyTerms rate: rho2·thk·snold − rho2·thk·snnew, as rho2·(clamp(hold) − clamp(h))
                # (the same, without the cancellation near sn = 1 in Float32; T6.1); with
                # the Newton formulation from the quadratic saturations (T6.2).
                tp, bt = conv.topoff[n], conv.botoff[n]
                if conv.newton
                    tk = conv.thk[n]
                    qy = conv.rho2[n] * (newton_satthk(tp, bt, tk, hold[n]) - newton_satthk(tp, bt, tk, hnew[n]))
                else
                    qy = conv.rho2[n] * (clamp(hold[n], bt, tp) - clamp(hnew[n], bt, tp))
                end
            end
        end
        strgss[n] = q
        conv === nothing || (strgsy[n] = qy)
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
    storage_rate_kernel!(backend, _WG)(sto.strgss, sto.strgsy, sto.rho, hnew, hold, ibound, sto.conv, _storage_change(sto); ndrange = sto.conn.nodes)
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
