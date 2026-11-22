# Time-varying hydraulic conductivity and storage, TVK6 and TVS6 (T6.12).
#
# Ports MF6 6.8.2:
# - TvBase.f90 `rp`, `ad`, `sync_node_changes`: every cell named in a PERIOD block is
#   tracked from then on; a setting holds until a later block sets the same property of the
#   cell (the input arrays of Mf6FileKeystring.f90 are permanent; a literal replaces a
#   time-series link, StructArray.f90 `ts_update_dedup`). Without a TS6 file the tracked
#   cells are applied at the start of each period with a block (`rp`); with one, at every
#   time step (`ad`), the series evaluated over the step as for the stress packages.
# - gwf-tvk.f90 `tvk_apply_row_changes`, `tvk_validate_change` (K, K22, K33 > 0; K22OVERK and
#   K33OVERK ratios times the current K), and gwf-npf.f90 `npf_ad`: `calc_condsat` for the
#   connections of the changed cells (`_condsat_connection`, host Float64), then one scatter
#   kernel writes the changed conductances (`condsat`, and `condsat`, `hyl`, `hyh` of the
#   `ConvertibleNpf`) on the device: no full rebuild.
# - gwf-tvs.f90 `tvs_apply_row_changes`, `tvs_validate_change` (SS, SY >= 0; SY only when a
#   cell is convertible): the storage side is in src/gwf/sto.jl (`TvsStorage`).
#
# Not supported: TVK with XT3D (T6.14), with HFB6 (MF6's `npf_ad` overwrites the barrier
# conductance of a changed cell and `condsat_reset` restores the value from before the
# change), and TVK or TVS in a model of a multi-model solution. PRINT_INPUT is accepted; the
# changes are not printed to the listing file.

"""
    TvSource(spec::TvSpec)

The settings of a TVK6 or TVS6 package in force (TvBase.f90): `tracked` (the user nodes
named so far, in first-set order) and `src` (per user node and property the value or the
name of the time series that sets it).
"""
struct TvSource
    spec::TvSpec
    tracked::Vector{Int32}
    seen::Set{Int32}
    src::Dict{Tuple{Int32, Symbol}, Union{Float64, Symbol}}
end

TvSource(spec::TvSpec) = TvSource(spec, Int32[], Set{Int32}(), Dict{Tuple{Int32, Symbol}, Union{Float64, Symbol}}())
Base.copy(s::TvSource) = TvSource(s.spec, copy(s.tracked), copy(s.seen), copy(s.src))

"""
    tv_ts_active(s::TvSource) -> Bool

Whether the package has a TS6 file (`ts_active`): the settings are applied every time step.
"""
tv_ts_active(s::TvSource) = !isempty(s.spec.ts_files)

"""
    tv_rp!(s::TvSource, kper) -> Bool

Stress period `kper` starts (TvBase.f90 `rp`): the records of its PERIOD block, if any, set
the sources of their cells. Returns whether the settings are to be applied now (a block for
`kper`, no TS6 file and a tracked cell).
"""
function tv_rp!(s::TvSource, kper::Integer)
    sp = s.spec.periods
    b = period_block(sp, kper)
    (b == 0 || sp.iper[b] != kper) && return false
    p = sp.data[b]
    for i in eachindex(p.nodeu)
        nu = p.nodeu[i]
        s.src[(nu, p.var[i])] = p.ts[i] === Symbol("") ? p.value[i] : p.ts[i]
        nu in s.seen || (push!(s.seen, nu); push!(s.tracked, nu))
    end
    return !tv_ts_active(s) && !isempty(s.tracked)
end

"""
    tv_value(s::TvSource, nodeu, var, window) -> Union{Nothing, Float64}

The value of property `var` of user node `nodeu` for the time step `window = (t0, t1)`:
`nothing` when no block set it (DNODATA), a time series evaluated over the step
(`ts_value`), or the literal value.
"""
function tv_value(s::TvSource, nodeu::Integer, var::Symbol, window)
    v = get(s.src, (Int32(nodeu), var), nothing)
    v isa Symbol || return v
    isnothing(window) && throw(ArgumentError("tv_value: time series $v needs the time step window"))
    return ts_value(s.spec.timeseries[v], window[1], window[2])
end

# The cell string of `noder_to_string` for a user node.
_tv_cell(dis::AbstractDisSpec, nodeu) = _shape_string(cellid(grid_dims(dis), nodeu))

# TvBase.f90 `apply_row_changes` for a tracked cell outside the active domain.
_tv_inactive(dis, nodeu) = "CELLID $(_tv_cell(dis, nodeu)) is not in the active model domain."

"""
    TvkState

TVK6 of a model on a device (T6.12): the `TvSource`, the grid (`conn`, `dis`), the current
NPF properties (`props`: `NpfProperties` with copies of K, K22, K33 that TVK changes,
Float64), the K22OVERK/K33OVERK options, and the current saturated conductances and
effective conductivities per symmetric connection (`condsat`, `hyl`, `hyh`, Float64, as on
the device).
"""
struct TvkState
    source::TvSource
    conn::Connectivity
    dis::AbstractDisSpec
    props::NpfProperties
    k22overk::Bool
    k33overk::Bool
    condsat::Vector{Float64}
    hyl::Vector{Float64}
    hyh::Vector{Float64}
end

function Base.copy(s::TvkState)
    p = s.props
    props = NpfProperties(
        p.icelltype, copy(p.k11), copy(p.k22), copy(p.k33), p.angle1, p.angle2, p.angle3, p.ik22, p.ik33, p.icellavg,
        p.thickstart, p.sat0, p.wetdry,
    )
    return TvkState(copy(s.source), s.conn, s.dis, props, s.k22overk, s.k33overk, copy(s.condsat), copy(s.hyl), copy(s.hyh))
end

"""
    tvk_state(spec::GwfSpec, conn, setup::NpfSetup) -> Union{TvkState, Nothing}

The `TvkState` of a model with TVK6 (`spec.npf.tvk`), `nothing` without. TVK6 with XT3D or
HFB6 is an `MF6UnsupportedError`.
"""
function tvk_state(spec::GwfSpec, conn::Connectivity, setup::NpfSetup)
    t = spec.npf.tvk
    # As `check_storage_spec` for TVS6: file names without the input (a spec not read by
    # `load_simulation`) would otherwise run without TVK6.
    (isnothing(t) && !isempty(spec.npf.tvk_files)) &&
        unsupported(spec.npf.file, "TVK6 file names without the TVK6 input (`NpfSpec.tvk`)")
    isnothing(t) && return nothing
    file = isempty(t.file) ? "TVK" : t.file
    spec.npf.xt3d && unsupported(file, "TVK6 with the NPF6 option XT3D (model $(spec.name))")
    isnothing(spec.hfb) || unsupported(file, "TVK6 with HFB6 (model $(spec.name))")
    p = setup.properties
    props = NpfProperties(
        p.icelltype, copy(p.k11), copy(p.k22), copy(p.k33), p.angle1, p.angle2, p.angle3, p.ik22, p.ik33, p.icellavg,
        p.thickstart, p.sat0, p.wetdry,
    )
    return TvkState(
        TvSource(t), conn, spec.dis, props, spec.npf.k22overk, spec.npf.k33overk, copy(setup.condsat), copy(setup.hyl), copy(setup.hyh),
    )
end

# Writes the conductance data of the connections `idx` (TVK6): `condsat` and, with a
# ConvertibleNpf (`cv`), its `condsat`, `hyl` and `hyh`.
@kernel function tvk_scatter_kernel!(condsat, cvcondsat, hyl, hyh, @Const(idx), @Const(csat), @Const(vhyl), @Const(vhyh), cv::Bool)
    i = @index(Global)
    @inbounds begin
        j = idx[i]
        condsat[j] = csat[i]
        if cv
            cvcondsat[j] = csat[i]
            hyl[j] = vhyl[i]
            hyh[j] = vhyh[i]
        end
    end
end

"""
    tvk_sync!(st::TvkState, npf::NpfModel, window) -> Bool

Apply the TVK6 settings in force to every tracked cell (TvBase.f90 `sync_node_changes`,
gwf-tvk.f90 `tvk_apply_row_changes` and `tvk_validate_change`; `window` is the time step,
for the time series), then recompute the saturated conductances of the connections of
these cells (gwf-npf.f90 `npf_ad` → `calc_condsat`) and write those that changed to the
device. MF6's errors (a cell outside the active domain, a value <= 0) are one
`MF6InputError`. Returns whether a conductance changed.
"""
function tvk_sync!(st::TvkState, npf::NpfModel{T}, window) where {T}
    src = st.source
    conn = st.conn
    p = st.props
    file = isempty(src.spec.file) ? "TVK" : src.spec.file
    errors = String[]
    nodes = Int32[]
    kerr(name, n, v) = push!(errors, rstrip(" TVK changed hydraulic property $name is <= 0 for cell $(_tv_cell(st.dis, conn.nodeuser[n])) " * _fortran_1pg(v; w = 15, d = 6)))
    for nu in src.tracked
        n = conn.nodereduced[nu]
        k = tv_value(src, nu, :K, window)
        k22 = tv_value(src, nu, :K22, window)
        k33 = tv_value(src, nu, :K33, window)
        any(!isnothing, (k, k22, k33)) || continue
        if n <= 0
            push!(errors, _tv_inactive(st.dis, nu))
            continue
        end
        push!(nodes, n)
        if !isnothing(k)
            p.k11[n] = k
            p.k11[n] <= 0.0 && kerr("K", n, p.k11[n])
        end
        if !isnothing(k22)
            p.k22[n] = st.k22overk ? k22 * p.k11[n] : k22
            p.k22[n] <= 0.0 && kerr("K22", n, p.k22[n])
        end
        if !isnothing(k33)
            p.k33[n] = st.k33overk ? k33 * p.k11[n] : k33
            p.k33[n] <= 0.0 && kerr("K33", n, p.k33[n])
        end
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    # npf_ad: calc_condsat(n, .false.) for every changed node: all its connections.
    idx = Int32[]
    done = Set{Int32}()
    for n in nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
        j = conn.jas[ii]
        j in done && continue
        push!(done, j)
        m = conn.ja[ii]
        lo, hi = min(n, m), max(n, m)
        c, hl, hh = _condsat_connection(p, conn, lo, hi, j)
        if c !== st.condsat[j] || hl !== st.hyl[j] || hh !== st.hyh[j]
            st.condsat[j], st.hyl[j], st.hyh[j] = c, hl, hh
            push!(idx, j)
        end
    end
    isempty(idx) && return false
    cv = npf.cv
    backend = get_backend(npf.condsat)
    dev(x) = _device_copy(backend, x)
    hascv = !isnothing(cv)
    tvk_scatter_kernel!(backend, _WG)(
        npf.condsat, hascv ? cv.condsat : npf.condsat, hascv ? cv.hyl : npf.condsat, hascv ? cv.hyh : npf.condsat,
        dev(idx), dev(T[T(st.condsat[j]) for j in idx]), dev(T[T(st.hyl[j]) for j in idx]), dev(T[T(st.hyh[j]) for j in idx]), hascv;
        ndrange = length(idx),
    )
    return true
end

"""
    tvk_rp!(st::TvkState, npf, kper; window = nothing) -> Bool

Stress period `kper` starts (gwf-npf.f90 `npf_rp` → TvBase.f90 `rp`, then `npf_ad`): the
period's block, applied now without a TS6 file. Returns whether a conductance changed.
"""
tvk_rp!(st::TvkState, npf::NpfModel, kper::Integer; window = nothing) = tv_rp!(st.source, kper) ? tvk_sync!(st, npf, window) : false

"""
    tvk_ad!(st::TvkState, npf, window) -> Bool

A time step starts (TvBase.f90 `ad`): with a TS6 file every tracked cell is applied with
the series evaluated over `window`. Returns whether a conductance changed.
"""
tvk_ad!(st::TvkState, npf::NpfModel, window) =
    (tv_ts_active(st.source) && !isempty(st.source.tracked)) ? tvk_sync!(st, npf, window) : false
