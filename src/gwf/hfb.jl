# Horizontal flow barriers, HFB6 (T6.11).
#
# Ports MF6 6.8.2 gwf-hfb.f90: `source_data` (cells to reduced nodes; a barrier with an
# excluded cell is dropped with a warning), `check_data` (unconnected cells, two barriers on
# one connection), `hfb_rp` (a PERIOD block replaces the barriers: `condsat_reset`, then
# `condsat_modify`), `condsat_modify` and `hfb_fc`; `hfb_cq` needs no port (see below).
#
# MF6 applies a barrier in one of two ways, decided per barrier from NPF's ICELLTYPE:
# - static (`condsat_modify`, at the start of the period): with the Newton formulation, or
#   when both cells have ICELLTYPE 0, the saturated conductance of the connection becomes
#   `c·ch/(c + ch)` with `ch = hydchr·faarea` (`faarea`: `hwva` for a vertical connection,
#   else `hwva` times the saturated face height), or `−hydchr·c` for `hydchr <= 0`;
# - dynamic (`hfb_fc`, every formulation of the standard formulation): when a cell has
#   ICELLTYPE 1, the same series form on the head-dependent NPF conductance, with the face
#   height from the heads below the top (`_hfb_conductance` in gwf/convertible.jl).
# A barrier between a cell of ICELLTYPE 0 and one of ICELLTYPE > 1 (and not Newton) changes
# nothing in MF6 (it tests `== 1`); the port does the same.
#
# On the device: the static values are computed on the host in Float64 from the saturated
# conductances (`NpfSetup.condsat`) and rounded once to `T`; the dynamic barriers are a flag
# (`ConvertibleNpf.ihcx` bit 16) and the hydraulic characteristic (`ConvertibleNpf.hfb`).
# At the start of a period with a PERIOD block one scatter kernel (`hfb_scatter_kernel!`)
# writes the connections of the previous and the new barriers: no full copy of `condsat`.
# Flows (`hfb_cq`): MF6 rewrites FLOW-JA-FACE of a dynamic barrier with the conductance of
# the last formulation; `npf_flowja!` evaluates `connection_terms` with the saturations of
# the last `npf_cf!` and the barrier at the final heads, the same at convergence. HFB6 has
# no budget term. T6.14: with XT3D every barrier scales the XT3D coefficients of its
# connection (`xt3d_fhfb`; `hfb_rp!` keeps the barriers in `HfbState.xjs`/`xchr`, and the
# model writes them to its `Xt3dVarNpf`, src/gwf/xt3d_var.jl). Not supported: VSC6 (out of
# scope), models of a multi-model solution.

"""
    HfbBarriers

The barriers of one HFB6 PERIOD block after MF6's checks (`source_data`, `check_data`), in
input order without the dropped ones: reduced nodes `noden`, `nodem`, `hydchr`, the
symmetric connection `jas` and `static` (applied to `condsat`, `condsat_modify`) or not
(dynamic, `hfb_fc`; or neither, see the header of src/gwf/hfb.jl: `dynamic`).
"""
struct HfbBarriers
    noden::Vector{Int32}
    nodem::Vector{Int32}
    hydchr::Vector{Float64}
    jas::Vector{Int32}
    static::Vector{Bool}
    dynamic::Vector{Bool}
end

"""
    hfb_barriers(spec::HfbSpec, block, dis, conn, icelltype; newton = false) -> HfbBarriers

Resolve the barriers of PERIOD block `block` of `spec` on the model grid (gwf-hfb.f90
`source_data`, `check_data`): user nodes to reduced nodes (a barrier with a cell outside the
active grid is dropped with a warning, "HFB connection between inactive cell(s) will be
excluded"), the connection of the two cells (an `MF6InputError` when they are not
connected, or when two barriers are on one connection), and how MF6 applies the barrier
from `icelltype` (NPF, by reduced node) and `newton`.
"""
function hfb_barriers(
        spec::HfbSpec, block::Integer, dis::AbstractDisSpec, conn::Connectivity, icelltype::AbstractVector{<:Integer};
        newton::Bool = false,
    )
    p = spec.periods.data[block]
    g = grid_dims(dis)
    file = isempty(spec.file) ? "HFB" : spec.file
    cstr(nu) = _shape_string(cellid(g, nu))
    nn = Int32[]
    mm = Int32[]
    chr = Float64[]
    for i in eachindex(p.cell1)
        nu1, nu2 = p.cell1[i], p.cell2[i]
        n = conn.nodereduced[nu1]
        m = conn.nodereduced[nu2]
        if n <= 0 || m <= 0
            @warn "$file: HFB connection between inactive cell(s) will be excluded: $(cstr(nu1)) to $(cstr(nu2))."
            continue
        end
        push!(nn, n)
        push!(mm, m)
        push!(chr, p.hydchr[i])
    end
    # check_data
    errors = String[]
    owner = zeros(Int, conn.njas)
    js = zeros(Int32, length(nn))
    for k in eachindex(nn)
        n, m = nn[k], mm[k]
        ii = getjaindex(conn, n, m)
        if ii <= 0 || n == m
            push!(errors, "HFB no. $k is between two unconnected cells: $(cstr(conn.nodeuser[n])) and $(cstr(conn.nodeuser[m]))")
            continue
        end
        j = conn.jas[ii]
        js[k] = j
        if owner[j] == 0
            owner[j] = k
        else
            push!(
                errors,
                "HFB no. $(owner[j]) and HFB no. $k are both between cells $(cstr(conn.nodeuser[n])) and $(cstr(conn.nodeuser[m])). Only one HFB can be assigned to a cell connection.",
            )
        end
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    static = Bool[newton || (icelltype[nn[k]] == 0 && icelltype[mm[k]] == 0) for k in eachindex(nn)]
    dynamic = Bool[!static[k] && (icelltype[nn[k]] == 1 || icelltype[mm[k]] == 1) for k in eachindex(nn)]
    return HfbBarriers(nn, mm, chr, js, static, dynamic)
end

"""
    hfb_condsat(conn, condsat, n, m, j, hydchr) -> Float64

The saturated conductance of connection `j` (cells `n`, `m`) with a barrier, from the
conductance without it, `condsat` (gwf-hfb.f90 `condsat_modify`; host, Float64): the face
area is `hwva` for a vertical connection, else `hwva` times the face height (the overlap of
the cells for a staggered DISU connection, `ihc` 2, else their mean thickness); then
`c·ch/(c + ch)` with `ch = hydchr·faarea`, or `−hydchr·c` for `hydchr <= 0`.
"""
function hfb_condsat(conn::Connectivity, condsat::Float64, n::Integer, m::Integer, j::Integer, hydchr::Float64)
    ihc = conn.ihc[j]
    topn, topm, botn, botm = conn.top[n], conn.top[m], conn.bot[n], conn.bot[m]
    faheight = ihc == 2 ? min(topn, topm) - max(botn, botm) : 0.5 * ((topn - botn) + (topm - botm))
    faarea = ihc == 0 ? conn.hwva[j] : conn.hwva[j] * faheight
    if hydchr > 0.0
        ch = hydchr * faarea
        return condsat * ch / (condsat + ch)
    else
        return -condsat * hydchr
    end
end

"""
    HfbState

The HFB6 package of a model on a device (T6.11): the spec, the grid and NPF data it needs
(host), `newton`, `block` (the PERIOD block applied last, 0 before), and the symmetric
connections whose conductance the current barriers change: `static` (`condsat`) and
`dynamic` (`ConvertibleNpf.ihcx` bit 16 and `hfb`). `hfb_rp!` applies a new block.
T6.14: `xt3d` (a model with XT3D: every barrier is applied to the XT3D coefficients,
`xt3d_fhfb`), with the connections `xjs` and hydraulic characteristics `xchr` of the
current barriers.
"""
mutable struct HfbState
    const spec::HfbSpec
    const dis::AbstractDisSpec
    const newton::Bool
    block::Int
    static::Vector{Int32}
    dynamic::Vector{Int32}
    const xt3d::Bool
    xjs::Vector{Int32}
    xchr::Vector{Float64}
end

Base.copy(s::HfbState) = HfbState(s.spec, s.dis, s.newton, s.block, copy(s.static), copy(s.dynamic), s.xt3d, copy(s.xjs), copy(s.xchr))

"""
    hfb_state(spec::GwfSpec) -> Union{HfbState, Nothing}

The `HfbState` of a model with HFB6, `nothing` without (T6.14: also with XT3D,
`xt3d_fhfb`).
"""
function hfb_state(spec::GwfSpec)
    h = spec.hfb
    isnothing(h) && return nothing
    return HfbState(h, spec.dis, spec.newton, 0, Int32[], Int32[], spec.npf.xt3d, Int32[], Float64[])
end

# Writes the conductance data of the connections `idx`: `condsat` (and the ConvertibleNpf's
# copy, `cvcondsat`) from `csat`; with `dyn`, also the dynamic-barrier flag (`ihcx` bit 16)
# from `flag` and the hydraulic characteristic `hfb` from `chr`.
@kernel function hfb_scatter_kernel!(condsat, cvcondsat, ihcx, hfb, @Const(idx), @Const(csat), @Const(chr), @Const(flag), dyn::Bool)
    i = @index(Global)
    @inbounds begin
        j = idx[i]
        condsat[j] = csat[i]
        cvcondsat[j] = csat[i]
        if dyn
            ihcx[j] = (ihcx[j] & ~Int32(16)) | (flag[i] != Int32(0) ? Int32(16) : Int32(0))
            hfb[j] = chr[i]
        end
    end
end

"""
    hfb_rp!(st::HfbState, npf::NpfModel, conn, setup::NpfSetup, kper) -> Bool

Stress period `kper` starts (gwf-hfb.f90 `hfb_rp`): when HFB6 has a PERIOD block for
`kper`, the barriers of the previous block are removed (`condsat_reset`), the new ones
resolved (`hfb_barriers`) and applied (`condsat_modify`; dynamic barriers flagged for
`hfb_fc`), with one launch of `hfb_scatter_kernel!` over the connections of both. Returns
whether a saturated conductance changed (the caller marks the off-diagonals stale).
"""
function hfb_rp!(st::HfbState, npf::NpfModel{T}, conn::Connectivity, setup::NpfSetup, kper::Integer) where {T}
    sp = st.spec.periods
    b = period_block(sp, kper)
    (b == 0 || sp.iper[b] != kper) && return false
    bar = hfb_barriers(st.spec, b, st.dis, conn, setup.properties.icelltype; newton = st.newton)
    st.block = b
    if st.xt3d
        # T6.14 (`hfb_fc`, XT3D branch): every barrier on the XT3D coefficients; the caller
        # writes them (`xvar_hfb!`). XT3D does not use `condsat`.
        st.xjs = bar.jas
        st.xchr = bar.hydchr
        return true
    end
    csat = Dict{Int32, Float64}()
    dyn = Dict{Int32, Float64}()
    for k in eachindex(bar.jas)
        j = bar.jas[k]
        if bar.static[k]
            csat[j] = hfb_condsat(conn, setup.condsat[j], bar.noden[k], bar.nodem[k], j, bar.hydchr[k])
        elseif bar.dynamic[k]
            dyn[j] = bar.hydchr[k]
        end
    end
    idx = sort!(unique!(Int32[st.static; st.dynamic; collect(keys(csat)); collect(keys(dyn))]))
    changed = !isempty(st.static) || !isempty(csat)
    st.static = sort!(collect(keys(csat)))
    st.dynamic = sort!(collect(keys(dyn)))
    isempty(idx) && return false
    cv = npf.cv
    hasdyn = !isnothing(cv) && length(cv.hfb) > 0
    !hasdyn && !isempty(dyn) && error("hfb_rp!: dynamic barriers need a ConvertibleNpf with `hfb`")
    backend = get_backend(npf.condsat)
    dev(x) = _device_copy(backend, x)
    vcsat = T[T(get(csat, j, setup.condsat[j])) for j in idx]
    vchr = T[T(get(dyn, j, 0.0)) for j in idx]
    vflag = Int32[haskey(dyn, j) ? 1 : 0 for j in idx]
    cvc = isnothing(cv) ? npf.condsat : cv.condsat
    ihcx = hasdyn ? cv.ihcx : dev(Int32[])
    hfb = hasdyn ? cv.hfb : npf.condsat
    hfb_scatter_kernel!(backend, _WG)(npf.condsat, cvc, ihcx, hfb, dev(idx), dev(vcsat), dev(vchr), dev(vflag), hasdyn; ndrange = length(idx))
    return changed
end
