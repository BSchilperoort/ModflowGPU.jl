# Node property flow (NPF), confined: saturated conductance per connection (host only).
#
# Ports MF6 6.8.2:
# - gwf-npf.f90 `prepcheck` (K, K33, K22 > 0 with K33OVERK/K22OVERK applied, WETDRY with
#   REWET, ANGLE1/2/3 consistency and the degree → radian conversion), `check_options`
#   (option combinations with NEWTON and XT3D, HIGHEST_CELL_SATURATION without NEWTON),
#   `source_griddata` (K22 = K33 = K when not given), `preprocess_input` (negative
#   ICELLTYPE → 1 without THICKSTRT), `calc_condsat` (upper triangle, saturation 1),
#   `hy_eff` (effective K in the connection direction);
# - GwfConductanceUtils.f90 `hcond`/`convertible_standard`, `vcond`, `condmean`,
#   `logmean`, `staggered_thkfrac`; HGeoUtil.f90 `hyeff`; Dis.f90 `connection_normal`.
#
# `calc_condsat` calls `hcond` and `vcond` with all cells active and convertible, saturation
# 1 and heads at the cell bottom (vertical) or top (horizontal), so condsat is the
# full-thickness conductance whatever ICELLTYPE is (THICKSTRT, which changes that, is
# rejected until T6.1). The branches taken reduce to:
# - horizontal (ihc ≠ 0): `condmean(K_n, K_m, top_n − bot_n, top_m − bot_m, cl1, cl2, hwva,
#   icellavg)` (staggered DISU connections use `staggered_thkfrac`);
# - vertical (ihc = 0): `hwva / ((top_n − bot_n)/2/K33_n + (top_m − bot_m)/2/K33_m)`.
# The arithmetic below keeps MF6's operation order, so results agree to the last bits up to
# the libm `sin`/`cos`/`log` of the angle and log-mean cases.
#
# Not ported (cannot occur, or handled elsewhere):
# - `preprocess_input`'s conversion to no-flow of cells with K = 0 (prepcheck already
#   rejects K <= 0), its negative-thickness check (`check_dis_grid`), wetting and drying
#   (T6.1), and the Newton bottom-node search (T6.2);
# - XT3D: since T6.5 in src/gwf/xt3d.jl (MF6 does not compute condsat with XT3D; it is
#   computed here but not used);
# - VSC/TVK modifications of K (out of scope / T6.x).

# GwfConductanceUtils.f90 CCOND_* (the ICELLAVG values).
const CCOND_HMEAN = 0
const CCOND_LMEAN = 1
const CCOND_AMTLMK = 2
const CCOND_AMTHMK = 3

const _ICELLAVG = Dict(:harmonic => CCOND_HMEAN, :logarithmic => CCOND_LMEAN, :amt_lmk => CCOND_AMTLMK, :amt_hmk => CCOND_AMTHMK)

# ConstantsModule DPIO180 = datan(DONE) / 4.5d1.
const _DPIO180 = atan(1.0) / 45.0

"""
    logmean(d1, d2)

Logarithmic mean of two positive numbers (GwfConductanceUtils.f90 `logmean`): `(d2 −
d1) / log(d2/d1)`, or the arithmetic mean when `0.995 < d2/d1 < 1.005`.
"""
@inline function logmean(d1::T, d2::T) where {T <: AbstractFloat}
    drat = d2 / d1
    if drat <= T(0.995) || drat >= T(1.005)  # DLNLOW, DLNHIGH
        return (d2 - d1) / log(drat)
    else
        return T(0.5) * (d1 + d2)
    end
end

"""
    condmean(k1, k2, thick1, thick2, cl1, cl2, width, iavgmeth)

Conductance between two cells (GwfConductanceUtils.f90 `condmean`) for the averaging
method `iavgmeth`: `CCOND_HMEAN` (harmonic mean of the transmissivities), `CCOND_LMEAN`
(logarithmic mean of the transmissivities), `CCOND_AMTLMK` (arithmetic-mean thickness,
logarithmic-mean K) or `CCOND_AMTHMK` (arithmetic-mean thickness, harmonic-mean K). `k1`,
`thick1`, `cl1` belong to the first cell and `k2`, `thick2`, `cl2` to the second; `width`
is the face width.
"""
@inline function condmean(k1::T, k2::T, thick1::T, thick2::T, cl1::T, cl2::T, width::T, iavgmeth::Integer) where {T <: AbstractFloat}
    t1 = k1 * thick1
    t2 = k2 * thick2
    if iavgmeth == CCOND_HMEAN
        return t1 * t2 > zero(T) ? width * t1 * t2 / (t1 * cl2 + t2 * cl1) : zero(T)
    elseif iavgmeth == CCOND_LMEAN
        tmean = t1 * t2 > zero(T) ? logmean(t1, t2) : zero(T)
        return tmean * width / (cl1 + cl2)
    elseif iavgmeth == CCOND_AMTLMK
        kmean = k1 * k2 > zero(T) ? logmean(k1, k2) : zero(T)
        return kmean * T(0.5) * (thick1 + thick2) * width / (cl1 + cl2)
    else  # CCOND_AMTHMK
        denom = k1 * cl2 + k2 * cl1
        kmean = denom > zero(T) ? k1 * k2 / denom : zero(T)
        return kmean * T(0.5) * (thick1 + thick2) * width
    end
end

"""
    staggered_thkfrac(top, bot, sat, topc, botc)

Saturated thickness of a cell within the overlap with a vertically staggered neighbour
(`topc`, `botc`) (GwfConductanceUtils.f90 `staggered_thkfrac`; DISU `ihc = 2`).
"""
@inline function staggered_thkfrac(top::T, bot::T, sat::T, topc::T, botc::T) where {T <: AbstractFloat}
    sill_top = min(top, topc)
    sill_bot = max(bot, botc)
    tp = bot + sat * (top - bot)
    return max(min(tp, sill_top) - sill_bot, zero(T))
end

"""
    hyeff(k11, k22, k33, ang1, ang2, ang3, vg1, vg2, vg3) -> K

Effective hydraulic conductivity in the direction of the unit vector `(vg1, vg2, vg3)`
(model coordinates) of the conductivity ellipsoid with principal values `k11`, `k22`,
`k33`, rotated by `ang1` (in the x-y plane), `ang2` (up/down from it) and `ang3` (about the
major axis), in radians (HGeoUtil.f90 `hyeff` with `iavgmeth = 0`, the value NPF uses:
`1/K = Σ ve_i²/k_i`, with a principal direction that matches the vector exactly giving
that principal value).
"""
function hyeff(k11::T, k22::T, k33::T, ang1::T, ang2::T, ang3::T, vg1::T, vg2::T, vg3::T) where {T <: AbstractFloat}
    s1, c1 = sin(ang1), cos(ang1)
    s2, c2 = sin(ang2), cos(ang2)
    s3, c3 = sin(ang3), cos(ang3)
    # Rotation matrix r (only the products with vg are needed).
    r11 = c1 * c2
    r12 = c1 * s2 * s3 - s1 * c3
    r13 = -c1 * s2 * c3 - s1 * s3
    r21 = s1 * c2
    r22 = s1 * s2 * s3 + c1 * c3
    r23 = -s1 * s2 * c3 + c1 * s3
    r31 = s2
    r32 = -c2 * s3
    r33 = c2 * c3
    # Unit vector in the ellipse axes.
    ve1 = r11 * vg1 + r21 * vg2 + r31 * vg3
    ve2 = r12 * vg1 + r22 * vg2 + r32 * vg3
    ve3 = r13 * vg1 + r23 * vg2 + r33 * vg3
    dnum = one(T)
    d1 = ve1^2
    d2 = ve2^2
    d3 = ve3^2
    if ve1 != zero(T)
        dnum = dnum * k11
        d2 = d2 * k11
        d3 = d3 * k11
    end
    if ve2 != zero(T)
        dnum = dnum * k22
        d1 = d1 * k22
        d3 = d3 * k22
    end
    if ve3 != zero(T)
        dnum = dnum * k33
        d1 = d1 * k33
        d2 = d2 * k33
    end
    denom = d1 + d2 + d3
    return denom > zero(T) ? dnum / denom : zero(T)
end

"""
    NpfProperties

Hydraulic properties of the NPF package by reduced node, in Float64, as MF6 holds them
after `source_griddata` and `prepcheck`:

- `icelltype` (negative values replaced by 1, `preprocess_input` without THICKSTRT);
- `k11`, `k22`, `k33`: K22 and K33 are K where not given (`ik22`, `ik33` false), and
  multiplied by K with K22OVERK/K33OVERK;
- `angle1`, `angle2`, `angle3`: in radians, `nothing` when not given;
- `icellavg`: `CCOND_HMEAN`, `CCOND_LMEAN`, `CCOND_AMTLMK` or `CCOND_AMTHMK`;
- `thickstart` (T6.1): cells with negative ICELLTYPE under THICKSTRT (`ithickstartflag`),
  confined (ICELLTYPE 0) with the saturated thickness of the starting head;
- `sat0` (T6.1): the initial saturation of each cell (`calc_initial_sat`: 1, or
  `(STRT − bot)/(top − bot)` for `thickstart` cells), used by `calc_condsat`;
- `wetdry` (T6.1): WETDRY by reduced node with REWET, `nothing` without.
"""
struct NpfProperties
    icelltype::Vector{Int32}
    k11::Vector{Float64}
    k22::Vector{Float64}
    k33::Vector{Float64}
    angle1::Union{Nothing, Vector{Float64}}
    angle2::Union{Nothing, Vector{Float64}}
    angle3::Union{Nothing, Vector{Float64}}
    ik22::Bool
    ik33::Bool
    icellavg::Int
    thickstart::Vector{Bool}
    sat0::Vector{Float64}
    wetdry::Union{Nothing, Vector{Float64}}
end

# gwf-npf.f90 `prepcheck` fmtkerr: "Hydraulic property <name> is <= 0 for cell <cellid> <1PG15.6>".
function _check_positive!(errors, name, k, nodeuser, g::GridDims)
    nerr = 0
    for n in eachindex(k)
        if k[n] <= 0.0
            nerr += 1
            if nerr <= 20
                cell = _shape_string(cellid(g, nodeuser[n]))
                push!(errors, rstrip("Hydraulic property $name is <= 0 for cell $cell " * _fortran_1pg(k[n]; w = 15, d = 6)))
            end
        end
    end
    nerr > 20 && push!(errors, "... $nerr additional errors not shown for $name")
    return nothing
end

# ICELLTYPE by user node as MF6 holds it after `preprocess_input` (as `npf_properties`):
# negative is 1, or 0 under THICKSTRT. What `dis_ar` writes to the binary grid file and
# `calc_spdis` uses (T6.9 review).
_mf6_icelltype(npf::NpfSpec) = Int32[t < 0 ? Int32(npf.thickstrt ? 0 : 1) : t for t in npf.icelltype]

"""
    npf_properties(npf::NpfSpec, dis::AbstractDisSpec, conn::Connectivity; newton = false) -> NpfProperties

The NPF properties by reduced node, after MF6's checks of the NPF input
(gwf-npf.f90 `prepcheck`, then `check_options`; each throws an `MF6InputError` for
`npf.file` with MF6's messages, all errors of one routine in one message). With
`newton = true` the NEWTON option combinations are checked (the model's NEWTON option).
XT3D with convertible cells or NEWTON is an `MF6UnsupportedError` (`check_xt3d`, T6.5).
"""
function npf_properties(
        npf::NpfSpec, dis::AbstractDisSpec, conn::Connectivity; newton::Bool = false,
        strt::Union{Nothing, AbstractVector{<:Real}} = nothing,
    )
    file = isempty(npf.file) ? "NPF" : npf.file
    g = grid_dims(dis)
    nu = conn.nodeuser
    nusr = Int(conn.nodesuser)
    for (name, a) in (
            ("ICELLTYPE", npf.icelltype), ("K", npf.k), ("K22", npf.k22), ("K33", npf.k33),
            ("ANGLE1", npf.angle1), ("ANGLE2", npf.angle2), ("ANGLE3", npf.angle3), ("WETDRY", npf.wetdry),
        )
        isnothing(a) || length(a) == nusr ||
            throw(ArgumentError("NpfSpec: length($name) = $(length(a)), expected $nusr user nodes"))
    end
    # gwf-npf.f90 `source_griddata`: by reduced node; K22, K33 = K when not given. The
    # K33OVERK/K22OVERK "specified but K33/K22 not specified" errors are the loader's.
    ik22, ik33 = !isnothing(npf.k22), !isnothing(npf.k33)
    (npf.k22overk && !ik22) && throw(MF6InputError(file, 0, "K22OVERK option specified but K22 not specified."))
    (npf.k33overk && !ik33) && throw(MF6InputError(file, 0, "K33OVERK option specified but K33 not specified."))
    k11 = npf.k[nu]
    k22 = ik22 ? npf.k22[nu] : copy(k11)
    k33 = ik33 ? npf.k33[nu] : copy(k11)
    angle1, angle2, angle3 = (isnothing(a) ? nothing : a[nu] for a in (npf.angle1, npf.angle2, npf.angle3))

    # gwf-npf.f90 `prepcheck`.
    errors = String[]
    _check_positive!(errors, "K", k11, nu, g)
    if ik33
        npf.k33overk && (k33 .*= k11)
        _check_positive!(errors, "K33", k33, nu, g)
    end
    if ik22
        npf.k22overk && (k22 .*= k11)
        _check_positive!(errors, "K22", k22, nu, g)
    end
    npf.rewet && isnothing(npf.wetdry) && push!(errors, "Error in GRIDDATA block: WETDRY not found.")
    isnothing(angle1) || (angle1 .*= _DPIO180)
    if !isnothing(angle2)
        isnothing(angle1) && push!(errors, "ANGLE2 SPECIFIED BUT NOT ANGLE1. ANGLE2 REQUIRES ANGLE1.")
        isnothing(angle3) && push!(errors, "ANGLE2 SPECIFIED BUT NOT ANGLE3. SPECIFY BOTH OR NEITHER ONE.")
        angle2 .*= _DPIO180
    end
    if !isnothing(angle3)
        isnothing(angle1) && push!(errors, "ANGLE3 SPECIFIED BUT NOT ANGLE1. ANGLE3 REQUIRES ANGLE1.")
        isnothing(angle2) && push!(errors, "ANGLE3 SPECIFIED BUT NOT ANGLE2. SPECIFY BOTH OR NEITHER ONE.")
        angle3 .*= _DPIO180
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))

    check_npf_options(npf; newton)

    _check_disu_angldegx(npf, dis, ik22)


    # gwf-npf.f90 `preprocess_input`: without THICKSTRT, negative ICELLTYPE means 1; with
    # it, such a cell is confined (ICELLTYPE 0) with the saturation of the starting head
    # (`ithickstartflag`, `calc_initial_sat`: `thksat` of STRT, not clamped), and STRT below
    # the cell bottom is an error.
    nodes = Int(conn.nodes)
    icelltype = Vector{Int32}(undef, nodes)
    thickstart = falses(nodes)
    sat0 = ones(nodes)
    for n in 1:nodes
        ict = npf.icelltype[nu[n]]
        if ict < 0 && npf.thickstrt
            isnothing(strt) && throw(ArgumentError("npf_properties: THICKSTRT needs the starting heads (`strt`)"))
            s = Float64(strt[nu[n]])
            topn, botn = conn.top[n], conn.bot[n]
            if botn > s
                push!(errors, "Negative cell thickness at cell $(_shape_string(cellid(g, nu[n])))")
                push!(errors, rstrip(" Initial head, bottom elevation:" * _fortran_1pg(s; w = 13, d = 5) * _fortran_1pg(botn; w = 13, d = 5)))
            end
            sat0[n] = s >= topn ? 1.0 : (s - botn) / (topn - botn)
            thickstart[n] = true
            icelltype[n] = 0
        else
            icelltype[n] = ict < 0 ? Int32(1) : ict
        end
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    # T6.5: XT3D (src/gwf/xt3d.jl) for confined models.
    check_xt3d(npf, dis, icelltype; newton)
    wetdry = npf.rewet && !isnothing(npf.wetdry) ? npf.wetdry[nu] : nothing
    return NpfProperties(
        icelltype, k11, k22, k33, angle1, angle2, angle3, ik22, ik33, _ICELLAVG[npf.cell_averaging], thickstart, sat0, wetdry,
    )
end

# gwf-npf.f90 `prepcheck` ("ANGLDEGX not provided ... but K22 was specified"), `npf_ar`
# (inconsistent ANGLDEGX with XT3D, K22 or SAVE_SPECIFIC_DISCHARGE) and `calc_spdis`
# (ANGLDEGX required for specific discharge): only DISU grids can lack ANGLDEGX.
_check_disu_angldegx(::NpfSpec, ::AbstractDisSpec, ::Bool) = nothing
function _check_disu_angldegx(npf::NpfSpec, dis::DisuSpec, ik22::Bool)
    file = isempty(npf.file) ? "NPF" : npf.file
    if isnothing(dis.angldegx)
        ik22 && throw(MF6InputError(file, 0, "Error.  ANGLDEGX not provided in discretization file, but K22 was specified. "))
        npf.save_specific_discharge && throw(
            MF6InputError(file, 0, "Error.  ANGLDEGX not provided in discretization file.  ANGLDEGX required for calculation of specific discharge."),
        )
    elseif npf.xt3d || ik22 || npf.save_specific_discharge
        nerr = disu_angldegx_errors(dis; warn = false)
        nerr > 0 && throw(
            MF6InputError(
                isempty(dis.file) ? "DISU" : dis.file, 0,
                "ANGLDEGX values in the DISU Package are inconsistent for $nerr cell faces (see the warnings written after the DISU Package input in the model listing file). ANGLDEGX must be correct because it is required input for the NPF Package when XT3D, K22, or SAVE_SPECIFIC_DISCHARGE is specified.",
            ),
        )
    end
    return nothing
end

"""
    check_npf_options(npf::NpfSpec; newton = false)

NPF option combinations rejected by MF6 (gwf-npf.f90 `check_options`): NEWTON with
PERCHED, VARIABLECV or REWET, and XT3D with ALTERNATIVE_CELL_AVERAGING, THICKSTRT,
PERCHED or VARIABLECV, as one `MF6InputError` with MF6's messages. HIGHEST_CELL_SATURATION
without NEWTON gives MF6's warning (MF6 then turns it off).
"""
function check_npf_options(npf::NpfSpec; newton::Bool = false)
    file = isempty(npf.file) ? "NPF" : npf.file
    errors = String[]
    if newton
        npf.perched && push!(errors, "ERROR IN NPF OPTIONS. NEWTON OPTION CANNOT BE USED WITH PERCHED OPTION.")
        npf.variablecv && push!(errors, "ERROR IN NPF OPTIONS. NEWTON OPTION CANNOT BE USED WITH VARIABLECV OPTION.")
        npf.rewet && push!(errors, "ERROR IN NPF OPTIONS. NEWTON OPTION CANNOT BE USED WITH REWET OPTION.")
    elseif npf.highest_cell_saturation
        @warn "HIGHEST_CELL_SATURATION option cannot be used when NEWTON option in not specified. Resetting HIGHEST_CELL_SATURATION option to off." file
    end
    if npf.xt3d
        npf.cell_averaging !== :harmonic &&
            push!(errors, "ERROR IN NPF OPTIONS. ALTERNATIVE_CELL_AVERAGING OPTION CANNOT BE USED WITH XT3D OPTION.")
        npf.thickstrt && push!(errors, "ERROR IN NPF OPTIONS. THICKSTRT OPTION CANNOT BE USED WITH XT3D OPTION.")
        npf.perched && push!(errors, "ERROR IN NPF OPTIONS. PERCHED OPTION CANNOT BE USED WITH XT3D OPTION.")
        npf.variablecv && push!(errors, "ERROR IN NPF OPTIONS. VARIABLECV OPTION CANNOT BE USED WITH XT3D OPTION.")
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    return nothing
end

# Unit vector from reduced node `n` towards its neighbour across symmetric connection `jj`
# (Dis.f90, Disv.f90, Disu.f90 `connection_normal`): vertical ±z (+1 when the neighbour `m`
# is above, i.e. `m < n`); horizontal from `anglex`, the angle of the normal pointing from
# the lower- to the higher-numbered node, negated when `m < n`. On a DIS grid the two
# directions (right: 0, front: 3π/2) are exact (Dis.f90 sets the components to exactly 0
# and ±1); DISV and DISU use cos/sin of the angle, as MF6 does (T6.4).
@inline function _connection_normal(conn::Connectivity, n::Integer, m::Integer, ihc::Integer, jj::Integer)
    if ihc == 0
        return (0.0, 0.0, m < n ? 1.0 : -1.0)
    end
    a = conn.anglex[jj]
    x, y = if conn.grid !== :dis
        (cos(a), sin(a))
    elseif a == 0.0
        (1.0, 0.0)
    elseif a == 3.0 / 2.0 * Float64(pi)  # Connections.f90 `disconnections`, front
        (0.0, -1.0)
    else
        (cos(a), sin(a))
    end
    return m > n ? (x, y, 0.0) : (-x, -y, 0.0)
end

"""
    hy_eff(p::NpfProperties, n, ihc, vg) -> Float64

Effective hydraulic conductivity of reduced node `n` for a connection of type `ihc` in
the direction `vg = (vg1, vg2, vg3)` (gwf-npf.f90 `hy_eff`): vertical connections use
K33, rotated with ANGLE1/2/3 when ANGLE2 was given; horizontal connections use K, or
`hyeff` with K22 and the angles that were given when K22 was given.
"""
function hy_eff(p::NpfProperties, n::Integer, ihc::Integer, vg::NTuple{3, Float64})
    hy11, hy22, hy33 = p.k11[n], p.k22[n], p.k33[n]
    if ihc == 0
        isnothing(p.angle2) && return hy33
        ang3 = isnothing(p.angle3) ? 0.0 : p.angle3[n]
        return hyeff(hy11, hy22, hy33, p.angle1[n], p.angle2[n], ang3, vg...)
    else
        p.ik22 || return hy11
        ang1 = ang2 = ang3 = 0.0
        if !isnothing(p.angle1)
            ang1 = p.angle1[n]
            if !isnothing(p.angle2)
                ang2 = p.angle2[n]
                isnothing(p.angle3) || (ang3 = p.angle3[n])
            end
        end
        return hyeff(hy11, hy22, hy33, ang1, ang2, ang3, vg...)
    end
end

"""
    calc_condsat(p::NpfProperties, conn::Connectivity) -> Vector{Float64}

Saturated conductance per symmetric connection (`njas`, indexed through `jas`) in Float64
(gwf-npf.f90 `calc_condsat` over all nodes with `upperOnly`, with the initial saturations
`p.sat0`, `calc_initial_sat`): `condmean` of the saturated thicknesses for horizontal
connections, the series conductance of the two half-cell saturated thicknesses for
vertical ones (`vcond` with VARIABLECV DEWATERED and heads at the cell bottoms), with
`hy_eff` in the connection direction.
"""
calc_condsat(p::NpfProperties, conn::Connectivity) = _calc_condsat(p, conn)[1]

# `calc_condsat` and the effective conductivities of each symmetric connection: `hyl` of
# the lower-numbered cell towards the higher, `hyh` the other way (`hy_eff`), as the
# convertible formulation needs them (T6.1).
function _calc_condsat(p::NpfProperties, conn::Connectivity)
    condsat = Vector{Float64}(undef, conn.njas)
    hyl = Vector{Float64}(undef, conn.njas)
    hyh = Vector{Float64}(undef, conn.njas)
    top, bot, sat0 = conn.top, conn.bot, p.sat0
    for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
        m = conn.ja[ii]
        m > n || continue
        jj = conn.jas[ii]
        ihc = conn.ihc[jj]
        vg = _connection_normal(conn, n, m, ihc, jj)
        hyn = hy_eff(p, n, ihc, vg)
        hym = hy_eff(p, m, ihc, .-vg)
        hyl[jj], hyh[jj] = hyn, hym
        topn, botn, topm, botm = top[n], bot[n], top[m], bot[m]
        satn, satm = sat0[n], sat0[m]
        if ihc == 0
            # GwfConductanceUtils.f90 `vcond(1, 1, 1, 1, 0, 1, 1, ...)` with the heads at the
            # cell bottoms: the partially saturated branch with DEWATERED.
            if botn > botm
                satm < 1.0 && (satm = 0.0)
            else
                satn < 1.0 && (satn = 0.0)
            end
            bovk1 = satn * (topn - botn) * 0.5 / hyn
            bovk2 = satm * (topm - botm) * 0.5 / hym
            denom = bovk1 + bovk2
            condsat[jj] = denom != 0.0 ? conn.hwva[jj] / denom : 0.0
        else
            # GwfConductanceUtils.f90 `hcond` → `convertible_standard`.
            if ihc == 2
                thksatn = staggered_thkfrac(topn, botn, satn, topm, botm)
                thksatm = staggered_thkfrac(topm, botm, satm, topn, botn)
            else
                thksatn = satn * (topn - botn)
                thksatm = satm * (topm - botm)
            end
            condsat[jj] = condmean(hyn, hym, thksatn, thksatm, conn.cl1[jj], conn.cl2[jj], conn.hwva[jj], p.icellavg)
        end
    end
    return condsat, hyl, hyh
end

"""
    NpfSetup

Host-side (Float64) result of the NPF setup of one model (`npf_setup`):

- `properties::NpfProperties` by reduced node;
- `condsat` `(njas)`: saturated conductance per symmetric connection (`calc_condsat`);
- `r` `(nodes)`: reference heads (`reference_heads` with the ICELLTYPE of `properties`:
  STRT for confined cells, `clamp(STRT, bot, top)` for convertible cells);
- `Δr` `(njas)`: `r[m] − r[n]` per symmetric connection with `n < m`
  (`reference_differences`; read a CSR entry with `reference_difference`);
- `hyl`, `hyh` `(njas)` (T6.1): the effective K of the lower- and the higher-numbered cell
  of each symmetric connection towards the other (`hy_eff`), for the convertible
  formulation.
"""
struct NpfSetup
    properties::NpfProperties
    condsat::Vector{Float64}
    r::Vector{Float64}
    Δr::Vector{Float64}
    hyl::Vector{Float64}
    hyh::Vector{Float64}
end

"""
    npf_setup(model::GwfSpec, conn::Connectivity = build_connectivity(model)) -> NpfSetup
    npf_setup(npf::NpfSpec, ic::IcSpec, dis::AbstractDisSpec, conn::Connectivity; newton = false) -> NpfSetup

NPF properties (with MF6's checks, `npf_properties`), saturated conductances
(`calc_condsat`) and the reference heads and differences (PLAN.md section 3.7, rule 1) of
a model, all in Float64 on the host.
"""
npf_setup(model::GwfSpec, conn::Connectivity = build_connectivity(model)) =
    npf_setup(model.npf, model.ic, model.dis, conn; newton = model.newton, iconvert = isnothing(model.sto) ? nothing : model.sto.iconvert)

function npf_setup(
        npf::NpfSpec, ic::IcSpec, dis::AbstractDisSpec, conn::Connectivity; newton::Bool = false,
        iconvert::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    )
    p = npf_properties(npf, dis, conn; newton, strt = ic.strt)
    # Reference heads (PLAN.md 3.7 rule 1): clamped into the cell for cells whose
    # saturation is computed (ICELLTYPE /= 0, or STO ICONVERT /= 0), so the offsets of
    # top and bottom stay within a cell thickness.
    conv = Int32[(p.icelltype[n] != 0 || (!isnothing(iconvert) && iconvert[conn.nodeuser[n]] != 0)) ? 1 : 0 for n in 1:conn.nodes]
    r = reference_heads(conn, ic.strt; icelltype = conv)
    condsat, hyl, hyh = _calc_condsat(p, conn)
    return NpfSetup(p, condsat, r, reference_differences(conn, r), hyl, hyh)
end

"""
    NpfModel{T}(setup::NpfSetup)
    NpfModel(T, setup::NpfSetup)

The per-connection NPF arrays of a model in the engine precision `T`: `condsat` and `Δr`
`(njas)`, converted from the Float64 host values of `setup` (one rounding each), and `cv`:
`nothing` for a confined model (every ICELLTYPE 0 after THICKSTRT, the constant-conductance
path), otherwise the `ConvertibleNpf` data of the saturation-dependent conductances (T6.1).
Host arrays here; `NpfModel(backend, T, setup)` or `adapt` puts them on a device.
"""
struct NpfModel{T <: AbstractFloat, V <: AbstractVector{T}, C}
    condsat::V
    Δr::V
    cv::C
end

NpfModel{T}(s::NpfSetup; options = nothing, conn = nothing, newton::Bool = false) where {T <: AbstractFloat} =
    NpfModel(T.(s.condsat), T.(s.Δr), _convertible_npf(Array, T, s, options, conn; newton))
NpfModel(::Type{T}, s::NpfSetup; kw...) where {T <: AbstractFloat} = NpfModel{T}(s; kw...)

"""
    NpfModel(backend, T, setup::NpfSetup; options = nothing, conn = nothing, newton = false)

The same on `backend`: `condsat` and `Δr` rounded to `T` on the host, then copied. With
`options` (the `NpfSpec`) and `conn` given and a cell with ICELLTYPE /= 0, `cv` holds the
convertible data (`ConvertibleNpf`; T6.2: with the Newton formulation when `newton`);
otherwise `cv = nothing`.
"""
function NpfModel(backend::Backend, ::Type{T}, s::NpfSetup; options = nothing, conn = nothing, newton::Bool = false) where {T <: AbstractFloat}
    dev(x) = _device_copy(backend, x)
    return NpfModel(dev(T.(s.condsat)), dev(T.(s.Δr)), _convertible_npf(backend, T, s, options, conn; newton))
end

Adapt.@adapt_structure NpfModel

Base.eltype(::NpfModel{T}) where {T} = T

# A confined NpfModel from device (or host) arrays.
NpfModel(condsat::AbstractVector{T}, Δr::AbstractVector{T}) where {T <: AbstractFloat} = NpfModel(condsat, Δr, nothing)
