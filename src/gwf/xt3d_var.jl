# XT3D and GNC6 with head-dependent coefficients (T6.14): XT3D with convertible cells, the
# XT3D Newton terms, HFB6 and TVK6 on XT3D connections; GNC6 with convertible cells, NEWTON,
# HFB6 and TVK6.
#
# Ports MF6 6.8.2:
# - Xt3dInterface.f90 `xt3d_fc` (the coefficients of every connection at the current
#   saturations; Newton: unit areas, the saturated flow `qsat` and the actual area from the
#   saturation of the cell the saturated flow comes from), `xt3d_areas` (non-Newton:
#   `hwva·sat·thk` per side, the overlap for a staggered DISU connection; Newton: `hwva` times
#   the mean saturated thickness, times the upstream saturation), `xt3d_fn` (the Jacobian
#   term `qsat·S'(h_up)` on the upstream column, for a convertible upstream cell or with the
#   RHS option), `xt3d_fhfb` and `xt3d_flowjahfb` (a barrier on an XT3D connection: the
#   coefficients times `1 − chat01/(chat01 + condhfb)`, or `1 + hydchr` for `hydchr <= 0`,
#   as MF6 writes it), `xt3d_flowja` (flows with the saturations of the last formulation),
#   `xt3d_load` / `xt3d_load_inbr` (an inactive neighbour is left out of a cell's stencil);
#   Xt3dAlgorithm.f90 `qconds` (the weighting of the two sides);
# - gwf-hfb.f90 `hfb_fc`, `hfb_cq` (their XT3D branches: `condhfb` from the head-dependent
#   face area, or `hydchr` with Newton); gwf-npf.f90 `npf_ad` (TVK: K changes recompute the
#   XT3D coefficients);
# - GhostNode.f90 `gnc_fmsav` (the n–m conductance of the matrix after NPF and HFB, here the
#   standard conductance of `connection_terms` at the current heads), `gnc_fc` (implicit and
#   EXPLICIT), `gnc_fn` (Newton: `condsat·α·(h_n − h_j)·S'(h_up)` with the upstream cell by
#   head, for a horizontal connection or with VARIABLECV), `deltaQgnc` (an inactive
#   contributing cell is skipped).
#
# Design. `qconds` is linear in the areas up to the weights: with the unscaled `ahat`/`bhat`
# of the two sides (`a0`, `b0`; `a1`, `b1`; `abhats` with unit area), the coefficients are
#
#     chat01 = w1·a1·ar10,   chati0 = w0·b0·ar01,   chat1j = w1·b1·ar10,
#     w1 = a0·ar01 / (a0·ar01 + a1·ar10),   w0 = 1 − w1,
#
# so each flux term stores its unscaled value (`tc`) and its side (`tside`: 0 the side of
# the lower-numbered cell, 1 the other), and the kernels scale them every outer iteration
# from the saturations (`ConvertibleNpf.sat`, of `npf_cf!`). The unscaled values depend only
# on the geometry, K and which neighbours are active: they are computed on the host in
# Float64 and updated when IBOUND changes (a cell dries or rewets: the connections of the
# cells next to it) or TVK6 changes K (the connections of the changed cells), with a scatter
# of the changed entries. Without ANGLE2 (`nozee`) the connection vectors do not depend on
# the saturation; XT3D with ANGLE2 and convertible cells is an `MF6UnsupportedError`.
#
# One kernel per row (`xvar_formulate_kernel!`) formulates the boundary and storage terms,
# assembles the row from the terms of its connections on the system structure (the
# extended stencil, or the connectivity with the RHS option / EXPLICIT GNC; `posa`, `posb`
# are the positions of each term's two columns in the row, `spos` that of the connection),
# adds the Newton terms, and computes the flux-form residual in offset heads (PLAN.md 3.7
# rules 1–2). Both rows of a connection compute the same coefficients, so the flows are
# exactly antisymmetric. A model with GNC6 on this path keeps the standard conductance
# formulation (`connection_terms`: convertible cells, Newton, HFB6, TVK6) for the n–m
# connections and adds the GNC terms `α·cond·(h_n − h_j)`.

const _DPREC = eps(Float64)  # ConstantsModule DPREC

"""
    xt3d_variable(spec::GwfSpec, icelltype) -> Bool

Whether a model with XT3D or GNC6 runs on the head-dependent path (T6.14,
src/gwf/xt3d_var.jl): with convertible cells, NEWTON, HFB6 or TVK6. Confined models
without them keep the constant coefficients of T6.5/T6.13.
"""
function xt3d_variable(spec::GwfSpec, icelltype::AbstractVector{<:Integer})
    (spec.npf.xt3d || !isnothing(spec.gnc)) || return false
    return spec.newton || any(!=(0), icelltype) || !isnothing(spec.hfb) || !isnothing(spec.npf.tvk)
end

"""
    Xt3dVarSetup

Host (Float64) data of a model on the head-dependent XT3D/GNC path (`xt3d_var_setup`,
`gnc_var_setup`):

- `gnc` (GNC6, else XT3D), `rhs` (the XT3D RHS option or EXPLICIT GNC: only the primary
  terms in the matrix), `newton`, `nozee`;
- the flux terms per symmetric connection `j` (`tptr`, `ta`, `tb`, `tdr = r_a − r_b` as in
  `Xt3dSetup`, primary first): `tc` the unscaled value (XT3D: `a1`, `b0`, `−b1`; GNC: the
  weight `α`; the primary of GNC is the standard conductance) and `tside` (0: scaled with
  the lower-numbered cell's side, 1: the other);
- per connection: `xa0`, `xa1` (the unscaled `ahat` of both sides), `xA` (Newton: the
  saturated area, `hwva` for a vertical connection), `xhw` (`hwva`), `xflag` (`ihc` in bits
  1–2, 4: a barrier on the connection), `xhyd` (its hydraulic characteristic);
- the system structure (`ia`, `ja`, `isym`, `jas`) and, per connectivity entry `ii`, the
  positions in its row of the columns of each term (`kptr[ii]`, `posa`, `posb`; 0 for a
  term not in the matrix) and of the connection itself (`spos`);
- per node: `topoff`, `botoff`, `thk`, `icelltype`, `sat0` (the saturation of a model
  without convertible cells), and `ibound` (the IBOUND the terms were computed with).
"""
struct Xt3dVarSetup
    gnc::Bool
    rhs::Bool
    newton::Bool
    nozee::Bool
    tptr::Vector{Int32}
    ta::Vector{Int32}
    tb::Vector{Int32}
    tc::Vector{Float64}
    tdr::Vector{Float64}
    tside::Vector{Int32}
    xa0::Vector{Float64}
    xa1::Vector{Float64}
    xA::Vector{Float64}
    xhw::Vector{Float64}
    xflag::Vector{Int32}
    xhyd::Vector{Float64}
    ia::Vector{Int32}
    ja::Vector{Int32}
    isym::Vector{Int32}
    jas::Vector{Int32}
    kptr::Vector{Int32}
    posa::Vector{Int32}
    posb::Vector{Int32}
    spos::Vector{Int32}
    topoff::Vector{Float64}
    botoff::Vector{Float64}
    thk::Vector{Float64}
    icelltype::Vector{Int32}
    sat0::Vector{Float64}
    ibound::Vector{Int32}
end

# Xt3dInterface.f90 `xt3d_areas` with `lsat` (Newton): `hwva` for a vertical connection,
# else `hwva` times the mean of the full thicknesses (their overlap for a staggered DISU
# connection).
function _xvar_newton_area(conn::Connectivity, n::Integer, m::Integer, j::Integer)
    ihc = conn.ihc[j]
    ihc == 0 && return conn.hwva[j]
    topn, botn, topm, botm = conn.top[n], conn.bot[n], conn.top[m], conn.bot[m]
    thkn, thkm = topn - botn, topm - botm
    if ihc == 2
        sill_top, sill_bot = min(topn, topm), max(botn, botm)
        thkn = max(min(botn + thkn, sill_top) - sill_bot, 0.0)
        thkm = max(min(botm + thkm, sill_top) - sill_bot, 0.0)
    end
    return conn.hwva[j] * 0.5 * (thkn + thkm)
end

# Position of the neighbour `n` in the connectivity row of `m` (`xt3d_indices`: il10).
_xvar_local(conn::Connectivity, m::Integer, n::Integer) = findfirst(==(n), view(conn.ja, (conn.ia[m] + 1):(conn.ia[m + 1] - 1)))

# The unscaled XT3D terms of connection `jj` (n < m, `il0` the local number of m in n's row)
# written to `tc[tptr[jj]:...]` and `xa0`, `xa1` (`abhats` of both sides with unit area):
# the primary `a1`, then `b0` over n's other neighbours, then `−b1` over m's.
function _xvar_xt3d_terms!(tc, xa0, xa1, tptr, conn::Connectivity, load, n, m, il0, jj)
    c0, c1 = load(n), load(m)
    il10 = _xvar_local(conn, m, n)
    a0, b0 = _xt3d_abhats(c0, il0, 1.0, _XT3D_VCTHRESH)
    a1, b1 = _xt3d_abhats(c1, il10, 1.0, _XT3D_VCTHRESH)
    xa0[jj], xa1[jj] = a0, a1
    t = Int(tptr[jj])
    tc[t] = a1
    for il in 1:c0.nnbr
        il == il0 && continue
        t += 1
        tc[t] = b0[il]
    end
    for il in 1:c1.nnbr
        il == il10 && continue
        t += 1
        tc[t] = -b1[il]
    end
    @assert t == tptr[jj + 1] - 1
    return nothing
end

# Positions of every term's columns in its rows and of each connection (see Xt3dVarSetup).
function _xvar_positions(conn::Connectivity, ia, ja, tptr, ta, tb, rhs::Bool, sorted::Bool)
    nja = Int(conn.nja)
    kptr = Vector{Int32}(undef, nja + 1)
    posa, posb = Int32[], Int32[]
    spos = zeros(Int32, nja)
    for n in 1:Int(conn.nodes), ii in conn.ia[n]:(conn.ia[n + 1] - 1)
        kptr[ii] = length(posa) + 1
        m = conn.ja[ii]
        spos[ii] = _csr_position(ia, ja, n, m, sorted)
        ii == conn.ia[n] && continue
        j = conn.jas[ii]
        for t in tptr[j]:(tptr[j + 1] - 1)
            inmat = !rhs || t == tptr[j]
            push!(posa, inmat ? _csr_position(ia, ja, n, ta[t], sorted) : 0)
            push!(posb, inmat ? _csr_position(ia, ja, n, tb[t], sorted) : 0)
        end
    end
    kptr[nja + 1] = length(posa) + 1
    return kptr, posa, posb, spos
end

function _xvar_nodes(conn::Connectivity, p::NpfProperties, r)
    return conn.top .- r, conn.bot .- r, conn.top .- conn.bot, copy(p.icelltype), copy(p.sat0)
end

function _xvar_conn_data(conn::Connectivity, newton::Bool)
    njas = Int(conn.njas)
    xA = zeros(njas)
    for n in 1:Int(conn.nodes), ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
        m = conn.ja[ii]
        m > n || continue
        j = conn.jas[ii]
        xA[j] = _xvar_newton_area(conn, n, m, j)
    end
    return xA, Float64.(conn.hwva), Int32.(conn.ihc), zeros(njas)
end

"""
    xt3d_var_setup(conn::Connectivity, p::NpfProperties, r; rhs = false, newton = false) -> Xt3dVarSetup

The XT3D terms of a model with convertible cells, NEWTON, HFB6 or TVK6 (T6.14), unscaled,
on the host in Float64, for all cells active: per symmetric connection n–m (n < m) the
primary term and one term per other neighbour of n and of m (a fixed layout: a neighbour
that becomes inactive gets a zero value), the extended stencil (`xt3d_ac`; the
connectivity with the RHS option) and the positions of the terms in it.
"""
function xt3d_var_setup(conn::Connectivity, p::NpfProperties, r::AbstractVector{Float64}; rhs::Bool = false, newton::Bool = false)
    nodes = Int(conn.nodes)
    njas = Int(conn.njas)
    ibound = ones(Int32, nodes)
    nozee = isnothing(p.angle2)
    loads = Dict{Int, Any}()
    load(n) = get!(() -> _xt3d_load(conn, p, ibound, nozee, n), loads, n)
    nnbr(n) = Int(conn.ia[n + 1] - conn.ia[n] - 1)
    tptr = Vector{Int32}(undef, njas + 1)
    ta, tb, tside = Int32[], Int32[], Int32[]
    tptr[1] = 1
    for n in 1:nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
        m = conn.ja[ii]
        m > n || continue
        jj = conn.jas[ii]
        @assert tptr[jj] == length(ta) + 1  # upper triangle in CSR order
        push!(ta, m), push!(tb, n), push!(tside, 1)
        for kk in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
            kk == ii && continue
            push!(ta, conn.ja[kk]), push!(tb, n), push!(tside, 0)
        end
        for kk in (conn.ia[m] + 1):(conn.ia[m + 1] - 1)
            conn.ja[kk] == n && continue
            push!(ta, conn.ja[kk]), push!(tb, m), push!(tside, 1)
        end
        tptr[jj + 1] = length(ta) + 1
        @assert tptr[jj + 1] - tptr[jj] == nnbr(n) + nnbr(m) - 1
    end
    tc = zeros(length(ta))
    xa0, xa1 = zeros(njas), zeros(njas)
    for n in 1:nodes, il0 in 1:nnbr(n)
        ii = conn.ia[n] + il0
        m = conn.ja[ii]
        m > n || continue
        _xvar_xt3d_terms!(tc, xa0, xa1, tptr, conn, load, n, m, il0, conn.jas[ii])
    end
    tdr = [r[ta[t]] - r[tb[t]] for t in eachindex(ta)]
    ia, ja, isym, jas = rhs ? (conn.ia, conn.ja, conn.isym, conn.jas) : _xt3d_structure(conn)
    kptr, posa, posb, spos = _xvar_positions(conn, ia, ja, tptr, ta, tb, rhs, !rhs)
    xA, xhw, xflag, xhyd = _xvar_conn_data(conn, newton)
    topoff, botoff, thk, ict, sat0 = _xvar_nodes(conn, p, r)
    return Xt3dVarSetup(
        false, rhs, newton, nozee, tptr, ta, tb, tc, tdr, tside, xa0, xa1, xA, xhw, xflag, xhyd, ia, ja, isym, jas,
        kptr, posa, posb, spos, topoff, botoff, thk, ict, sat0, ibound,
    )
end

"""
    gnc_var_setup(spec::GwfSpec, conn::Connectivity, setup::NpfSetup) -> Xt3dVarSetup

The GNC6 terms of a model with convertible cells, NEWTON, HFB6 or TVK6 (T6.14): per
connection the standard conductance (a placeholder primary term, formulated by
`connection_terms` on the device) and the GNC terms with their weights `α` (`gnc_terms!`;
`cond` is applied on the device), on the GNC structure (`gnc_ac`; the connectivity with
EXPLICIT).
"""
function gnc_var_setup(spec::GwfSpec, conn::Connectivity, setup::NpfSetup)
    g = spec.gnc
    check_gnc(spec, setup.properties.icelltype)
    n, m, js = _gnc_reduced(g, conn, conn)
    terms = _conductance_terms(conn, setup.condsat)
    gnc_terms!(terms, conn, ones(Int(conn.njas)), n, m, js, g.alphasj; file = isempty(g.file) ? "GNC" : g.file, model1 = spec.name)
    # The side of each GNC term, in the order `gnc_terms!` added them: 0 when the record's
    # n is the lower-numbered cell of the connection.
    sides = [Int32[1] for _ in 1:conn.njas]
    for k in eachindex(n)
        jj = conn.jas[getjaindex(conn, n[k], m[k])]
        for i in axes(js, 1)
            (js[i, k] == 0 || g.alphasj[i, k] == 0.0) && continue
            push!(sides[jj], n[k] < m[k] ? 0 : 1)
        end
    end
    tptr, ta, tb, tc = _flatten_terms(terms)
    tside = reduce(vcat, sides; init = Int32[])
    r = setup.r
    tdr = [r[ta[t]] - r[tb[t]] for t in eachindex(ta)]
    rhs = g.explicit
    ia, ja, isym, jas = rhs ? (conn.ia, conn.ja, conn.isym, conn.jas) : _terms_structure(conn, tptr, ta, tb)
    kptr, posa, posb, spos = _xvar_positions(conn, ia, ja, tptr, ta, tb, rhs, !rhs)
    njas = Int(conn.njas)
    xA, xhw, xflag, xhyd = _xvar_conn_data(conn, spec.newton)
    topoff, botoff, thk, ict, sat0 = _xvar_nodes(conn, setup.properties, r)
    return Xt3dVarSetup(
        true, rhs, spec.newton, true, tptr, ta, tb, tc, tdr, tside, zeros(njas), zeros(njas), xA, xhw, xflag, xhyd,
        ia, ja, isym, jas, kptr, posa, posb, spos, topoff, botoff, thk, ict, sat0, ones(Int32, Int(conn.nodes)),
    )
end

# The system structure of a model on the head-dependent path (as for `Xt3dSetup`).
function CsrStructure(backend::Backend, s::Xt3dVarSetup)
    dev(x) = _device_copy(backend, x)
    nodes = length(s.ia) - 1
    return CsrStructure(
        nodes, length(s.ja), dev(s.ia), dev(s.ja), dev(s.jas), dev(s.isym), dev(s.ia[1:nodes]), dev(collect(Int32(1):Int32(nodes))),
    )
end

"""
    Xt3dVarNpf{T}

The device data of `Xt3dVarSetup` (`Xt3dVarNpf(backend, T, setup)`; values in `T`,
indices `Int32`, Adapt moves it): `gnc`, `rhs`, `newton` and the arrays of the setup. The
model's `xt3d` on the head-dependent path (T6.14).
"""
struct Xt3dVarNpf{T <: AbstractFloat, V <: AbstractVector{T}, VI <: AbstractVector{Int32}}
    gnc::Bool
    rhs::Bool
    newton::Bool
    tptr::VI
    ta::VI
    tb::VI
    tside::VI
    tc::V
    tdr::V
    xa0::V
    xa1::V
    xA::V
    xhw::V
    xflag::VI
    xhyd::V
    kptr::VI
    posa::VI
    posb::VI
    spos::VI
    topoff::V
    botoff::V
    thk::V
    icelltype::VI
    sat0::V
end

Adapt.@adapt_structure Xt3dVarNpf

function Xt3dVarNpf(backend::Backend, ::Type{T}, s::Xt3dVarSetup) where {T <: AbstractFloat}
    di(x) = _device_copy(backend, x)
    dt(x) = _device_copy(backend, T.(x))
    return Xt3dVarNpf(
        s.gnc, s.rhs, s.newton, di(s.tptr), di(s.ta), di(s.tb), di(s.tside), dt(s.tc), dt(s.tdr), dt(s.xa0), dt(s.xa1),
        dt(s.xA), dt(s.xhw), di(s.xflag), dt(s.xhyd), di(s.kptr), di(s.posa), di(s.posb), di(s.spos), dt(s.topoff),
        dt(s.botoff), dt(s.thk), di(s.icelltype), dt(s.sat0),
    )
end

# --- Device functions -------------------------------------------------------------------

# Xt3dInterface.f90 `xt3d_areas` (non-Newton, `lsat` false) of connection `j` between
# `lo < hi`: `hwva` (vertical), else `hwva` times the saturated thickness of each side (the
# overlap with the other cell for a staggered DISU connection, in the frame of `lo`).
@inline function _xvar_areas(x::Xt3dVarNpf{T}, Δr, sat, lo, hi, j, ihc) where {T}
    return @inbounds begin
        hw = x.xhw[j]
        if ihc == Int32(0)
            hw, hw
        else
            sn, sm = sat[lo], sat[hi]
            if ihc == Int32(2)
                dr = Δr[j]
                topn, botn = x.topoff[lo], x.botoff[lo]
                topm, botm = x.topoff[hi] + dr, x.botoff[hi] + dr
                hw * staggered_thkfrac(topn, botn, sn, topm, botm), hw * staggered_thkfrac(topm, botm, sm, topn, botn)
            else
                hw * (sn * x.thk[lo]), hw * (sm * x.thk[hi])
            end
        end
    end
end

# gwf-hfb.f90 `hfb_fc` (XT3D branch, non-Newton): the face area of a barrier, `hwva` for a
# vertical connection, else `hwva` times the face height from the cell tops (the head when
# below the top, for ICELLTYPE 1) and bottoms (the overlap for `ihc` 2, else the mean).
@inline function _xvar_faarea(x::Xt3dVarNpf{T}, Δr, h, lo, hi, j, ihc) where {T}
    return @inbounds begin
        if ihc == Int32(0)
            x.xhw[j]
        else
            dr = Δr[j]
            topn, botn = x.topoff[lo], x.botoff[lo]
            (x.icelltype[lo] == Int32(1) && h[lo] < topn) && (topn = h[lo])
            topm, botm = x.topoff[hi], x.botoff[hi]
            (x.icelltype[hi] == Int32(1) && h[hi] < topm) && (topm = h[hi])
            fh = ihc == Int32(2) ? min(topn, topm + dr) - max(botn, botm + dr) : T(0.5) * ((topn - botn) + (topm - botm))
            x.xhw[j] * fh
        end
    end
end

# Add `v` to entry `pos` of the row whose diagonal is `sid`: the diagonal is accumulated in
# `dsum`, an off-diagonal written (`+=`) only with `offdiag`.
@inline function _xvar_add(amat, sid, pos, v, offdiag::Bool, dsum)
    pos == sid && return dsum + v
    offdiag && (@inbounds amat[pos] += v)
    return dsum
end

# GhostNode.f90 `gnc_fn`: the Newton term of GNC term `t` of connection `j` (between
# `lo < hi`) in row `n`, with `dt = h_a − h_b + Δr` of the term; nothing without a
# ConvertibleNpf with NEWTON. Returns the updated `dsum` and the right-hand-side amount.
@inline _xvar_gnc_jac(cond::AbstractVector, x, Δr, h, n, m, lo, hi, j, t, w, dt, ii, sid, amat, offdiag, dsum) = (dsum, zero(eltype(h)))
@inline function _xvar_gnc_jac(cv::ConvertibleNpf{T}, x, Δr, h, n, m, lo, hi, j, t, w, dt, ii, sid, amat, offdiag, dsum) where {T}
    cv.newton || return dsum, zero(T)
    return @inbounds begin
        ihc = cv.ihcx[j] & Int32(3)
        side0 = x.tside[t] == Int32(0)
        rn, rm = side0 ? (lo, hi) : (hi, lo)
        # Upstream: the record's m when its head is higher (absolute heads; `Δr` moves m
        # into the frame of n).
        drnm = rm > rn ? Δr[j] : -Δr[j]
        up = h[rm] - h[rn] + drnm > zero(T) ? rm : rn
        if (ihc == Int32(0) && !cv.ivarcv) || cv.icelltype[up] == Int32(0)
            dsum, zero(T)
        else
            tp, bt = cv.topoff[up], cv.botoff[up]
            if ihc == Int32(2)
                other = up == lo ? hi : lo
                dr = other > up ? Δr[j] : -Δr[j]
                tp = min(tp, cv.topoff[other] + dr)
                bt = max(bt, cv.botoff[other] + dr)
            end
            consterm = cv.condsat[j] * w * (side0 ? dt : -dt)
            g = consterm * newton_saturation_derivative(tp, bt, h[up])
            g = n == rn ? g : -g
            pos = up == n ? sid : x.spos[ii]
            _xvar_add(amat, sid, pos, g, offdiag, dsum), g * h[up]
        end
    end
end

"""
    xvar_connection(x::Xt3dVarNpf, cond, Δr, jas, sat, h, ibound, n, m, ii, sid, amat, asm, offdiag, jac, dsum)
        -> (q, qa, dsum, sadd)

The flow `q` into `n` from `m` (connectivity entry `ii`) of a model on the head-dependent
XT3D/GNC path at the offset heads `h` and saturations `sat`, and `qa`, the sum of the
magnitudes of its terms. With `asm` the matrix terms of row `n` (diagonal `sid` of the
system structure) are added: off-diagonals to `amat` (with `offdiag`), the diagonal to
`dsum`; `sadd` is the amount for the offset right-hand side. `jac` adds the Newton terms
(`xt3d_fn`, `gnc_fn`). `cond` is the standard conductance argument (`condsat` or the
`ConvertibleNpf`) for GNC. Device-safe.
"""
@inline function xvar_connection(x::Xt3dVarNpf{T}, cond, Δr, jas, sat, h, ibound, n, m, ii, sid, amat, asm::Bool, offdiag::Bool, jac::Bool, dsum) where {T}
    return @inbounds begin
        j = jas[ii]
        lo, hi = n < m ? (n, m) : (m, n)
        σ = n == lo ? one(T) : -one(T)
        t1 = x.tptr[j]
        t2 = x.tptr[j + Int32(1)] - Int32(1)
        k0 = x.kptr[ii] - t1
        sadd = zero(T)
        s = zero(T)
        c = zero(T)
        qa = zero(T)
        if x.gnc
            # The standard conductance (with convertible cells, Newton, HFB6, TVK6) and the
            # n–m conductance of each side (`gnc_fmsav`: the matrix entry of the record's n).
            a, dd, sb, q0, _, _ = connection_terms(cond, Δr, jas, ibound, h, n, m, ii, jac)
            if asm
                dsum -= dd
                offdiag && (amat[x.spos[ii]] += a)
                sadd += sb
            end
            qa += abs(q0)
            clo = connection_terms(cond, Δr, jas, ibound, h, lo, hi, ii, false)[1]
            chi = connection_terms(cond, Δr, jas, ibound, h, hi, lo, ii, false)[1]
            for t in (t1 + Int32(1)):t2
                ta, tb = x.ta[t], x.tb[t]
                (ibound[ta] == Int32(0) || ibound[tb] == Int32(0)) && continue
                w = x.tc[t]
                cf = w * (x.tside[t] == Int32(0) ? clo : chi)
                dt = x.tdr[t]
                d = h[ta] - h[tb] + dt
                v = cf * d
                s, c = _sum_add(s, c, v)
                qa += abs(v)
                if asm
                    if !x.rhs
                        k = k0 + t
                        dsum = _xvar_add(amat, sid, x.posa[k], σ * cf, offdiag, dsum)
                        dsum = _xvar_add(amat, sid, x.posb[k], -σ * cf, offdiag, dsum)
                        sadd -= σ * cf * dt
                    else
                        sadd -= σ * v
                    end
                    if jac
                        dsum, sj = _xvar_gnc_jac(cond, x, Δr, h, n, m, lo, hi, j, t, w, d, ii, sid, amat, offdiag, dsum)
                        sadd += sj
                    end
                end
            end
            q0 + σ * (s + c), qa, dsum, sadd
        else
            fl = x.xflag[j]
            ihc = fl & Int32(3)
            # qconds: the weights of the two sides at the current areas (Newton: unit
            # areas, the actual area applied below).
            a01, a10 = x.newton ? (one(T), one(T)) : _xvar_areas(x, Δr, sat, lo, hi, j, ihc)
            f0 = zero(T)
            f1 = zero(T)
            if a01 != zero(T)
                d0 = x.xa0[j] * a01
                den = d0 + x.xa1[j] * a10
                w1 = abs(den) > T(_DPREC) ? d0 / den : one(T)
                f0 = (one(T) - w1) * a01
                f1 = w1 * a10
            end
            # xt3d_fhfb: a barrier scales the coefficients by 1 − term.
            term = zero(T)
            if (fl & Int32(4)) != Int32(0)
                chr = x.xhyd[j]
                chat01 = f1 * x.tc[t1]
                condhfb = chr > zero(T) ? (x.newton ? chr : chr * _xvar_faarea(x, Δr, h, lo, hi, j, ihc)) : chr
                term = condhfb > zero(T) ? chat01 / (chat01 + condhfb) : -condhfb
            end
            qt = zero(T)
            if x.newton
                # xt3d_fc (Newton): the saturated flow, the saturation of the cell it comes
                # from (the flow with the barrier for the barrier's part, xt3d_fhfb).
                A = x.xA[j]
                for t in t1:t2
                    v = x.tc[t] * (x.tside[t] == Int32(0) ? f0 : f1) * A * (h[x.ta[t]] - h[x.tb[t]] + x.tdr[t])
                    s, c = _sum_add(s, c, v)
                end
                qs = s + c
                s = zero(T)
                c = zero(T)
                horiz = ihc != Int32(0)
                satm = horiz ? (qs < zero(T) ? sat[lo] : sat[hi]) : one(T)
                qt = qs * (one(T) - term)
                sath = horiz ? (qt < zero(T) ? sat[lo] : sat[hi]) : one(T)
                mult = A * (satm - term * sath)
            else
                mult = one(T) - term
            end
            g0 = f0 * mult
            g1 = f1 * mult
            for t in t1:t2
                cf = x.tc[t] * (x.tside[t] == Int32(0) ? g0 : g1)
                ta, tb = x.ta[t], x.tb[t]
                dt = x.tdr[t]
                v = cf * (h[ta] - h[tb] + dt)
                s, c = _sum_add(s, c, v)
                qa += abs(v)
                if asm
                    if !x.rhs || t == t1
                        k = k0 + t
                        dsum = _xvar_add(amat, sid, x.posa[k], σ * cf, offdiag, dsum)
                        dsum = _xvar_add(amat, sid, x.posb[k], -σ * cf, offdiag, dsum)
                        sadd -= σ * cf * dt
                    else
                        # xt3d_rhs: the other terms on the right-hand side.
                        sadd -= σ * v
                    end
                end
            end
            if asm && jac && x.newton
                # xt3d_fn: `qsat·S'(h_up)` on the column of the upstream cell (by the
                # direction of the saturated flow), for a convertible upstream cell or with
                # the RHS option.
                up = qt < zero(T) ? lo : hi
                if x.icelltype[up] != Int32(0) || x.rhs
                    tp, bt = x.topoff[up], x.botoff[up]
                    if ihc == Int32(2)
                        other = up == lo ? hi : lo
                        dr = other > up ? Δr[j] : -Δr[j]
                        tp = min(tp, x.topoff[other] + dr)
                        bt = max(bt, x.botoff[other] + dr)
                    end
                    jt = σ * qt * newton_saturation_derivative(tp, bt, h[up])
                    dsum = _xvar_add(amat, sid, up == n ? sid : x.spos[ii], jt, offdiag, dsum)
                    sadd += jt * h[up]
                end
            end
            σ * (s + c), qa, dsum, sadd
        end
    end
end

@kernel function xvar_formulate_kernel!(
        amat, rhs, diag, bdiag, brhs, res, scale, @Const(ia), @Const(ja), @Const(jas), @Const(sia), x, cond, @Const(Δr),
        @Const(sat), @Const(h), @Const(ibound), pkgs, sto, offdiag::Bool, terms::Bool, tiny, jac::Bool,
    )
    n = @index(Global)
    T = eltype(amat)
    @inbounds begin
        bd, br, jd = _add_package_terms(n, zero(T), zero(T), zero(T), pkgs, h, ibound)
        bd, br, jd = _add_storage_terms(n, bd, br, jd, sto, ibound, h, jac)
        if terms
            bdiag[n] = bd
            brhs[n] = br
        end
        sid = sia[n]
        si2 = sia[n + 1] - Int32(1)
        rs = zero(T)
        rc = zero(T)
        sc = zero(T)
        if ibound[n] > 0
            hn = h[n]
            if offdiag
                for ii in (sid + Int32(1)):si2
                    amat[ii] = zero(T)
                end
            end
            s = br
            dsum = zero(T)
            for ii in (ia[n] + Int32(1)):(ia[n + 1] - Int32(1))
                m = ja[ii]
                ibound[m] != 0 || continue
                q, qa, dsum, sadd = xvar_connection(x, cond, Δr, jas, sat, h, ibound, n, m, ii, sid, amat, true, offdiag, jac, dsum)
                rs, rc = _sum_add(rs, rc, -q)
                sc += qa
                s += sadd
            end
            d = dsum + bd
            if jac
                # Newton–Raphson diagonal terms of the packages and storage (T6.2).
                d += jd
                s += jd * hn
            end
            if abs(d) < tiny
                d = -one(T)
                s -= hn
            end
            amat[sid] = d
            diag[n] = d
            terms && (rhs[n] = s)
            tb_ = bd * hn
            rs, rc = _sum_add(rs, rc, -tb_)
            rs, rc = _sum_add(rs, rc, br)
            sc += abs(tb_) + abs(br)
        else
            amat[sid] = one(T)
            diag[n] = one(T)
            if offdiag
                for ii in (sid + Int32(1)):si2
                    amat[ii] = zero(T)
                end
            end
            terms && (rhs[n] = h[n])
        end
        res[n] = rs + rc
        scale === nothing || (scale[n] = sc)
    end
end

# The saturation argument of the kernels: the ConvertibleNpf's (`npf_cf!`), else constant.
_xvar_sat(x::Xt3dVarNpf, npf) = isnothing(npf.cv) ? x.sat0 : npf.cv.sat

"""
    xvar_formulate_system!(sys, res, csr, scsr, x::Xt3dVarNpf, npf, h, ibound, packages, storage;
                           offdiag = true, terms = true, scale = nothing, jac = false) -> sys

`formulate_system!` of a model on the head-dependent XT3D/GNC path (T6.14): per row the
boundary and storage terms, the XT3D or GNC terms of its connections scaled at the current
saturations (`npf.cv.sat`) and assembled on the system structure `scsr`, the Newton terms
with `jac`, `sln_ls`'s adjustments, and the flux-form residual `res` over the connectivity
`csr`. Arguments as `xt3d_formulate_system!`.
"""
function xvar_formulate_system!(
        sys::LinearSystem, res, csr::CsrStructure, scsr::CsrStructure, x::Xt3dVarNpf, npf::NpfModel, h, ibound, packages::Tuple, storage;
        offdiag::Bool = true, terms::Bool = true, scale = nothing, jac::Bool = false,
    )
    backend = get_backend(sys.amat)
    xvar_formulate_kernel!(backend, _WG)(
        sys.amat, sys.rhs, sys.diag, sys.bdiag, sys.brhs, res, scale, csr.ia, csr.ja, csr.jas, scsr.ia, x, _cond_arg(npf), npf.Δr,
        _xvar_sat(x, npf), h, ibound, packages, storage, offdiag, terms, eltype(sys)(1.0e-15), jac; ndrange = csr.nodes,
    )
    return sys
end

@kernel function xvar_flowja_kernel!(flowja, @Const(ia), @Const(ja), @Const(jas), @Const(irow), x, cond, @Const(Δr), @Const(sat), @Const(h), @Const(ibound))
    ii = @index(Global)
    T = eltype(flowja)
    @inbounds begin
        n = irow[ii]
        if ii == ia[n]
            flowja[ii] = zero(T)
        else
            m = ja[ii]
            if ibound[n] == 0 || ibound[m] == 0
                flowja[ii] = zero(T)
            else
                q, _, _, _ = xvar_connection(x, cond, Δr, jas, sat, h, ibound, n, m, ii, Int32(0), flowja, false, false, false, zero(T))
                flowja[ii] = q
            end
        end
    end
end

"""
    xvar_flowja!(flowja, csr, x::Xt3dVarNpf, npf, h, ibound) -> flowja

Connection flows of a model on the head-dependent XT3D/GNC path (`xt3d_flowja`,
`xt3d_flowjahfb`, `npf_cq` with `gnc_cq`) at the heads `h` with the saturations of the last
formulation; diagonal entries zeroed, exactly antisymmetric.
"""
function xvar_flowja!(flowja, csr::CsrStructure, x::Xt3dVarNpf, npf::NpfModel, h, ibound)
    xvar_flowja_kernel!(get_backend(flowja), _WG)(
        flowja, csr.ia, csr.ja, csr.jas, csr.irow, x, _cond_arg(npf), npf.Δr, _xvar_sat(x, npf), h, ibound; ndrange = csr.nja,
    )
    return flowja
end

# --- Host updates (IBOUND, TVK6, HFB6) ----------------------------------------------------

@kernel function xvar_scatter_kernel!(dst, @Const(idx), @Const(val))
    i = @index(Global)
    @inbounds dst[idx[i]] = val[i]
end

# Write `vals` at `idx` of the device array `dst` (one launch; nothing when empty).
function _xvar_scatter!(dst, idx::Vector{Int32}, vals::AbstractVector)
    isempty(idx) && return nothing
    backend = get_backend(dst)
    xvar_scatter_kernel!(backend, _WG)(dst, _device_copy(backend, idx), _device_copy(backend, eltype(dst).(vals)); ndrange = length(idx))
    return nothing
end

"""
    xvar_recompute!(xs::Xt3dVarSetup, x::Xt3dVarNpf, conn, p::NpfProperties, nodes) -> Bool

Recompute on the host the unscaled XT3D terms of every connection of `nodes` with the
current properties `p` (TVK6) and IBOUND `xs.ibound` (`xt3d_load`: inactive neighbours left
out), and write the changed values to the device. Returns whether a value changed.
"""
function xvar_recompute!(xs::Xt3dVarSetup, x::Xt3dVarNpf, conn::Connectivity, p::NpfProperties, nodes)
    xs.gnc && return false
    loads = Dict{Int, Any}()
    load(n) = get!(() -> _xt3d_load(conn, p, xs.ibound, xs.nozee, n), loads, n)
    js = Set{Int32}()
    it, ij0, ij1 = Int32[], Int32[], Int32[]
    # Host work in proportion to the connections recomputed (not to the model size): only
    # their own term ranges are compared and scattered.
    for n in nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
        j = conn.jas[ii]
        j in js && continue
        push!(js, j)
        m = conn.ja[ii]
        lo, hi = min(n, m), max(n, m)
        (xs.ibound[lo] == 0 || xs.ibound[hi] == 0) && continue
        rng = xs.tptr[j]:(xs.tptr[j + 1] - 1)
        old, a0, a1 = xs.tc[rng], xs.xa0[j], xs.xa1[j]
        _xvar_xt3d_terms!(xs.tc, xs.xa0, xs.xa1, xs.tptr, conn, load, lo, hi, _xvar_local(conn, lo, hi), j)
        for (k, t) in enumerate(rng)
            xs.tc[t] !== old[k] && push!(it, t)
        end
        xs.xa0[j] !== a0 && push!(ij0, j)
        xs.xa1[j] !== a1 && push!(ij1, j)
    end
    _xvar_scatter!(x.tc, it, xs.tc[it])
    _xvar_scatter!(x.xa0, ij0, xs.xa0[ij0])
    _xvar_scatter!(x.xa1, ij1, xs.xa1[ij1])
    return !(isempty(it) && isempty(ij0) && isempty(ij1))
end

"""
    xvar_ibound!(xs::Xt3dVarSetup, x::Xt3dVarNpf, conn, p, ibound) -> Bool

After IBOUND changed (`wetdry!`, a cell dried or rewetted; `apply_period!`): the XT3D terms
of the connections of the cells next to a changed cell (their stencils, `xt3d_load`) are
recomputed (`xvar_recompute!`). Returns whether a value changed.
"""
function xvar_ibound!(xs::Xt3dVarSetup, x::Xt3dVarNpf, conn::Connectivity, p::NpfProperties, ibound::AbstractVector)
    xs.gnc && return false
    changed = Int32[n for n in eachindex(ibound) if (ibound[n] == 0) != (xs.ibound[n] == 0)]
    isempty(changed) && return false
    for n in changed
        xs.ibound[n] = ibound[n] == 0 ? 0 : 1
    end
    nodes = Set{Int32}(changed)
    for n in changed, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
        push!(nodes, conn.ja[ii])
    end
    return xvar_recompute!(xs, x, conn, p, sort!(collect(nodes)))
end

"""
    xvar_hfb!(xs::Xt3dVarSetup, x::Xt3dVarNpf, old, jas, hydchr)

HFB6 on XT3D connections (`xt3d_fhfb`): the barriers of the previous block (connections
`old`) are removed and the new ones (`jas`, `hydchr`) set (`xflag` bit 4, `xhyd`).
"""
function xvar_hfb!(xs::Xt3dVarSetup, x::Xt3dVarNpf, old::AbstractVector{<:Integer}, jas::AbstractVector{<:Integer}, hydchr::AbstractVector{Float64})
    idx = sort!(unique!(Int32[old; jas]))
    isempty(idx) && return nothing
    for j in old
        xs.xflag[j] &= ~Int32(4)
        xs.xhyd[j] = 0.0
    end
    for (j, c) in zip(jas, hydchr)
        xs.xflag[j] |= Int32(4)
        xs.xhyd[j] = c
    end
    _xvar_scatter!(x.xflag, idx, xs.xflag[idx])
    _xvar_scatter!(x.xhyd, idx, xs.xhyd[idx])
    return nothing
end
