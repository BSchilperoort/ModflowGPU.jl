# Convertible cells and wetting/drying, standard (non-Newton) formulation (T6.1).
#
# Ports MF6 6.8.2:
# - gwf-npf.f90 `cf_default_flow` / `sgwf_npf_thksat` (saturation per cell), `fc_default_flow`
#   (VARIABLECV, PERCHED), `sgwf_npf_qcalc` (flows), `sgwf_npf_wetdry` / `rewet_check`
#   (REWET: WETFCT, IWETIT, IHDWET; drying of convertible cells), `npf_ad` (old head of a
#   dry wettable cell = its bottom);
# - GwfConductanceUtils.f90 `hcond` / `convertible_standard`, `vcond`.
#
# Precision (PLAN.md 3.7 rule 1): heads are offsets from the cell references `r` (clamped
# into the cell for convertible cells), so the saturation is computed from `h' − bot'` with
# `top' = top − r`, `bot' = bot − r` and `thk = top − bot`, all computed on the host in
# Float64 and rounded once to `T`. Elevations of the neighbour are moved into the cell's
# frame with the reference difference `Δr` (staggered DISU connections only).
#
# Confined models (every ICELLTYPE 0 after THICKSTRT) do not use any of this: their
# `NpfModel.cv` is `nothing` and the kernels take the constant-`condsat` path.

"""
    ConvertibleNpf{T}

Device data of the saturation-dependent NPF conductances of a model with convertible cells
(T6.1), in the engine precision `T`:

- per node: `icelltype` (`Int32`, 0 confined), `topoff`, `botoff` (`top − r`, `bot − r`),
  `thk` (`top − bot`), and `sat`, the saturation of the last `npf_cf!` (`cf_default_flow`;
  1, or the THICKSTRT saturation, for confined cells);
- per symmetric connection (`njas`): `condsat`, `Δr`, `hyl`/`hyh` (effective K of the
  lower/higher-numbered cell towards the other), `cl1`, `cl2`, `hwva`, and `ihcx` (`ihc`
  plus 4 when the bottom of the lower-numbered cell is above that of the other);
- options: `ivarcv` (VARIABLECV), `idewatcv` (DEWATERED), `iperched` (PERCHED), `icellavg`.

Moved to a device with Adapt.
"""
struct ConvertibleNpf{T <: AbstractFloat, V <: AbstractVector{T}, VI <: AbstractVector{Int32}}
    icelltype::VI
    topoff::V
    botoff::V
    thk::V
    sat::V
    condsat::V
    Δr::V
    hyl::V
    hyh::V
    cl1::V
    cl2::V
    hwva::V
    ihcx::VI
    ivarcv::Bool
    idewatcv::Bool
    iperched::Bool
    icellavg::Int32
end

Adapt.@adapt_structure ConvertibleNpf

Base.eltype(::ConvertibleNpf{T}) where {T} = T

# The ConvertibleNpf of a setup, or `nothing` for a confined model (or without options).
function _convertible_npf(backend, ::Type{T}, s::NpfSetup, options, conn) where {T}
    (isnothing(options) || isnothing(conn)) && return nothing
    p = s.properties
    all(==(0), p.icelltype) && return nothing
    dev(x) = backend === Array ? x : _device_copy(backend, x)
    top, bot, r = conn.top, conn.bot, s.r
    ihcx = Vector{Int32}(undef, conn.njas)
    for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
        m = conn.ja[ii]
        m > n || continue
        j = conn.jas[ii]
        ihcx[j] = conn.ihc[j] + (bot[n] > bot[m] ? 4 : 0)
    end
    return ConvertibleNpf(
        dev(p.icelltype), dev(T.(top .- r)), dev(T.(bot .- r)), dev(T.(top .- bot)), dev(T.(p.sat0)),
        dev(T.(s.condsat)), dev(T.(s.Δr)), dev(T.(s.hyl)), dev(T.(s.hyh)), dev(T.(conn.cl1)), dev(T.(conn.cl2)),
        dev(T.(conn.hwva)), dev(ihcx), options.variablecv, options.dewatered, options.perched, Int32(p.icellavg),
    )
end

"""
    thksat(h, topoff, botoff, thk) -> sat

Saturated fraction of a convertible cell, standard formulation (gwf-npf.f90
`sgwf_npf_thksat`): 1 when the head is at or above the top, else `(h − bot)/(top − bot)`
(not clamped below; a cell with `h <= bot` dries in `wetdry!`). Offsets: `h`, `topoff`,
`botoff` from the same reference, `thk = top − bot`. Device-safe.
"""
@inline thksat(h::T, topoff::T, botoff::T, thk::T) where {T} = h >= topoff ? one(T) : (h - botoff) / thk

# Conductance of the symmetric connection `j` between `lo < hi` (`hcond`/`vcond` of
# GwfConductanceUtils.f90 with the saturations of the last `npf_cf!`), and whether the
# flow is the PERCHED flow from `lo` (`fc_default_flow`, `sgwf_npf_qcalc`).
@inline function _cv_conductance(cv::ConvertibleNpf{T}, ibound, h, lo, hi, j) where {T}
    @inbounds begin
        (ibound[lo] == 0 || ibound[hi] == 0) && return zero(T), false
        ictl = cv.icelltype[lo]
        icth = cv.icelltype[hi]
        csat = cv.condsat[j]
        code = cv.ihcx[j]
        ihc = code & Int32(3)
        if ihc == Int32(0)
            c = csat
            if cv.ivarcv && (ictl != Int32(0) || icth != Int32(0)) && !(h[lo] >= cv.topoff[lo] && h[hi] >= cv.topoff[hi])
                satn = cv.sat[lo]
                satm = cv.sat[hi]
                lobot_higher = code >= Int32(4)
                if !cv.idewatcv
                    lobot_higher ? (satm = one(T)) : (satn = one(T))
                else
                    if lobot_higher
                        satm < one(T) && (satm = zero(T))
                    else
                        satn < one(T) && (satn = zero(T))
                    end
                end
                bovk1 = satn * cv.thk[lo] * T(0.5) / cv.hyl[j]
                bovk2 = satm * cv.thk[hi] * T(0.5) / cv.hyh[j]
                denom = bovk1 + bovk2
                c = denom != zero(T) ? cv.hwva[j] / denom : zero(T)
            end
            perched = cv.iperched && icth != Int32(0) && h[hi] < cv.topoff[hi]
            return c, perched
        else
            (ictl == Int32(0) && icth == Int32(0)) && return csat, false
            satn = cv.sat[lo]
            satm = cv.sat[hi]
            if ihc == Int32(2)
                # Elevations of `hi` in the frame of `lo`.
                dr = cv.Δr[j]
                topn, botn = cv.topoff[lo], cv.botoff[lo]
                topm, botm = cv.topoff[hi] + dr, cv.botoff[hi] + dr
                thn = staggered_thkfrac(topn, botn, satn, topm, botm)
                thm = staggered_thkfrac(topm, botm, satm, topn, botn)
            else
                thn = satn * cv.thk[lo]
                thm = satm * cv.thk[hi]
            end
            return condmean(cv.hyl[j], cv.hyh[j], thn, thm, cv.cl1[j], cv.cl2[j], cv.hwva[j], cv.icellavg), false
        end
    end
end

"""
    connection_terms(cond, Δr, jas, ibound, h, n, m, ii) -> (a, dd, sb, q, c, dr)

The terms of off-diagonal entry `ii` (row `n`, column `m`) for the row kernels: the matrix
entry `a`, the amount `dd` subtracted from the diagonal, the amount `sb` added to the offset
right-hand side, the flow `q` into `n` at the offset heads `h`, the conductance `c` and the
reference difference `dr = r_m − r_n`. `cond` is `condsat` (a confined model: `a = dd = c`,
`sb = −c·dr`, `q = c·(h'_m − h'_n + dr)`) or a `ConvertibleNpf`; with PERCHED flow from the
upper cell `lo` to `hi` (`fc_default_flow`) the row of `lo` gets `dd = c`, `sb = −c·bot'_lo`,
`a = 0` and the row of `hi` `a = c`, `dd = 0`, `sb = c·bot'_lo`. Device-safe.
"""
@inline function connection_terms(condsat::AbstractVector, Δr, jas, ibound, h, n, m, ii)
    c, dr = _connection(condsat, Δr, jas, ibound, n, m, ii)
    return c, c, -(c * dr), c * (h[m] - h[n] + dr), c, dr
end

@inline function connection_terms(cv::ConvertibleNpf{T}, Δr, jas, ibound, h, n, m, ii) where {T}
    @inbounds begin
        j = jas[ii]
        dr = m > n ? Δr[j] : -Δr[j]
        lo, hi = n < m ? (n, m) : (m, n)
        c, perched = _cv_conductance(cv, ibound, h, lo, hi, j)
        if !perched
            return c, c, -(c * dr), c * (h[m] - h[n] + dr), c, dr
        end
        b = cv.botoff[lo]
        if n == lo
            return zero(T), c, -(c * b), c * (b - h[n]), c, dr
        else
            return c, zero(T), c * b, c * (h[m] - b), c, dr
        end
    end
end

# The conductance argument of the row kernels: `condsat` (confined) or the ConvertibleNpf.
_cond_arg(npf) = isnothing(npf.cv) ? npf.condsat : npf.cv

@kernel function npf_sat_kernel!(sat, @Const(icelltype), @Const(topoff), @Const(botoff), @Const(thk), @Const(h), @Const(ibound))
    n = @index(Global)
    @inbounds if icelltype[n] != Int32(0)
        sat[n] = ibound[n] == Int32(0) ? zero(eltype(sat)) : thksat(h[n], topoff[n], botoff[n], thk[n])
    end
end

"""
    npf_cf!(cv::ConvertibleNpf, h, ibound) -> cv

The saturation `cv.sat` of every convertible cell from the offset heads `h`
(`cf_default_flow`): 0 for a dry cell (`ibound == 0`), `thksat` otherwise. Confined cells
keep their saturation. One work item per node.
"""
function npf_cf!(cv::ConvertibleNpf, h, ibound)
    npf_sat_kernel!(get_backend(h), _WG)(cv.sat, cv.icelltype, cv.topoff, cv.botoff, cv.thk, h, ibound; ndrange = length(h))
    return cv
end

# ---------------------------------------------------------------------------------------
# Wetting and drying (gwf-npf.f90 `sgwf_npf_wetdry`, `rewet_check`, `npf_ad`)
# ---------------------------------------------------------------------------------------

"""
    WetDry

Wetting and drying state of a model with convertible cells (T6.1; device arrays in `T`):
`wetdry` (WETDRY by node, 0 for cells that cannot rewet; empty without REWET), the rewet
work arrays `flag`/`flag2` (`Int32`) and `hwet`/`hwet2` (Jacobi passes, see `wetdry!`),
`count` (`Int32[3]`: a pass changed something, IBOUND changed, a constant-head cell went
dry), and the host options `rewet`, `wetfct`, `iwetit`, `ihdwet`.
"""
struct WetDry{V, VI}
    wetdry::V
    flag::VI
    flag2::VI
    hwet::V
    hwet2::V
    count::VI
    rewet::Bool
    wetfct::Float64
    iwetit::Int
    ihdwet::Int
end

Adapt.@adapt_structure WetDry

function WetDry(backend::Backend, ::Type{T}, p::NpfProperties, npf::NpfSpec, nodes::Integer) where {T}
    rewet = npf.rewet && !isnothing(p.wetdry)
    nw = rewet ? Int(nodes) : 0
    zi(k) = KernelAbstractions.zeros(backend, Int32, k)
    zt(k) = KernelAbstractions.zeros(backend, T, k)
    wd = rewet ? _device_copy(backend, T.(p.wetdry)) : zt(0)
    return WetDry(wd, zi(nw), zi(nw), zt(nw), zt(nw), zi(3), rewet, Float64(npf.wetfct), Int(npf.iwetit), Int(npf.ihdwet))
end

# One Jacobi pass of `rewet_check` over all dry wettable cells. MF6 loops over the nodes in
# order, and a cell rewetted earlier in the loop (IBOUND 30000) counts as active for the
# later ones: a neighbour `m` counts when it was active (IBOUND > 0) before the call, with
# its head, or when `m < n` was rewetted in this call, with its new head (read from the
# previous pass). Passes repeat until nothing changes; the result is that of the sequential
# loop, since a cell depends only on lower-numbered ones. The first qualifying neighbour in
# CSR order sets the new head (`ihdwet = 0`: bot + WETFCT·(h_m − bot); else bot +
# WETFCT·|WETDRY|).
@kernel function npf_rewet_kernel!(
        flagout, hwetout, count, @Const(flagin), @Const(hwetin), @Const(ia), @Const(ja), @Const(jas), @Const(ihcx),
        @Const(Δr), @Const(h), @Const(ibound), @Const(wetdry), @Const(botoff), wetfct, ihdwet::Bool,
    )
    n = @index(Global)
    T = eltype(hwetout)
    @inbounds begin
        f = Int32(0)
        hw = zero(T)
        if ibound[n] == Int32(0) && wetdry[n] != zero(T)
            wd = wetdry[n]
            awd = abs(wd)
            bb = botoff[n]
            turnon = bb + awd
            for ii in (ia[n] + Int32(1)):(ia[n + Int32(1)] - Int32(1))
                m = ja[ii]
                ok = false
                hm = zero(T)
                if ibound[m] > Int32(0)
                    ok = true
                    hm = h[m]
                elseif m < n && flagin[m] != Int32(0)
                    ok = true
                    hm = hwetin[m]
                end
                if ok
                    j = jas[ii]
                    hmn = hm + (m > n ? Δr[j] : -Δr[j])
                    vert = (ihcx[j] & Int32(3)) == Int32(0)
                    if (vert || wd > zero(T)) && hmn >= turnon
                        f = Int32(1)
                        hw = ihdwet ? bb + wetfct * awd : bb + wetfct * (hmn - bb)
                        break
                    end
                end
            end
        end
        flagout[n] = f
        hwetout[n] = hw
        if f != flagin[n] || hw != hwetin[n]
            count[1] = Int32(1)
        end
    end
end

@kernel function npf_rewet_apply_kernel!(h, ibound, count, @Const(flag), @Const(hwet))
    n = @index(Global)
    @inbounds if flag[n] != Int32(0)
        h[n] = hwet[n]
        ibound[n] = Int32(1)
        count[2] = Int32(1)
    end
end

# `sgwf_npf_wetdry`, drying: a convertible cell whose head is at or below its bottom
# becomes inactive; its head is kept at the bottom (MF6 sets HDRY; the head of a dry cell is
# never read, and the output writes HDRY for IBOUND 0). A constant-head cell going dry
# flags an error.
@kernel function npf_dry_kernel!(h, ibound, count, @Const(icelltype), @Const(topoff), @Const(botoff))
    n = @index(Global)
    @inbounds if ibound[n] != Int32(0) && icelltype[n] != Int32(0)
        hn = h[n]
        ttop = min(topoff[n], hn)
        if ttop - botoff[n] <= zero(hn)
            if ibound[n] < Int32(0)
                count[3] = Int32(1)
            else
                ibound[n] = Int32(0)
                h[n] = botoff[n]
                count[2] = Int32(1)
            end
        end
    end
end

# `npf_ad`: the old head of a dry wettable cell is its bottom.
@kernel function npf_ad_kernel!(hold, @Const(ibound), @Const(wetdry), @Const(botoff))
    n = @index(Global)
    @inbounds if wetdry[n] != zero(eltype(wetdry)) && ibound[n] == Int32(0)
        hold[n] = botoff[n]
    end
end

"""
    npf_ad!(wd::WetDry, cv::ConvertibleNpf, hold, ibound)

The old heads of dry cells that can rewet are set to their bottom (`npf_ad` with REWET),
at the start of a time step.
"""
function npf_ad!(wd::WetDry, cv::ConvertibleNpf, hold, ibound)
    wd.rewet || return nothing
    npf_ad_kernel!(get_backend(hold), _WG)(hold, ibound, wd.wetdry, cv.botoff; ndrange = length(hold))
    return nothing
end

"""
    wetdry_device!(wd::WetDry, cv::ConvertibleNpf, csr, Δr, h, ibound, kiter) -> (changed, chd_dry)

`sgwf_npf_wetdry` for outer iteration `kiter` (0 at the start of the run): with REWET and
`kiter % IWETIT == 0` the dry wettable cells next to a wet enough cell are rewetted
(`rewet_check`, Jacobi passes until nothing changes, then applied), then convertible cells
with heads at or below their bottom are dried. Returns whether IBOUND changed and whether a
constant-head cell went dry (one synchronisation per pass).
"""
function wetdry_device!(wd::WetDry, cv::ConvertibleNpf, csr::CsrStructure, Δr, h, ibound, kiter::Integer)
    backend = get_backend(h)
    nodes = length(h)
    T = eltype(h)
    fill!(wd.count, Int32(0))
    if wd.rewet && kiter % wd.iwetit == 0
        fill!(wd.flag, Int32(0))
        fill!(wd.hwet, zero(T))
        fin, hin, fout, hout = wd.flag, wd.hwet, wd.flag2, wd.hwet2
        for _ in 1:(nodes + 1)
            fill!(wd.count, Int32(0))
            npf_rewet_kernel!(backend, _WG)(
                fout, hout, wd.count, fin, hin, csr.ia, csr.ja, csr.jas, cv.ihcx, Δr, h, ibound, wd.wetdry, cv.botoff,
                T(wd.wetfct), wd.ihdwet != 0; ndrange = nodes,
            )
            fin, hin, fout, hout = fout, hout, fin, hin
            Array(wd.count)[1] == 0 && break
        end
        fill!(wd.count, Int32(0))
        npf_rewet_apply_kernel!(backend, _WG)(h, ibound, wd.count, fin, hin; ndrange = nodes)
    end
    npf_dry_kernel!(backend, _WG)(h, ibound, wd.count, cv.icelltype, cv.topoff, cv.botoff; ndrange = nodes)
    c = Array(wd.count)
    return c[2] != 0, c[3] != 0
end
