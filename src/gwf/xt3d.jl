# XT3D: the extended (full-tensor) flow formulation of NPF (T6.5).
#
# Ports MF6 6.8.2:
# - Xt3dAlgorithm.f90 `qconds`, `abhats`, `getrot`, `tranvc`, `abwts`: the "conductances"
#   of the normal flux across one interface (`_xt3d_qconds` and helpers);
# - Xt3dInterface.f90 `xt3d_load` (conductivity tensor, connection unit vectors and
#   normals, split connection lengths; `xt3d_fillrmatck`), `xt3d_areas` (non-Newton),
#   `xt3d_indices`, `xt3d_ac` (the extended stencil: neighbours of neighbours),
#   `xt3d_fcpc`/`xt3d_fc` (matrix terms, `xt3d_amat_nbrs`, `xt3d_amat_nbrnbrs`; RHS option:
#   `xt3d_rhs`), `xt3d_flowja` (`xt3d_qnbrs`);
# - Dis.f90, Disv.f90, Disu.f90 `connection_vector` (with DisvGeom.f90 `line_unit_vector`)
#   and `connection_normal` (`_connection_normal` in npf.jl);
# - gwf-npf.f90 `npf_ar` (ANGLE1 = 0 when not given with XT3D, `nozee` without ANGLE2) and
#   the XT3D checks of `xt3d_ar`.
#
# Scope: XT3D for models whose cells are all confined (ICELLTYPE 0). There, MF6 computes the
# coefficients once (`lamatsaved`, every cell `iallpc`, `xt3d_fcpc`) and the matrix is
# constant. So does this port: the coefficients are computed on the host in Float64
# (`xt3d_setup`, like `calc_condsat`), and every outer iteration formulates per cell in one
# kernel (`xt3d_formulate_kernel!`): the boundary and storage terms, the constant XT3D matrix
# row and the flux-form residual. XT3D with convertible cells or with NEWTON (saturation-
# dependent areas, `xt3d_fn`) is an `MF6UnsupportedError`.
#
# Flux of a connection. For the connection between `n < m` MF6 forms (xt3d_flowja)
#
#     Q = chat01·(h_m − h_n) + Σ_i chati0_i·(h_i − h_n) − Σ_j chat1j_j·(h_j − h_m),
#
# the flow into `n` (i over the neighbours of n, j over those of m). Each term is stored as
# `c·(h_a − h_b)` (`ta`, `tb`, `tc`, the primary term first), with the reference difference
# `r_a − r_b` (`tdr`, Float64 rounded to `T`), so that on the device
#
#     Q = Σ_t c_t·(h'_a − h'_b + Δr_t)
#
# in offset heads (PLAN.md section 3.7, rules 1 and 2), accumulated with compensated
# summation. The matrix row of `n` is `∂(Σ_m Q_nm)/∂h`: the primary terms on MF6's stencil,
# the others on the neighbours and the neighbours of neighbours (`xt3d_ac`). With the RHS
# option only the primary terms enter the matrix (MF6 puts the others on the right-hand
# side with the current heads; with correction solves of the flux-form residual this is
# the same as an approximate Jacobian, and the converged heads are the same).

# Xt3dInterface.f90 `allocate_scalars`: attenuation threshold `vcthresh`.
const _XT3D_VCTHRESH = 1.0e-10

"""
    check_xt3d(npf::NpfSpec, dis::AbstractDisSpec, icelltype; newton = false)

What XT3D needs (Xt3dInterface.f90 `xt3d_ar`) and what this port supports: a DISU grid
without vertices or without ANGLDEGX is MF6's error (`MF6InputError`); convertible cells
(ICELLTYPE /= 0) or the NEWTON formulation with XT3D are an `MF6UnsupportedError`.
"""
function check_xt3d(npf::NpfSpec, dis::AbstractDisSpec, icelltype::AbstractVector{<:Integer}; newton::Bool = false)
    npf.xt3d || return nothing
    file = isempty(npf.file) ? "NPF" : npf.file
    if dis isa DisuSpec
        dfile = isempty(dis.file) ? "DISU" : dis.file
        dis.nvert > 0 || throw(
            MF6InputError(
                dfile, 0,
                "Vertices not specified for discretization package, but XT3D is active: $(uppercase(file)). Vertices must be specified in discretization package in order to use XT3D.",
            ),
        )
        isnothing(dis.angldegx) && throw(
            MF6InputError(
                dfile, 0,
                "ANGLDEGX is not specified in the DIS package, but XT3D is active: $(uppercase(file)). ANGLDEGX must be provided in discretization package in order to use XT3D.",
            ),
        )
    end
    any(!=(0), icelltype) && unsupported(file, "NPF6 option XT3D with convertible cells (ICELLTYPE /= 0)")
    newton && unsupported(file, "NPF6 option XT3D with the NEWTON formulation")
    return nothing
end

# Xt3dInterface.f90 `xt3d_fillrmatck`: rotation matrix of the conductivity ellipsoid
# (angles in radians; without ANGLE2, `nozee`, only ANGLE1 is used).
function _xt3d_rmatck(p::NpfProperties, n::Integer, nozee::Bool)
    ang1 = isnothing(p.angle1) ? 0.0 : p.angle1[n]
    ang2 = (nozee || isnothing(p.angle2)) ? 0.0 : p.angle2[n]
    ang3 = (nozee || isnothing(p.angle3)) ? 0.0 : p.angle3[n]
    s1, c1 = sin(ang1), cos(ang1)
    s2, c2 = sin(ang2), cos(ang2)
    s3, c3 = sin(ang3), cos(ang3)
    return [
        c1 * c2 (c1 * s2 * s3 - s1 * c3) (-c1 * s2 * c3 - s1 * s3)
        s1 * c2 (s1 * s2 * s3 + c1 * c3) (-s1 * s2 * c3 + c1 * s3)
        s2 (-c2 * s3) (c2 * c3)
    ]
end

# DisvGeom.f90 `line_unit_vector`.
function _line_unit_vector(x0, y0, z0, x1, y1, z1)
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    vmag = sqrt(dx^2 + dy^2 + dz^2)
    return (dx / vmag, dy / vmag, dz / vmag), vmag
end

# Dis.f90 / Disv.f90 / Disu.f90 `connection_vector` (saturation 1, confined): unit vector
# from `n` to `m` and the straight-line connection length.
function _xt3d_connection_vector(conn::Connectivity, n::Integer, m::Integer, nozee::Bool, ihc::Integer, jj::Integer)
    zc(k) = conn.bot[k] + 0.5 * (conn.top[k] - conn.bot[k])
    if conn.grid === :disu
        zn, zm = if ihc == 0
            zc(n), zc(m)
        else
            nozee ? (0.0, 0.0) : (zc(n), zc(m))
        end
        return _line_unit_vector(conn.xc[n], conn.yc[n], zn, conn.xc[m], conn.yc[m], zm)
    end
    if ihc == 0
        return (0.0, 0.0, m < n ? 1.0 : -1.0), abs(zc(m) - zc(n))
    end
    zn, zm = nozee ? (0.0, 0.0) : (zc(n), zc(m))
    if conn.grid === :dis
        # Dis.f90: x1 = y1 = 0, (x2, y2) = ±ds along the exact face normal.
        ds = conn.cl1[jj] + conn.cl2[jj]
        vx, vy, _ = _connection_normal(conn, n, m, ihc, jj)
        return _line_unit_vector(0.0, 0.0, zn, vx * ds, vy * ds, zm)
    end
    return _line_unit_vector(conn.xc[n], conn.yc[n], zn, conn.xc[m], conn.yc[m], zm)
end

# Xt3dInterface.f90 `xt3d_load` for cell `n`: neighbours (0 when inactive), connection unit
# vectors `vc` and face normals `vn` (nnbr × 3), split lengths `dl`, `dln`, the rotated
# conductivity tensor `ck` and whether every active connection is horizontal.
function _xt3d_load(conn::Connectivity, p::NpfProperties, ibound, nozee::Bool, n::Integer)
    i0 = conn.ia[n]
    nnbr = Int(conn.ia[n + 1] - i0 - 1)
    rm = _xt3d_rmatck(p, n, nozee)
    ck = rm * [p.k11[n] 0.0 0.0; 0.0 p.k22[n] 0.0; 0.0 0.0 p.k33[n]] * transpose(rm)
    inbr = zeros(Int32, nnbr)
    vc = zeros(nnbr, 3)
    vn = zeros(nnbr, 3)
    dl = zeros(nnbr)
    dln = zeros(nnbr)
    allhc = true
    for il in 1:nnbr
        ii = i0 + il
        jj = conn.ja[ii]
        jjs = conn.jas[ii]
        ibound[jj] != 0 || continue
        inbr[il] = jj
        ihc = conn.ihc[jjs]
        vn[il, :] .= _connection_normal(conn, n, jj, ihc, jjs)
        v, dltot = _xt3d_connection_vector(conn, n, jj, nozee, ihc, jjs)
        vc[il, :] .= v
        cl1, cl2 = jj > n ? (conn.cl1[jjs], conn.cl2[jjs]) : (conn.cl2[jjs], conn.cl1[jjs])
        ooclsum = 1.0 / (cl1 + cl2)
        dl[il] = dltot * cl1 * ooclsum
        dln[il] = dltot * cl2 * ooclsum
        ihc == 0 && (allhc = false)
    end
    return (; nnbr, inbr, vc, vn, dl, dln, ck, allhc)
end

# Xt3dAlgorithm.f90 `getrot`: rotation to the (c, d, e) coordinates of connection `il01`;
# `iml0 = 0` when no other active connection has a perpendicular component.
function _xt3d_getrot(c, il01)
    vcc = c.vc[il01, :]
    acmpmn = 1.0 - 1.0e-10
    cmpmn = 0.0
    iml0 = 0
    for il in 1:c.nnbr
        (il == il01 || c.inbr[il] == 0) && continue
        cmp = c.vc[il, 1] * vcc[1] + c.vc[il, 2] * vcc[2] + c.vc[il, 3] * vcc[3]
        acmp = abs(cmp)
        if acmp < acmpmn
            cmpmn = cmp
            acmpmn = acmp
            iml0 = il
        end
    end
    rmat = zeros(3, 3)
    rmat[:, 1] .= vcc
    iml0 == 0 && return rmat, 0
    vcd = (c.vc[iml0, :] .- cmpmn .* vcc) ./ sqrt(1.0 - cmpmn * cmpmn)
    vce = [vcc[2] * vcd[3] - vcc[3] * vcd[2], vcc[3] * vcd[1] - vcc[1] * vcd[3], vcc[1] * vcd[2] - vcc[2] * vcd[1]]
    rmat[:, 2] .= vcd
    rmat[:, 3] .= vce
    return rmat, iml0
end

# Xt3dAlgorithm.f90 `abwts`: "a" and "b" weights for the perpendicular direction `nde1`.
function _xt3d_abwts(c, il01, nde1, vccde, vcthresh)
    nnbr = c.nnbr
    nde2 = 5 - nde1
    omwt = zeros(nnbr)
    dsum = 0.0
    vcmx = 0.0
    skip(il) = il == il01 || c.inbr[il] == 0
    for il in 1:nnbr
        skip(il) && continue
        vcmx = max(abs(vccde[il, nde1]), vcmx)
        dlm = 0.5 * (c.dl[il] + c.dln[il])
        cosang = vccde[il, 1]
        dl4wt = sqrt(dlm * dlm + c.dl[il01] * c.dl[il01] - 2.0 * dlm * c.dl[il01] * cosang)
        omwt[il] = abs(vccde[il, nde1]) * dl4wt
        dsum += omwt[il]
    end
    dsum = dsum + 1.0e-10 * dsum
    for il in 1:nnbr
        skip(il) && continue
        fact = dsum - omwt[il]
        omwt[il] = fact * abs(vccde[il, nde1])
    end
    bd = zeros(nnbr)
    dsum = 0.0
    for il in 1:nnbr
        skip(il) && continue
        bd[il] = omwt[il] * copysign(1.0, vccde[il, nde1])
        dsum += omwt[il] * abs(vccde[il, nde1])
    end
    oodsum = 1.0 / dsum
    for il in 1:nnbr
        skip(il) && continue
        bd[il] *= oodsum
    end
    add = 1.0
    acd = 0.0
    aed = 0.0
    for il in 1:nnbr
        skip(il) && continue
        acd += bd[il] * vccde[il, 1]
        aed += bd[il] * vccde[il, nde2]
    end
    if vcmx < vcthresh
        fatten = vcmx / vcthresh
        acd *= fatten
        aed *= fatten
        bd .*= fatten
    end
    return acd, add, aed, bd
end

# Xt3dAlgorithm.f90 `abhats`: "ahat" and "bhat" of one side of the interface (cell `c`,
# connection `il01`), times the area `ar01`.
function _xt3d_abhats(c, il01, ar01, vcthresh)
    nnbr = c.nnbr
    rmat, iml0 = _xt3d_getrot(c, il01)
    vn = c.vn[il01, :]
    bhat = zeros(nnbr)
    if iml0 == 0
        sigma1 = sum(vn .* (c.ck * rmat[:, 1]))
        ahat = sigma1 / c.dl[il01]
    else
        # tranvc: unit vectors in (c, d, e) coordinates.
        vccde = zeros(nnbr, 3)
        for il in 1:nnbr
            vccde[il, :] .= transpose(rmat) * c.vc[il, :]
        end
        acd, add, aed, bd = _xt3d_abwts(c, il01, 2, vccde, vcthresh)
        skip(il) = il == il01 || c.inbr[il] == 0
        ace, aee, ade, be = 0.0, 1.0, 0.0, zeros(nnbr)
        if !c.allhc && any(il -> !skip(il) && abs(vccde[il, 3]) > 1.0e-10, 1:nnbr)
            ace, aee, ade, be = _xt3d_abwts(c, il01, 3, vccde, vcthresh)
        end
        determ = add * aee - ade * aed
        oodet = 1.0 / determ
        alphad = (acd * aee - ace * aed) * oodet
        alphae = (ace * add - acd * ade) * oodet
        betad = zeros(nnbr)
        betae = zeros(nnbr)
        for il in 1:nnbr
            skip(il) && continue
            betad[il] = (bd[il] * aee - be[il] * aed) * oodet
            betae[il] = (be[il] * add - bd[il] * ade) * oodet
        end
        sigma = transpose(transpose(vn) * (c.ck * rmat))
        ahat = (sigma[1] - sigma[2] * alphad - sigma[3] * alphae) / c.dl[il01]
        for il in 1:nnbr
            skip(il) && continue
            dl0il = c.dl[il] + c.dln[il]
            bhat[il] = (sigma[2] * betad[il] + sigma[3] * betae[il]) / dl0il
        end
        bhat[il01] = 0.0
    end
    return ahat * ar01, bhat .* ar01
end

# Xt3dAlgorithm.f90 `qconds`: chat01, chati0 (per neighbour of cell 0), chat1j (per
# neighbour of cell 1).
function _xt3d_qconds(c0, il01, c1, il10, ar01, ar10, vcthresh)
    if ar01 == 0.0
        return 0.0, zeros(c0.nnbr), zeros(c1.nnbr)
    end
    ahat0, bhat0 = _xt3d_abhats(c0, il01, ar01, vcthresh)
    ahat1, bhat1 = _xt3d_abhats(c1, il10, ar10, vcthresh)
    denom = ahat0 + ahat1
    wght1 = abs(denom) > eps(Float64) ? ahat0 / (ahat0 + ahat1) : 1.0  # DPREC
    wght0 = 1.0 - wght1
    return wght1 * ahat1, wght0 .* bhat0, wght1 .* bhat1
end

# Xt3dInterface.f90 `xt3d_areas` (non-Newton; saturation 1): interface areas seen from n
# and from m.
function _xt3d_areas(conn::Connectivity, n::Integer, m::Integer, jj::Integer)
    ihc = conn.ihc[jj]
    ihc == 0 && return conn.hwva[jj], conn.hwva[jj]
    topn, botn, topm, botm = conn.top[n], conn.bot[n], conn.top[m], conn.bot[m]
    thksatn, thksatm = topn - botn, topm - botm
    if ihc == 2
        sill_top, sill_bot = min(topn, topm), max(botn, botm)
        thksatn = max(min(botn + thksatn, sill_top) - sill_bot, 0.0)
        thksatm = max(min(botm + thksatm, sill_top) - sill_bot, 0.0)
    end
    return conn.hwva[jj] * thksatn, conn.hwva[jj] * thksatm
end

"""
    Xt3dSetup

Host (Float64) result of the XT3D setup of a model (`xt3d_setup`):

- `rhs`: the RHS option (the matrix holds the primary terms only);
- the flux terms of each symmetric connection `j` (`njas`): `tptr[j]:tptr[j+1]-1` index
  `ta`, `tb` (nodes), `tc` (coefficient) and `tdr` (`r_a − r_b`), the flow into the
  lower-numbered node being `Σ tc·(h_a − h_b)`; the primary term (`chat01`) first;
- the CSR structure of the system (`ia`, `ja`, `isym`, `jas`; diagonal first, then the
  columns in increasing order): MF6's connectivity extended with the neighbours of
  neighbours (`xt3d_ac`), or the connectivity itself with the RHS option;
- `amat`: the XT3D part of the matrix on that structure (diagonal included);
- `sb` `(nodes)`: `−Σ_{k≠n} a_nk·(r_k − r_n)`, the XT3D part of the offset right-hand side
  `b − A·r`.
"""
struct Xt3dSetup
    rhs::Bool
    tptr::Vector{Int32}
    ta::Vector{Int32}
    tb::Vector{Int32}
    tc::Vector{Float64}
    tdr::Vector{Float64}
    ia::Vector{Int32}
    ja::Vector{Int32}
    isym::Vector{Int32}
    jas::Vector{Int32}
    amat::Vector{Float64}
    sb::Vector{Float64}
end

# Xt3dInterface.f90 `xt3d_ac`: row n holds n, its neighbours and their neighbours.
function _xt3d_structure(conn::Connectivity)
    nodes = Int(conn.nodes)
    ia = Vector{Int32}(undef, nodes + 1)
    ja = Int32[]
    cols = Int32[]
    ia[1] = 1
    for n in 1:nodes
        empty!(cols)
        for ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
            j = conn.ja[ii]
            push!(cols, j)
            for kk in (conn.ia[j] + 1):(conn.ia[j + 1] - 1)
                k = conn.ja[kk]
                k != n && push!(cols, k)
            end
        end
        sort!(unique!(cols))
        push!(ja, Int32(n))
        append!(ja, cols)
        ia[n + 1] = length(ja) + 1
    end
    isym = Vector{Int32}(undef, length(ja))
    fill_isym!(isym, ia, ja)
    jas = Vector{Int32}(undef, length(ja))
    fill_jas!(jas, ia, ja, isym)
    return ia, ja, isym, jas
end

# Position of column k in row n (diagonal first; the rest sorted when `sorted`, as in the
# extended structure, otherwise searched linearly, as in a DISU connectivity).
function _csr_position(ia, ja, n::Integer, k::Integer, sorted::Bool)
    k == n && return Int(ia[n])
    lo, hi = Int(ia[n]) + 1, Int(ia[n + 1]) - 1
    p = sorted ? searchsortedfirst(view(ja, lo:hi), k) + lo - 1 : something(findfirst(==(k), view(ja, lo:hi)), hi - lo + 2) + lo - 1
    (p <= hi && ja[p] == k) || error("xt3d: column $k not in row $n of the XT3D stencil")
    return p
end

"""
    xt3d_setup(conn::Connectivity, p::NpfProperties, r; rhs = false) -> Xt3dSetup

The XT3D coefficients of a confined model on the host in Float64 (`xt3d_fcpc`: every cell
permanently confined, full-thickness areas; all reduced cells active), the extended system
structure and its constant XT3D matrix, with the reference heads `r` for the offset terms.
`rhs`: the XT3D RHS option.
"""
function xt3d_setup(conn::Connectivity, p::NpfProperties, r::AbstractVector{Float64}; rhs::Bool = false)
    nodes = Int(conn.nodes)
    ibound = ones(Int32, nodes)
    # gwf-npf.f90 `npf_ar` / `xt3d_ar`: no z component for horizontal connections
    # without ANGLE2.
    nozee = isnothing(p.angle2)
    loads = [_xt3d_load(conn, p, ibound, nozee, n) for n in 1:nodes]
    tptr = Vector{Int32}(undef, conn.njas + 1)
    ta, tb, tc = Int32[], Int32[], Float64[]
    tptr[1] = 1
    for n in 1:nodes, il0 in 1:loads[n].nnbr
        ii = conn.ia[n] + il0
        m = conn.ja[ii]
        m > n || continue
        jj = conn.jas[ii]
        @assert tptr[jj] == length(tc) + 1  # upper triangle in CSR order
        c0, c1 = loads[n], loads[m]
        # xt3d_indices: local number of n among m's neighbours.
        il10 = findfirst(==(n), c1.inbr)
        ar01, ar10 = _xt3d_areas(conn, n, m, jj)
        chat01, chati0, chat1j = _xt3d_qconds(c0, il0, c1, il10, ar01, ar10, _XT3D_VCTHRESH)
        push!(ta, m), push!(tb, n), push!(tc, chat01)
        for il in 1:c0.nnbr
            (c0.inbr[il] == 0 || il == il0 || chati0[il] == 0.0) && continue
            push!(ta, c0.inbr[il]), push!(tb, n), push!(tc, chati0[il])
        end
        for il in 1:c1.nnbr
            (c1.inbr[il] == 0 || il == il10 || chat1j[il] == 0.0) && continue
            push!(ta, c1.inbr[il]), push!(tb, m), push!(tc, -chat1j[il])
        end
        tptr[jj + 1] = length(tc) + 1
    end
    tdr = [r[ta[t]] - r[tb[t]] for t in eachindex(ta)]
    ia, ja, isym, jas = rhs ? (conn.ia, conn.ja, conn.isym, conn.jas) : _xt3d_structure(conn)
    amat = zeros(length(ja))
    for n in 1:nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
        m = conn.ja[ii]
        m > n || continue
        jj = conn.jas[ii]
        t2 = rhs ? tptr[jj] : tptr[jj + 1] - 1
        for t in tptr[jj]:t2
            a, b, c = ta[t], tb[t], tc[t]
            # Row n: the flow into n; row m: its negative.
            amat[_csr_position(ia, ja, n, a, !rhs)] += c
            amat[_csr_position(ia, ja, n, b, !rhs)] -= c
            amat[_csr_position(ia, ja, m, a, !rhs)] -= c
            amat[_csr_position(ia, ja, m, b, !rhs)] += c
        end
    end
    sb = zeros(nodes)
    for n in 1:nodes, ii in (ia[n] + 1):(ia[n + 1] - 1)
        k = ja[ii]
        sb[n] -= amat[ii] * (r[k] - r[n])
    end
    return Xt3dSetup(rhs, tptr, ta, tb, tc, tdr, ia, ja, isym, jas, amat, sb)
end

"""
    Xt3dNpf{T}

The XT3D data of a model on a device (`Xt3dNpf(backend, T, setup::Xt3dSetup)`): `rhs`
(the RHS option), the flux terms `tptr`, `ta`, `tb` (Int32), `tc`, `tdr` (in `T`), the XT3D
matrix `xval` on the system structure and `xsb` (see `Xt3dSetup`). Adapt moves it.
"""
struct Xt3dNpf{T <: AbstractFloat, V <: AbstractVector{T}, VI <: AbstractVector{Int32}}
    rhs::Bool
    tptr::VI
    ta::VI
    tb::VI
    tc::V
    tdr::V
    xval::V
    xsb::V
end

Adapt.@adapt_structure Xt3dNpf

function Xt3dNpf(backend::Backend, ::Type{T}, s::Xt3dSetup) where {T <: AbstractFloat}
    dev(x) = _device_copy(backend, x)
    return Xt3dNpf(s.rhs, dev(s.tptr), dev(s.ta), dev(s.tb), dev(T.(s.tc)), dev(T.(s.tdr)), dev(T.(s.amat)), dev(T.(s.sb)))
end

"""
    CsrStructure(backend, s::Xt3dSetup) -> CsrStructure

The system structure of an XT3D model (the extended stencil, or the connectivity with the
RHS option) on `backend`.
"""
function CsrStructure(backend::Backend, s::Xt3dSetup)
    dev(x) = _device_copy(backend, x)
    nodes = length(s.ia) - 1
    return CsrStructure(
        nodes, length(s.ja), dev(s.ia), dev(s.ja), dev(s.jas), dev(s.isym), dev(s.ia[1:nodes]), dev(collect(Int32(1):Int32(nodes))),
    )
end

# Flow into the lower-numbered node of symmetric connection `j` (offset heads `h`), its
# primary term (`chat01`) and the sum of the terms' magnitudes (the scale of the rounding
# of the flow, for the Float32 rclose floor: the terms of a face cancel), compensated sum in
# `T`. Device-safe.
@inline function _xt3d_flux(tptr, ta, tb, tc, tdr, h, j)
    T = eltype(tc)
    s = zero(T)
    c = zero(T)
    qp = zero(T)
    a = zero(T)
    @inbounds begin
        t1 = tptr[j]
        for t in t1:(tptr[j + 1] - Int32(1))
            x = tc[t] * (h[ta[t]] - h[tb[t]] + tdr[t])
            t == t1 && (qp = x)
            s, c = _sum_add(s, c, x)
            a += abs(x)
        end
    end
    return s + c, qp, a
end

@kernel function xt3d_formulate_kernel!(
        amat, rhs, diag, bdiag, brhs, res, scale, @Const(ia), @Const(ja), @Const(jas), @Const(sia), @Const(tptr), @Const(ta),
        @Const(tb), @Const(tc), @Const(tdr), @Const(xval), @Const(xsb), @Const(h), @Const(ibound), pkgs, sto, offdiag::Bool,
        terms::Bool, tiny, rhsmode::Bool,
    )
    n = @index(Global)
    T = eltype(amat)
    @inbounds begin
        bd, br, _ = _add_package_terms(n, zero(T), zero(T), zero(T), pkgs, h, ibound)
        bd, br, _ = _add_storage_terms(n, bd, br, zero(T), sto, ibound, h, false)
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
            s = br + xsb[n]
            for ii in (ia[n] + Int32(1)):(ia[n + 1] - Int32(1))
                m = ja[ii]
                ibound[m] != 0 || continue
                q, qp, qa = _xt3d_flux(tptr, ta, tb, tc, tdr, h, jas[ii])
                if m < n
                    q, qp = -q, -qp
                end
                rs, rc = _sum_add(rs, rc, -q)
                sc += qa
                # xt3d_rhs: the non-primary terms on the right-hand side.
                rhsmode && (s -= q - qp)
            end
            d = xval[sid] + bd
            if abs(d) < tiny
                d = -one(T)
                s -= hn
            end
            amat[sid] = d
            diag[n] = d
            if offdiag
                for ii in (sid + Int32(1)):si2
                    amat[ii] = xval[ii]
                end
            end
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

"""
    xt3d_formulate_system!(sys, res, csr, scsr, x::Xt3dNpf, h, ibound, packages, storage;
                           offdiag = true, terms = true, scale = nothing) -> sys

`formulate_system!` of an XT3D model (T6.5): per row the boundary and storage terms, the
constant XT3D matrix row on the system structure `scsr` (off-diagonals written only with
`offdiag`), the small-diagonal and constant-head adjustments of `sln_ls`, and the flux-form
residual `res` from the XT3D connection flows over the connectivity `csr`. `rhs` (with
`terms`) is `b − A·r` in offsets; with the RHS option it includes the non-primary flux
terms at the heads `h` (`xt3d_rhs`).
"""
function xt3d_formulate_system!(
        sys::LinearSystem, res, csr::CsrStructure, scsr::CsrStructure, x::Xt3dNpf, h, ibound, packages::Tuple, storage;
        offdiag::Bool = true, terms::Bool = true, scale = nothing,
    )
    backend = get_backend(sys.amat)
    xt3d_formulate_kernel!(backend, _WG)(
        sys.amat, sys.rhs, sys.diag, sys.bdiag, sys.brhs, res, scale, csr.ia, csr.ja, csr.jas, scsr.ia, x.tptr, x.ta, x.tb,
        x.tc, x.tdr, x.xval, x.xsb, h, ibound, packages, storage, offdiag, terms, eltype(sys)(1.0e-15), x.rhs; ndrange = csr.nodes,
    )
    return sys
end

@kernel function xt3d_flowja_kernel!(flowja, @Const(ia), @Const(ja), @Const(jas), @Const(irow), @Const(tptr), @Const(ta), @Const(tb), @Const(tc), @Const(tdr), @Const(h), @Const(ibound))
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
                q, _, _ = _xt3d_flux(tptr, ta, tb, tc, tdr, h, jas[ii])
                flowja[ii] = m > n ? q : -q
            end
        end
    end
end

"""
    xt3d_flowja!(flowja, csr::CsrStructure, x::Xt3dNpf, h, ibound) -> flowja

Connection flows of an XT3D model (`xt3d_flowja`): entry `ii` of row `n` gets the XT3D flow
into `n` from `ja[ii]` (zero when either cell is inactive), the diagonal entries are
zeroed. Exactly antisymmetric (both entries evaluate the same sum).
"""
function xt3d_flowja!(flowja, csr::CsrStructure, x::Xt3dNpf, h, ibound)
    xt3d_flowja_kernel!(get_backend(flowja), _WG)(
        flowja, csr.ia, csr.ja, csr.jas, csr.irow, x.tptr, x.ta, x.tb, x.tc, x.tdr, h, ibound; ndrange = csr.nja,
    )
    return flowja
end
