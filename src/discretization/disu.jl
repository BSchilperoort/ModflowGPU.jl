# Unstructured grid (DISU) → Connectivity (host only, T6.4).
#
# Ports MF6 6.8.2:
# - Disu.f90 `disu_ck` (first connection of each cell is itself, valid node numbers,
#   IDOMAIN in 0..1, positive thickness and area of active cells, VERTICAL_OFFSET_TOLERANCE
#   ≥ 0, a cell below another (IHC 0, m > n) has its top at or below n's bottom within the
#   tolerance, and the ANGLDEGX consistency warnings of horizontal connections between
#   active cells, whose count `nangldegxerr` NPF turns into an error when it needs
#   ANGLDEGX), `disu_face_normal`, `grid_finalize` (active node count, `nodereduced` /
#   `nodeuser`, reduced `top`, `bot`, `area`, `xc`, `yc`; `xc`/`yc` are empty without
#   vertices);
# - Connections.f90 `disuconnections` (the user connectivity as given, or with excluded
#   cells removed and rows sorted, the per-connection input arrays compressed in input
#   order) and `con_finalize` (sorted rows with the diagonal first, symmetric `ja`, `ihc`
#   and `hwva`, IHC in 0..2; `cl12` into `cl1` (n < m) and `cl2` (n > m); ANGLDEGX of the
#   upper triangle in radians, or `DNODATA` without ANGLDEGX).
#
# IDOMAIN < 0 is invalid for DISU (disu_ck), so there are no pass-through cells.

# ConstantsModule DNODATA.
const DNODATA = 3.0e30

# Disu.f90 `iac_to_ia` / Connections.f90: IAC (entries per row) → IA.
function _iac_to_ia(iac::AbstractVector{<:Integer})
    ia = Vector{Int32}(undef, length(iac) + 1)
    ia[1] = 1
    for n in eachindex(iac)
        ia[n + 1] = ia[n] + iac[n]
    end
    return ia
end

# Fortran `1PG15.6` without leading blanks (the number format of the DISU messages).
_g15(x) = strip(_fortran_1pg(Float64(x); w = 15, d = 6))

# Fortran `f0.3`.
_f03(x) = @sprintf("%.3f", x)

# Disu.f90 `disu_face_normal`: unit normal of the face shared by user cells n and m (exactly
# two shared vertices, compared within 1e-6 of the centre distance), pointing from n
# towards m; `nothing` otherwise.
function _disu_face_normal(dis::DisuSpec, iavert, javert, n::Integer, m::Integer)
    v = dis.vertices
    dx = dis.cell_center[1, m] - dis.cell_center[1, n]
    dy = dis.cell_center[2, m] - dis.cell_center[2, n]
    tol = 1.0e-6 * sqrt(dx * dx + dy * dy)
    nshared = 0
    ivshared = (Int32(0), Int32(0))
    for i in iavert[n]:(iavert[n + 1] - 1)
        iv = javert[i]
        any(==(iv), view(javert, iavert[n]:(i - 1))) && continue
        for j in iavert[m]:(iavert[m + 1] - 1)
            jv = javert[j]
            if abs(v[1, jv] - v[1, iv]) <= tol && abs(v[2, jv] - v[2, iv]) <= tol
                nshared += 1
                nshared > 2 && return nothing
                ivshared = nshared == 1 ? (iv, ivshared[2]) : (ivshared[1], iv)
                break
            end
        end
    end
    nshared == 2 || return nothing
    ex = v[1, ivshared[2]] - v[1, ivshared[1]]
    ey = v[2, ivshared[2]] - v[2, ivshared[1]]
    elen = sqrt(ex * ex + ey * ey)
    elen <= 0.0 && return nothing
    xnorm = ey / elen
    ynorm = -ex / elen
    if xnorm * dx + ynorm * dy < 0.0
        xnorm = -xnorm
        ynorm = -ynorm
    end
    return xnorm, ynorm
end

"""
    check_disu_grid(dis::DisuSpec) -> Int

MF6's setup-time checks of a DISU grid (Disu.f90 `disu_ck`, `grid_finalize`), as one
`MF6InputError` (for `dis.file`) with MF6's messages; array lengths that do not match the
dimensions are an `ArgumentError`. The ANGLDEGX consistency checks of horizontal
connections between active cells (reverse connections 180° apart; with vertices, the
normal pointing towards the connected cell and agreeing with the shared face within 1°)
are warnings, as in MF6; returns their count (MF6's `nangldegxerr`), which NPF turns into
an error when it needs ANGLDEGX (`npf_properties`).
"""
function check_disu_grid(dis::DisuSpec)
    file = isempty(dis.file) ? "DISU" : dis.file
    nodes, nja, nvert = Int(dis.nodes), Int(dis.nja), Int(dis.nvert)
    errors = String[]
    nodes < 1 && push!(errors, "NODES was not specified or was specified incorrectly.")
    nja < 1 && push!(errors, "NJA was not specified or was specified incorrectly.")
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    for (name, a) in (("top", dis.top), ("bot", dis.bot), ("area", dis.area), ("idomain", dis.idomain), ("iac", dis.iac))
        length(a) == nodes || throw(ArgumentError("DisuSpec: length($name) = $(length(a)), expected $nodes"))
    end
    for (name, a) in (("ja", dis.ja), ("ihc", dis.ihc), ("cl12", dis.cl12), ("hwva", dis.hwva), ("angldegx", dis.angldegx))
        isnothing(a) || length(a) == nja || throw(ArgumentError("DisuSpec: length($name) = $(length(a)), expected $nja"))
    end
    sum(Int, dis.iac) == nja || throw(ArgumentError("DisuSpec: sum(iac) = $(sum(Int, dis.iac)), expected NJA = $nja"))
    if nvert > 0
        size(dis.vertices) == (2, nvert) || throw(ArgumentError("DisuSpec: size(vertices) = $(size(dis.vertices)), expected (2, $nvert)"))
        size(dis.cell_center) == (2, nodes) || throw(ArgumentError("DisuSpec: size(cell_center) = $(size(dis.cell_center)), expected (2, $nodes)"))
        length(dis.cell_vertices) == nodes || throw(ArgumentError("DisuSpec: length(cell_vertices) = $(length(dis.cell_vertices)), expected $nodes"))
    end
    ia = _iac_to_ia(dis.iac)
    ja = abs.(dis.ja)
    for n in 1:nodes
        m = ja[ia[n]]
        n == m || push!(errors, "Cell $n must have its first connection be itself.  Found: $m")
        for ipos in (ia[n] + 1):(ia[n + 1] - 1)
            m = dis.ja[ipos]
            (m < 0 || m > nodes) && push!(errors, "Cell $n has invalid connection in JA.  Found: $m")
        end
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    for n in 1:nodes
        id = dis.idomain[n]
        (id > 1 || id < 0) && push!(errors, "Invalid idomain value $id specified for node $n")
    end
    for n in 1:nodes
        dis.idomain[n] == 1 || continue
        dz = dis.top[n] - dis.bot[n]
        dz <= 0.0 && push!(errors, "Cell $n with thickness <= 0. Top, bot: " * _fortran_1pg(dis.top[n]) * _fortran_1pg(dis.bot[n]))
        dis.area[n] <= 0.0 && push!(errors, "Cell $n with area <= 0. Area: " * _fortran_1pg(dis.area[n]))
    end
    voff = dis.vertical_offset_tolerance
    if voff < 0.0
        push!(errors, "Vertical offset tolerance must be greater than zero. Found " * _fortran_1pg(voff; w = 15, d = 6))
        throw(MF6InputError(file, 0, join(errors, "\n")))
    end
    for n in 1:nodes, ipos in (ia[n] + 1):(ia[n + 1] - 1)
        m = ja[ipos]
        if dis.ihc[ipos] == 0 && m > n && dis.top[m] - dis.bot[n] > voff
            push!(
                errors, "Top elevation (" * _fortran_1pg(dis.top[m]; w = 15, d = 6) * ") for cell $m is above bottom elevation (" *
                    _fortran_1pg(dis.bot[n]; w = 15, d = 6) * ") for cell $n. Based on node numbering rules cell $m must be below cell $n.",
            )
        end
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    return disu_angldegx_errors(dis)
end

"""
    disu_angldegx_errors(dis::DisuSpec; warn = true) -> Int

The ANGLDEGX part of Disu.f90 `disu_ck`: the number of inconsistent ANGLDEGX values of
horizontal connections between active cells (MF6's `nangldegxerr`; 0 without ANGLDEGX),
with MF6's warnings when `warn`.
"""
function disu_angldegx_errors(dis::DisuSpec; warn::Bool = true)
    isnothing(dis.angldegx) && return 0
    file = isempty(dis.file) ? "DISU" : dis.file
    nodes = Int(dis.nodes)
    ia = _iac_to_ia(dis.iac)
    ja = abs.(dis.ja)
    ang = dis.angldegx
    dpio180 = atan(1.0) / 45.0
    cos45 = 0.70710678118654752
    hasvert = dis.nvert > 0
    iavert, javert = hasvert ? _cell_vertex_csr(dis.cell_vertices) : (Int32[], Int32[])
    nsym = ndir = nvrt = nwarn = 0
    sym = dir = vrt = wrn = ()
    for n in 1:nodes
        dis.idomain[n] == 0 && continue
        for ipos in (ia[n] + 1):(ia[n + 1] - 1)
            m = ja[ipos]
            (m < 1 || m > nodes) && continue
            dis.ihc[ipos] == 0 && continue
            dis.idomain[m] == 0 && continue
            angn = ang[ipos]
            if m > n
                jpos = 0
                for kpos in (ia[m] + 1):(ia[m + 1] - 1)
                    if ja[kpos] == n
                        jpos = kpos
                        break
                    end
                end
                if jpos > 0
                    angm = ang[jpos]
                    dang = mod(angm - angn, 360.0)
                    if abs(dang - 180.0) > 0.01
                        nsym += 1
                        nsym == 1 && (sym = (angn, n, m, angm))
                    end
                end
            end
            if hasvert
                dx = dis.cell_center[1, m] - dis.cell_center[1, n]
                dy = dis.cell_center[2, m] - dis.cell_center[2, n]
                dist = sqrt(dx * dx + dy * dy)
                if dist > 0.0
                    cosang = (cos(angn * dpio180) * dx + sin(angn * dpio180) * dy) / dist
                    angc = mod(atan(dy, dx) / dpio180, 360.0)
                    if cosang <= 0.0
                        ndir += 1
                        ndir == 1 && (dir = (angn, n, m, angc))
                    elseif cosang < cos45
                        nwarn += 1
                        nwarn == 1 && (wrn = (angn, n, m, angc))
                    end
                end
                if m > n
                    fn = _disu_face_normal(dis, iavert, javert, n, m)
                    if !isnothing(fn)
                        angv = mod(atan(fn[2], fn[1]) / dpio180, 360.0)
                        dang = abs(mod(angn - angv + 180.0, 360.0) - 180.0)
                        if dang > 1.0
                            nvrt += 1
                            nvrt == 1 && (vrt = (angn, n, m, angv))
                        end
                    end
                end
            end
        end
    end
    warn || return nsym + ndir + nvrt
    if nsym > 0
        a, n, m, b = sym
        @warn "ANGLDEGX values for $nsym cell faces associated with horizontal connections in the DISU Package are inconsistent with the values for the reverse connections (for example, ANGLDEGX = $(_f03(a)) for the connection from cell $n to cell $m and ANGLDEGX = $(_f03(b)) for the connection from cell $m to cell $n). The two values must differ by 180 degrees (within 0.01 degrees) because they are the outward normals of the two sides of the same cell face. Only the value for the connection from the lower to the higher cell number is used." file
    end
    if ndir > 0
        a, n, m, c = dir
        @warn "ANGLDEGX values for $ndir cell faces associated with horizontal connections in the DISU Package point away from the connected cell (for example, ANGLDEGX = $(_f03(a)) for the connection from cell $n to cell $m, but the direction from the center of cell $n to the center of cell $m is $(_f03(c)) degrees). The centers of two connected cells must lie on opposite sides of their shared face, so ANGLDEGX, the outward normal of the face, must be within 90 degrees of the direction from the cell center to the center of the connected cell, in the coordinate system of the VERTICES." file
    end
    if nvrt > 0
        a, n, m, c = vrt
        @warn "ANGLDEGX values for $nvrt cell faces associated with horizontal connections in the DISU Package differ by more than 1 degree from the outward normal of the face computed from the two VERTICES shared by the connected cells (for example, ANGLDEGX = $(_f03(a)) for the connection from cell $n to cell $m, but the normal computed from the vertices is $(_f03(c)) degrees). ANGLDEGX must be expressed in the coordinate system of the VERTICES." file
    end
    if nwarn > 0
        a, n, m, c = wrn
        @warn "ANGLDEGX values for $nwarn cell faces associated with horizontal connections in the DISU Package deviate by more than 45 degrees from the direction between the centers of the connected cells (for example, ANGLDEGX = $(_f03(a)) for the connection from cell $n to cell $m, but the direction between the cell centers is $(_f03(c)) degrees). This is not necessarily an error, but such connections depart substantially from the assumptions of the control-volume finite-difference method and may warrant a closer look at the grid." file
    end
    return nsym + ndir + nvrt
end

"""
    build_connectivity(dis::DisuSpec) -> Connectivity

The reduced nodes, cell geometry and connectivity of an unstructured grid, as MF6's
Disu.f90 `grid_finalize` and Connections.f90 `disuconnections` / `con_finalize` build them,
after the grid checks of `check_disu_grid`. Errors of `con_finalize` (unsorted rows,
asymmetric JA, IHC or HWVA, IHC outside 0..2) are an `MF6InputError` with MF6's messages.
"""
function build_connectivity(dis::DisuSpec)
    check_disu_grid(dis)
    file = isempty(dis.file) ? "DISU" : dis.file
    nodesuser = Int(dis.nodes)
    iainp = _iac_to_ia(dis.iac)
    jainp = abs.(dis.ja)
    hasangle = !isnothing(dis.angldegx)
    angin = hasangle ? dis.angldegx : zeros(length(jainp))
    nodereduced, nodeuser = reduced_numbering(dis.idomain)
    nodes = length(nodeuser)

    if nodes == nodesuser
        ia = iainp
        ja = Vector{Int32}(jainp)
        ihct, cl12t, hwvat, angt = dis.ihc, dis.cl12, dis.hwva, angin
    else
        # Connections.f90 `disuconnections`, reduced system: rows of the active cells
        # sorted (diagonal first), the input arrays compressed in input order.
        ia = Vector{Int32}(undef, nodes + 1)
        ia[1] = 1
        ja = Int32[]
        buf = Int32[]
        for nu in 1:nodesuser
            nr = nodereduced[nu]
            nr > 0 || continue
            empty!(buf)
            for ipos in (iainp[nu] + 1):(iainp[nu + 1] - 1)
                mr = nodereduced[jainp[ipos]]
                mr > 0 && !(mr in buf) && mr != nr && push!(buf, mr)
            end
            sort!(buf)
            push!(ja, nr)
            append!(ja, buf)
            ia[nr + 1] = length(ja) + 1
        end
        keep = Int[]
        for nu in 1:nodesuser
            nodereduced[nu] > 0 || continue
            for ipos in iainp[nu]:(iainp[nu + 1] - 1)
                nodereduced[jainp[ipos]] > 0 && push!(keep, ipos)
            end
        end
        length(keep) == length(ja) || throw(MF6InputError(file, 0, "Duplicate connections in JA."))
        ihct, cl12t, hwvat, angt = dis.ihc[keep], dis.cl12[keep], dis.hwva[keep], angin[keep]
    end
    nja = length(ja)

    # Connections.f90 `con_finalize`.
    errors = String[]
    for n in 1:nodes
        m = ja[ia[n]]
        n == m || push!(errors, "Error in array: JA. First value for cell : $n must equal $n. Found $m instead.")
        for ii in (ia[n] + 1):(ia[n + 1] - 2)
            ja[ii] > ja[ii + 1] && push!(errors, "Error in array: JA. Entries not sorted for row: $n. Offending entries are: $(ja[ii]) and $(ja[ii + 1])")
        end
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    isym = fill_isym!(Vector{Int32}(undef, nja), ia, ja)
    for n in 1:nodes, ii in ia[n]:(ia[n + 1] - 1)
        isym[ii] == 0 && push!(errors, "Error in array: JA. Array does not have symmetric counterpart in position $ii for cell $n connected to cell $(ja[ii])")
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    jas = Vector{Int32}(undef, nja)
    njas = Int(fill_jas!(jas, ia, ja, isym))
    ihc = Vector{Int32}(undef, njas)
    cl1 = Vector{Float64}(undef, njas)
    cl2 = Vector{Float64}(undef, njas)
    hwva = Vector{Float64}(undef, njas)
    anglex = fill(DNODATA, njas)
    symerr(name, ii, a, b, n, m) = "Error in array: $name. Array is not symmetric in positions: $ii and $(isym[ii]). Values in these positions are: $(_g15(a)) and $(_g15(b)) For node $n connected to node $m"
    for n in 1:nodes, ii in (ia[n] + 1):(ia[n + 1] - 1)
        m = ja[ii]
        if ihct[ii] != ihct[isym[ii]]
            push!(errors, symerr("IHC", ii, ihct[ii], ihct[isym[ii]], n, m))
        else
            ihc[jas[ii]] = ihct[ii]
        end
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    for n in 1:nodes, ii in (ia[n] + 1):(ia[n + 1] - 1)
        m = ja[ii]
        if m > n
            cl1[jas[ii]] = cl12t[ii]
        elseif n > m
            cl2[jas[ii]] = cl12t[ii]
        end
    end
    for n in 1:nodes, ii in (ia[n] + 1):(ia[n + 1] - 1)
        m = ja[ii]
        hwvat[ii] != hwvat[isym[ii]] && push!(errors, symerr("HWVA", ii, hwvat[ii], hwvat[isym[ii]], n, m))
        (ihct[ii] < 0 || ihct[ii] > 2) && push!(errors, "IHC must be 0, 1, or 2.  Found: $(ihct[ii])")
        hwva[jas[ii]] = hwvat[ii]
    end
    isempty(errors) || throw(MF6InputError(file, 0, join(errors, "\n")))
    if hasangle
        dpio180 = atan(1.0) / 45.0
        for n in 1:nodes, ii in (ia[n] + 1):(ia[n + 1] - 1)
            ja[ii] < n && continue
            anglex[jas[ii]] = angt[ii] * dpio180
        end
    end

    # Reduced cell arrays (Disu.f90 `grid_finalize`).
    top = dis.top[nodeuser]
    bot = dis.bot[nodeuser]
    area = dis.area[nodeuser]
    xc = dis.nvert > 0 ? dis.cell_center[1, nodeuser] : Float64[]
    yc = dis.nvert > 0 ? dis.cell_center[2, nodeuser] : Float64[]
    return Connectivity(
        Int32(nodes), Int32(nodesuser), Int32(nja), Int32(njas), nodeuser, nodereduced,
        Vector{Int32}(ia), ja, isym, jas, ihc, cl1, cl2, hwva, anglex, top, bot, area, xc, yc, :disu,
    )
end
