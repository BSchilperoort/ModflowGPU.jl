# Specific discharge at the cell centres (T6.9; gwf-npf.f90 `calc_spdis`, `sav_spdis`,
# `set_edge_properties`; GwfConductanceUtils.f90 `thksatnm`; exg-gwfgwf.f90
# `gwf_gwf_set_flow_to_npf`). Host, Float64, from the FLOW-JA-FACE of the step (the model's
# `nja` values as written to the budget file) when NPF SAVE_SPECIFIC_DISCHARGE asks for the
# DATA-SPDIS budget record. Every cell is independent; the work is a loop over the
# connections, done only for steps whose budget is saved.

# GwfConductanceUtils.f90 `thksatnm`: the saturated thickness of a horizontal connection.
function _thksatnm(ibn, ibm, ictn, ictm, upstream::Bool, ihc, hn, hm, satn, satm, topn, topm, botn, botm)
    (ibn == 0 || ibm == 0) && return 0.0
    if ictn == 0 && ictm == 0
        if ihc == 2
            thkn = max(min(topn, topm) - max(botn, botm), 0.0)
            thkm = thkn
        else
            thkn = topn - botn
            thkm = topm - botm
        end
        return 0.5 * (thkn + thkm)
    elseif upstream
        return hn > hm ? satn * (topn - botn) : satm * (topm - botm)
    end
    if ihc == 2
        thkn = staggered_thkfrac(topn, botn, satn, topm, botm)
        thkm = staggered_thkfrac(topm, botm, satm, topn, botn)
    else
        thkn = satn * (topn - botn)
        thkm = satm * (topm - botm)
    end
    return 0.5 * (thkn + thkm)
end

# An exchange connection seen from one of its cells (`set_edge_properties`): the flow `q`
# (into the cell; vertical: positive upward), `area`, the normal (`nx`, `ny`) and the
# distance from the cell centre.
struct _SpdisEdge
    node::Int
    ihc::Int
    q::Float64
    area::Float64
    nx::Float64
    ny::Float64
    dist::Float64
end

"""
    check_spdis(dis, conn, file)

The checks of `calc_spdis` for NPF SAVE_SPECIFIC_DISCHARGE: a DISU grid needs ANGLDEGX
("ANGLDEGX not provided in discretization file") and its vertices (`connection_vector`:
"Cannot calculate unit vector components for DISU grid if VERTEX data are not
specified"). Throws an `MF6InputError`.
"""
function check_spdis(dis::AbstractDisSpec, conn::Connectivity, file::AbstractString)
    dis isa DisuSpec || return nothing
    isnothing(dis.angldegx) && throw(MF6InputError(file, 0, "Error.  ANGLDEGX not provided in discretization file.  ANGLDEGX required for calculation of specific discharge."))
    isempty(conn.xc) && throw(MF6InputError(file, 0, "Cannot calculate unit vector components for DISU grid if VERTEX data are not specified"))
    return nothing
end

"""
    specific_discharge(m, flowja) -> Matrix{Float64}

The specific discharge `(3, nodes)` (qx, qy, qz per reduced node) of model `m` (a
`GwfModel` or a `ModelView` of a multi-model solution) for the FLOW-JA-FACE `flowja` of
the model (`model_flowja`), as gwf-npf.f90 `calc_spdis` computes it: per cell the face
velocities of its connections (flow over the saturated face area, `thksatnm`; vertical over
the cell area), weighted by the distances and normals (`nozee`), plus the connections of the
GWF-GWF exchanges of the model (`gwf_gwf_set_flow_to_npf`: ANGLDEGX and CDIST).
"""
function specific_discharge(m::Union{GwfModel, ModelView}, flowja::AbstractVector{Float64})
    conn = _vconn(m)
    spec = _vspec(m)
    mm = _vmodel(m)
    rng = _vrange(m)
    nodes = Int(conn.nodes)
    ib = mm.ibound_host[rng]
    ictu = _mf6_icelltype(spec.npf)
    ict = Int32[ictu[conn.nodeuser[n]] for n in 1:nodes]
    cv = mm.npf.cv
    sat = isnothing(cv) ? ones(nodes) : Float64.(Array(cv.sat))[rng]
    upstream = spec.newton
    h = upstream && !isnothing(cv) ? absolute_heads(m) : zeros(nodes)
    edges = m isa ModelView ? _spdis_edges(m) : _SpdisEdge[]
    eptr = _edge_lookup(edges, nodes)
    out = zeros(3, nodes)
    nix, niy, di, vi = Float64[], Float64[], Float64[], Float64[]
    viz, diz = Float64[], Float64[]
    for n in 1:nodes
        empty!(nix); empty!(niy); empty!(di); empty!(vi); empty!(viz); empty!(diz)
        for ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
            mn = Int(conn.ja[ii])
            jj = Int(conn.jas[ii])
            ihc = Int(conn.ihc[jj])
            area = conn.hwva[jj]
            _, dltot = _xt3d_connection_vector(conn, n, mn, true, ihc, jj)
            cl1, cl2 = mn < n ? (conn.cl2[jj], conn.cl1[jj]) : (conn.cl1[jj], conn.cl2[jj])
            d = dltot * cl1 * (1.0 / (cl1 + cl2))
            if ihc == 0
                push!(diz, d)
                qz = n > mn ? -flowja[ii] : flowja[ii]
                push!(viz, qz / area)
            else
                dz = _thksatnm(ib[n], ib[mn], ict[n], ict[mn], upstream, ihc, h[n], h[mn], sat[n], sat[mn], conn.top[n], conn.top[mn], conn.bot[n], conn.bot[mn])
                area *= dz
                xn, yn, _ = _connection_normal(conn, n, mn, ihc, jj)
                push!(nix, -xn)
                push!(niy, -yn)
                push!(di, d)
                push!(vi, area > 0 ? flowja[ii] / area : 0.0)
            end
        end
        for k in eptr[n]:(eptr[n + 1] - 1)
            e = edges[k]
            if e.ihc == 0
                push!(viz, e.q / e.area)
                push!(diz, e.dist)
            else
                push!(nix, -e.nx)
                push!(niy, -e.ny)
                push!(di, e.dist)
                push!(vi, e.area > 0 ? e.q / e.area : 0.0)
            end
        end
        out[:, n] .= _spdis_cell(nix, niy, di, vi, viz, diz)
    end
    return out
end

# The weighting of `calc_spdis` for one cell (DEM10 = 1e-10).
function _spdis_cell(nix, niy, di, vi, viz, diz)
    ncz = length(viz)
    vz = 0.0
    if ncz == 1
        vz = viz[1]
    elseif ncz > 1
        dsumz = sum(diz)
        dsumz = dsumz + 1.0e-10 * dsumz
        denom = ncz - 1.0
        for iz in 1:ncz
            w = dsumz > 0 ? 1.0 - diz[iz] / dsumz : 0.0
            vz += (w / denom) * viz[iz]
        end
    end
    nc = length(vi)
    wix = [di[ic] * abs(nix[ic]) for ic in 1:nc]
    wiy = [di[ic] * abs(niy[ic]) for ic in 1:nc]
    dsumx = sum(wix; init = 0.0)
    dsumy = sum(wiy; init = 0.0)
    dsumx = dsumx + 1.0e-10 * dsumx
    dsumy = dsumy + 1.0e-10 * dsumy
    for ic in 1:nc
        wix[ic] = (dsumx - wix[ic]) * abs(nix[ic])
        wiy[ic] = (dsumy - wiy[ic]) * abs(niy[ic])
    end
    bix = similar(wix)
    biy = similar(wiy)
    dsumx = 0.0
    dsumy = 0.0
    for ic in 1:nc
        bix[ic] = wix[ic] * sign1(nix[ic])
        biy[ic] = wiy[ic] * sign1(niy[ic])
        dsumx += wix[ic] * abs(nix[ic])
        dsumy += wiy[ic] * abs(niy[ic])
    end
    dsumx > 0 && (dsumx = 1.0 / dsumx)
    dsumy > 0 && (dsumy = 1.0 / dsumy)
    axy = 0.0
    ayx = 0.0
    for ic in 1:nc
        bix[ic] *= dsumx
        biy[ic] *= dsumy
        axy += bix[ic] * niy[ic]
        ayx += biy[ic] * nix[ic]
    end
    vx = 0.0
    vy = 0.0
    for ic in 1:nc
        vx += (bix[ic] - axy * biy[ic]) * vi[ic]
        vy += (biy[ic] - ayx * bix[ic]) * vi[ic]
    end
    denom = 1.0 - axy * ayx
    if denom != 0
        vx /= denom
        vy /= denom
    end
    return (vx, vy, vz)
end

# Fortran `sign(1.0, x)`: +1 for x = +0.0, -1 for x = -0.0.
sign1(x::Float64) = signbit(x) ? -1.0 : 1.0

# The edges of each node in edge order (`prepare_edge_lookup`): `edges` sorted by node
# (stable) and the first edge of each node (`nodes + 1` entries).
function _edge_lookup(edges::Vector{_SpdisEdge}, nodes::Int)
    sort!(edges; by = e -> e.node, alg = Base.Sort.DEFAULT_STABLE)
    ptr = ones(Int, nodes + 1)
    p = 1
    for n in 1:nodes
        ptr[n] = p
        while p <= length(edges) && edges[p].node == n
            p += 1
        end
    end
    ptr[nodes + 1] = p
    return ptr
end

# The exchange connections of model `v` (exg-gwfgwf.f90 `gwf_gwf_set_flow_to_npf`), in
# exchange order: for each connection the edge of the model-1 cell, then that of the
# model-2 cell (a periodic exchange gives both to the same model). Multi-model solutions
# are confined (saturation 1).
function _spdis_edges(v::ModelView)
    mm = v.model
    sol = mm.solution
    edges = _SpdisEdge[]
    x = _exchange_flow_values(v)
    for (ie, ex) in enumerate(sol.exchanges)
        (ex.m1 == v.index || ex.m2 == v.index) || continue
        d = ex.spec.data
        ia = findfirst(==("ANGLDEGX"), uppercase.(d.auxnames))
        ic = findfirst(==("CDIST"), uppercase.(d.auxnames))
        p1, p2 = sol.models[ex.m1], sol.models[ex.m2]
        ictu1, ictu2 = _mf6_icelltype(p1.spec.npf), _mf6_icelltype(p2.spec.npf)
        q = x[ie]
        for i in eachindex(ex.node1)
            n1, n2 = Int(ex.node1[i]), Int(ex.node2[i])
            ihc = Int(d.ihc[i])
            rrate = q[i]
            c1, c2 = p1.conn, p2.conn
            if ihc == 0
                nx = ny = 0.0
                area = d.hwva[i]
                c1.bot[n1] < c2.bot[n2] && (rrate = -rrate)
            else
                isnothing(ia) && throw(MF6InputError(ex.spec.file, 0, "error in gwf_gwf_cq"))
                a = d.auxvar[ia, i] * (pi / 180)
                nx, ny = cos(a), sin(a)
                ict1 = ictu1[c1.nodeuser[n1]]
                ict2 = ictu2[c2.nodeuser[n2]]
                thk = _thksatnm(1, 1, ict1, ict2, false, ihc, 0.0, 0.0, 1.0, 1.0, c1.top[n1], c2.top[n2], c1.bot[n1], c2.bot[n2])
                area = d.hwva[i] * thk
            end
            isnothing(ic) && throw(MF6InputError(ex.spec.file, 0, "error in gwf_gwf_cq"))
            dltot = d.auxvar[ic, i]
            ex.m1 == v.index && push!(edges, _SpdisEdge(n1, ihc, rrate, area, nx, ny, dltot * d.cl1[i] / (d.cl1[i] + d.cl2[i])))
            if ex.m2 == v.index
                r2 = ihc != 0 ? -rrate : rrate
                push!(edges, _SpdisEdge(n2, ihc, r2, area, -nx, -ny, dltot * d.cl2[i] / (d.cl1[i] + d.cl2[i])))
            end
        end
    end
    return edges
end

# The flows into the model-1 cells of every exchange (MF6's SIMVALS), from the merged
# FLOW-JA-FACE on the host.
function _exchange_flow_values(v::ModelView)
    mm = v.model
    f = Float64.(Array(mm.flowja))
    return [f[ex.pos] for ex in mm.solution.exchanges]
end

"""
    write_spdis(io, layout, m, flowja; kw...)

The DATA-SPDIS record of the budget file (gwf-npf.f90 `sav_spdis`): a list with one entry
per reduced node (`id1 = id2` its user node, `q = 0`, auxiliary `qx`, `qy`, `qz`), source and
destination the model and its NPF package.
"""
function write_spdis(io::IO, layout::OutputLayout, m::Union{GwfModel, ModelView}, flowja::AbstractVector{Float64}; kw...)
    spec = _vspec(m)
    conn = _vconn(m)
    sp = specific_discharge(m, flowja)
    nodeu = Int32.(conn.nodeuser)
    list = BudgetList(nodeu, copy(nodeu), zeros(Int(conn.nodes)), sp)
    name = uppercase(spec.name)
    npfname = uppercase(_npf_package_name(spec))
    aux = [lpad(s, 16) for s in ("qx", "qy", "qz")]
    write_budget_list(io, layout, "DATA-SPDIS", list; kw..., modelnam1 = name, paknam1 = npfname, modelnam2 = name, paknam2 = npfname, auxtxt = aux)
    return nothing
end

"""
    cell_saturation(m) -> Vector{Float64}

The saturation NPF holds for each reduced node of model `m` (a `GwfModel` or a
`ModelView`; gwf-npf.f90 `this%sat`, T6.10): for convertible cells the value of the last
formulation (`cf_default_flow`: `thksat` of the heads of the last outer iteration, 0 for
a dry cell; the smoothed saturation with Newton), for confined cells the initial
saturation (`npf_ar`: 1, or the THICKSTRT saturation of the starting head).
"""
function cell_saturation(m::Union{GwfModel, ModelView})
    mm = _vmodel(m)
    rng = _vrange(m)
    cv = mm.npf.cv
    return isnothing(cv) ? mm.setup.properties.sat0[rng] : Float64.(Array(cv.sat))[rng]
end

"""
    write_sat(io, layout, m; kw...)

The DATA-SAT record of the budget file (gwf-npf.f90 `sav_sat`, NPF SAVE_SATURATION; T6.10):
a list with one entry per reduced node (`id1 = id2` its user node, `q = 0`, auxiliary
`sat`, `cell_saturation`), source and destination the model and its NPF package.
"""
function write_sat(io::IO, layout::OutputLayout, m::Union{GwfModel, ModelView}; kw...)
    spec = _vspec(m)
    conn = _vconn(m)
    nodeu = Int32.(conn.nodeuser)
    list = BudgetList(nodeu, copy(nodeu), zeros(Int(conn.nodes)), permutedims(cell_saturation(m)))
    name = uppercase(spec.name)
    npfname = uppercase(_npf_package_name(spec))
    write_budget_list(io, layout, "DATA-SAT", list; kw..., modelnam1 = name, paknam1 = npfname, modelnam2 = name, paknam2 = npfname, auxtxt = [lpad("sat", 16)])
    return nothing
end

# The name of the model's NPF package (name file PACKAGES block; "NPF" by default).
function _npf_package_name(spec::GwfSpec)
    k = findfirst(p -> uppercase(p.ftype) == "NPF6", spec.packages)
    return (isnothing(k) || isempty(spec.packages[k].name)) ? "NPF" : spec.packages[k].name
end
