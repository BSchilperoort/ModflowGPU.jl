# Multi-model solutions and GWF-GWF exchanges (T6.6): the host setup, in Float64.
#
# Ports MF6 6.8.2:
# - NumericalSolution.f90 (`sln_df`, `sln_connect`): one matrix over the models of a
#   solution, each model's nodes after those of the models before it (`moffset`), the
#   exchange connections added to the sparse structure (exg-gwfgwf.f90 `gwf_gwf_ac`/`_mc`);
# - exg-gwfgwf.f90 `validate_exchange` (the checks), `calc_cond_sat` (the saturated
#   conductance of each exchange connection: `hcond`/`vcond` with the cells' `hy_eff`, the
#   exchange's CELL_AVERAGING), `condcalc` (confined cells: the conductance is condsat),
#   `gwf_gwf_fc` (the conductances into the solution matrix: off-diagonal entries and the
#   diagonals of both cells);
# - the interface model of an exchange with XT3D (ConnectionBuilder.f90,
#   SpatialModelConnection.f90 `spatialcon_fc`/`maskOwnerConnections`, GridConnection.f90
#   `extendConnection`/`createConnectionMask`/`fillConnectionData*`/`getDiscretization`,
#   GwfGwfConnection.f90 `setGridExtent`, `gwfgwfcon_df`, GwfInterfaceModel.f90): see
#   `_interface_connections` and `_interface_xt3d!`.
#
# Port of behaviour, not of structure. MF6 assembles the models' and the exchanges' terms
# into one solution matrix. Here the models of a solution become one merged model on the
# device (`SolutionSetup`): the connectivity of every model, renumbered with the model's node
# offset, plus the exchange connections as ordinary connections of the merged CSR (diagonal
# first, then the columns in increasing order), with their saturated conductance. Confined
# models (the scope of this card) have constant conductances, so the existing NPF kernels
# formulate, solve and compute the flows of the merged model unchanged, and the flow of an
# exchange connection is the merged FLOW-JA-FACE entry of its cells (`qcalc`:
# `cond·(h_m2 − h_m1)`, positive into the model-1 cell).
#
# XT3D on the exchange (MF6: the interface model). In serial MF6 makes one interface model
# per side of an exchange with XT3D (`use_interface_model`): a DISU model of the cells
# within the stencil depth of the exchange (two connections across it; also two into the
# owner model when that model has XT3D itself, else one), in global coordinates (XORIGIN,
# YORIGIN, ANGROT), with ANGLE1–3 enabled (`nozee` false). Its NPF (XT3D) computes the
# coefficients of the connections the mask leaves to it: the exchange connections, and with
# XT3D in the owner model the connections from the owner's exchange cells to their
# neighbours in the owner model (masked out of the owner's NPF). The rows of the owner's
# cells go into the solution matrix. The interface region holds the complete neighbour
# lists of the cells of these connections, so their XT3D fluxes are those of XT3D on the
# merged grid with the interface's geometry; `_interface_xt3d!` computes them there. Every
# other connection keeps its model's formulation (XT3D of the model alone, or the
# conductance), and an exchange without XT3D its conductance. All of these are flux terms
# `c·(h_a − h_b)` per connection (`Xt3dSetup`), so the merged model runs on the XT3D kernels
# when any XT3D is involved.
#
# Scope: confined models (ICELLTYPE 0, STO ICONVERT 0, no NEWTON) in one IMS solution.
# Convertible cells, NEWTON, GNC6 and MVR6 on exchanges, and more than one solution are an
# `MF6UnsupportedError`.

"""
    ModelPart

One model of a multi-model solution (host): `spec`, its own `conn` and NPF `setup` (as for a
single-model run), `offset` and `uoffset` (its first reduced and user node in the solution's
numbering, minus one), `jamap` (model CSR entry → merged CSR entry, `nja` of the model) and
`packages` (its stress packages in the merged package list).
"""
struct ModelPart
    spec::GwfSpec
    conn::Connectivity
    setup::NpfSetup
    offset::Int32
    uoffset::Int32
    jamap::Vector{Int32}
    packages::UnitRange{Int}
end

"""
    ExchangeSetup

One GWF-GWF exchange of a solution (host): `spec`, `name` (MF6's `GWF-GWF_<id>`), `id`,
the model indices `m1`, `m2` in the solution (equal for a periodic exchange), per exchange
connection the reduced nodes `node1` (model 1) and `node2` (model 2), the merged nodes `g1`,
`g2`, `pos` (the merged CSR entry of row `g1`, column `g2`: its FLOW-JA-FACE is the flow into
the model-1 cell, MF6's SIMVALS) and `condsat` (`calc_cond_sat`).
"""
struct ExchangeSetup
    spec::ExchangeSpec
    name::String
    id::Int
    m1::Int
    m2::Int
    node1::Vector{Int32}
    node2::Vector{Int32}
    g1::Vector{Int32}
    g2::Vector{Int32}
    pos::Vector{Int32}
    condsat::Vector{Float64}
end

"""
    SolutionSetup

The host setup of a multi-model solution (`solution_setup`): the `models` (`ModelPart`),
the `exchanges` (`ExchangeSetup`), the merged connectivity `conn` and NPF setup `setup`
(condsat of the model and exchange connections, reference heads of every model), `exgentry`
(per merged CSR entry: an exchange connection), `xt3d` (`Xt3dSetup` of the merged model when
a model or an exchange uses XT3D, else `nothing`), and the merged `ic` and `sto` (`nothing`
when no model has STO).
"""
struct SolutionSetup
    models::Vector{ModelPart}
    exchanges::Vector{ExchangeSetup}
    conn::Connectivity
    setup::NpfSetup
    exgentry::Vector{Bool}
    xt3d::Union{Nothing, Xt3dSetup}
    ic::IcSpec
    sto::Union{Nothing, StoSpec}
end

nmodels(s::SolutionSetup) = length(s.models)

# Index of an auxiliary variable of an exchange (0 when not given).
_exg_aux(ex::ExchangeSpec, name) = something(findfirst(==(name), ex.auxnames), 0)

# What this port supports in a multi-model solution, and MF6's own checks of
# `validate_exchange` and `validateConnection`.
function _check_solution(models::Vector{GwfSpec}, setups, exchanges::Vector{ExchangeSpec})
    for (m, s) in zip(models, setups)
        file = isempty(m.file) ? m.name : m.file
        m.newton && unsupported(file, "NEWTON in model $(m.name) of a multi-model solution (only confined models are supported with exchanges)")
        any(!=(0), s.properties.icelltype) &&
            unsupported(file, "convertible cells (ICELLTYPE /= 0) in model $(m.name) of a multi-model solution (only confined models are supported with exchanges)")
        !isnothing(m.sto) && any(!=(0), m.sto.iconvert) &&
            unsupported(file, "STO ICONVERT /= 0 in model $(m.name) of a multi-model solution (only confined storage is supported with exchanges)")
    end
    rhs = [m.npf.xt3d_rhs for m in models if m.npf.xt3d]
    length(unique(rhs)) <= 1 || unsupported(models[1].file, "XT3D with the RHS option in some models of a solution and without it in others")
    any(rhs) && any(ex -> ex.xt3d, exchanges) && unsupported(models[1].file, "XT3D with the RHS option in a model and XT3D on an exchange")
    byname = Dict(m.name => i for (i, m) in enumerate(models))
    for (id, ex) in enumerate(exchanges)
        file = isempty(ex.file) ? "GWF6-GWF6" : ex.file
        name = "GWF-GWF_$id"
        i1, i2 = byname[ex.model1], byname[ex.model2]
        # exg-gwfgwf.f90 validate_exchange
        ex.xt3d && i1 == i2 && throw(
            MF6InputError(file, 0, "GWF-GWF exchange $name is a periodic boundary condition which cannot be configured with XT3D"),
        )
        ianglex = _exg_aux(ex, "ANGLDEGX")
        icdist = _exg_aux(ex, "CDIST")
        ex.xt3d && ianglex == 0 && throw(
            MF6InputError(file, 0, "GWF-GWF exchange $name requires that ANGLDEGX be specified as an auxiliary variable because XT3D is enabled"),
        )
        m1, m2 = models[i1], models[i2]
        if (setups[i1].properties.ik22 || setups[i2].properties.ik22) && ianglex == 0
            throw(
                MF6InputError(
                    file, 0,
                    "GWF-GWF exchange $name requires that ANGLDEGX be specified as an auxiliary variable because K22 was specified in one or both groundwater models.",
                ),
            )
        end
        if m1.npf.save_specific_discharge || m2.npf.save_specific_discharge
            ianglex == 0 && throw(
                MF6InputError(
                    file, 0,
                    "GWF-GWF exchange $name requires that ANGLDEGX be specified as an auxiliary variable because specific discharge is being calculated in one or both groundwater models.",
                ),
            )
            icdist == 0 && throw(
                MF6InputError(
                    file, 0,
                    "GWF-GWF exchange $name requires that CDIST be specified as an auxiliary variable because specific discharge is being calculated in one or both groundwater models.",
                ),
            )
        end
        if ex.xt3d
            # SpatialModelConnection.f90 validateConnection: models without ANGLDEGX.
            for m in (m1, m2)
                if m.dis isa DisuSpec && (isnothing(m.dis.angldegx) || m.dis.nvert == 0)
                    unsupported(file, "XT3D on exchange $name with model $(m.name), a DISU grid without ANGLDEGX or vertices")
                end
            end
        end
    end
    return nothing
end

# exg-gwfgwf.f90 `calc_cond_sat` for exchange connection `i`: cells `n` (model 1, properties
# `p1`, connectivity `c1`) and `m` (model 2). Saturation 1 (confined cells).
function _exchange_condsat(ex::ExchangeSpec, i, p1::NpfProperties, c1::Connectivity, n, p2::NpfProperties, c2::Connectivity, m)
    d = ex.data
    ihc = d.ihc[i]
    topn, botn, topm, botm = c1.top[n], c1.bot[n], c2.top[m], c2.bot[m]
    satn = satm = 1.0
    if ihc == 0
        vg = (0.0, 0.0, 1.0)
        hyn = hy_eff(p1, n, 0, vg)
        hym = hy_eff(p2, m, 0, vg)
        # GwfConductanceUtils.f90 vcond(1, 1, 1, 1, 0, 1, 1, DONE, botn, botm, ...): the
        # DEWATERED branch with the heads at the cell bottoms.
        if botn > botm
            satm < 1.0 && (satm = 0.0)
        else
            satn < 1.0 && (satn = 0.0)
        end
        bovk1 = satn * (topn - botn) * 0.5 / hyn
        bovk2 = satm * (topm - botm) * 0.5 / hym
        denom = bovk1 + bovk2
        return (denom != 0.0 ? d.hwva[i] / denom : 0.0), hyn, hym
    end
    hyn = p1.k11[n]
    hym = p2.k11[m]
    ianglex = _exg_aux(ex, "ANGLDEGX")
    if ianglex > 0
        angle = d.auxvar[ianglex, i] * (pi / 180.0)  # DPIO180
        vg = (abs(cos(angle)), abs(sin(angle)), 0.0)
        p1.ik22 && (hyn = hy_eff(p1, n, ihc, vg))
        p2.ik22 && (hym = hy_eff(p2, m, ihc, vg))
    end
    # hcond(1, 1, 1, 1, 0, ihc, icellavg, DONE, ...) → convertible_standard.
    if ihc == 2
        thksatn = staggered_thkfrac(topn, botn, satn, topm, botm)
        thksatm = staggered_thkfrac(topm, botm, satm, topn, botn)
    else
        thksatn = satn * (topn - botn)
        thksatm = satm * (topm - botm)
    end
    return condmean(hyn, hym, thksatn, thksatm, d.cl1[i], d.cl2[i], d.hwva[i], _ICELLAVG[ex.cell_averaging]), hyn, hym
end

# Concatenate the NPF properties of the models (angles: zeros where a model has none).
function _merge_properties(ps::Vector{NpfProperties})
    cat(f) = reduce(vcat, (getfield(p, f) for p in ps))
    function angles(f)
        all(p -> isnothing(getfield(p, f)), ps) && return nothing
        return reduce(vcat, (isnothing(getfield(p, f)) ? zeros(length(p.k11)) : getfield(p, f) for p in ps))
    end
    wetdry = all(p -> isnothing(p.wetdry), ps) ? nothing : reduce(vcat, (isnothing(p.wetdry) ? zeros(length(p.k11)) : p.wetdry for p in ps))
    return NpfProperties(
        cat(:icelltype), cat(:k11), cat(:k22), cat(:k33), angles(:angle1), angles(:angle2), angles(:angle3),
        any(p -> p.ik22, ps), any(p -> p.ik33, ps), ps[1].icellavg, cat(:thickstart), cat(:sat0), wetdry,
    )
end

# BaseDis.f90 `dis_transform_xy`: model to global coordinates.
function _transform_xy(x, y, xorigin, yorigin, angrot)
    xg, yg = x, y
    if angrot != 0.0
        a = angrot * (pi / 180.0)
        xg = x * cos(a) - y * sin(a)
        yg = x * sin(a) + y * cos(a)
    end
    return xg + xorigin, yg + yorigin
end

_grid_origin(d::AbstractDisSpec) = (d.xorigin, d.yorigin, d.angrot)

"""
    solution_setup(sim::SimulationSpec) -> SolutionSetup

The host setup of the models of the (only) IMS solution of `sim`, in the order of the
solution's model list, connected by `sim.exchanges` (see the header of
src/gwf/exchange.jl). Throws `MF6UnsupportedError` for what is out of scope and MF6's input
errors of `validate_exchange`.
"""
function solution_setup(sim::SimulationSpec)
    ims = only(only(sim.solution_groups).solutions)
    order = isempty(ims.models) ? [m.name for m in sim.models] : ims.models
    models = GwfSpec[model_spec(sim, n) for n in order]
    return solution_setup(models, sim.exchanges)
end

function solution_setup(models::Vector{GwfSpec}, exchanges::Vector{ExchangeSpec})
    conns = Connectivity[build_connectivity(m) for m in models]
    setups = NpfSetup[npf_setup(m, c) for (m, c) in zip(models, conns)]
    _check_solution(models, setups, exchanges)
    byname = Dict(m.name => i for (i, m) in enumerate(models))
    nm = length(models)
    offset = zeros(Int32, nm)
    uoffset = zeros(Int32, nm)
    for k in 2:nm
        offset[k] = offset[k - 1] + conns[k - 1].nodes
        uoffset[k] = uoffset[k - 1] + conns[k - 1].nodesuser
    end
    nodes = Int(offset[end] + conns[end].nodes)
    nodesuser = Int(uoffset[end] + conns[end].nodesuser)
    # Exchange connections in merged numbering.
    exg = Tuple{Int, Int, Int32, Int32, Int32, Int32}[]  # (exchange, connection, node1, node2, g1, g2)
    for (ie, ex) in enumerate(exchanges)
        k1, k2 = byname[ex.model1], byname[ex.model2]
        for i in eachindex(ex.data.nodem1)
            n1 = conns[k1].nodereduced[ex.data.nodem1[i]]
            n2 = conns[k2].nodereduced[ex.data.nodem2[i]]
            push!(exg, (ie, i, n1, n2, n1 + offset[k1], n2 + offset[k2]))
        end
    end
    # Rows: the models' own connections and the exchange partners, sorted, diagonal first.
    rows = [Int32[] for _ in 1:nodes]
    for k in 1:nm, n in 1:conns[k].nodes
        c = conns[k]
        g = n + offset[k]
        for ii in (c.ia[n] + 1):(c.ia[n + 1] - 1)
            push!(rows[g], c.ja[ii] + offset[k])
        end
    end
    for (ie, i, _, _, g1, g2) in exg
        file = isempty(exchanges[ie].file) ? "GWF6-GWF6" : exchanges[ie].file
        (g1 == g2) && unsupported(file, "exchange connection $i of GWF-GWF_$ie connects a cell to itself")
        (g2 in rows[g1]) && unsupported(
            file, "exchange connection $i of GWF-GWF_$ie connects two cells that are already connected (in the model or by another exchange connection)",
        )
        push!(rows[g1], g2)
        push!(rows[g2], g1)
    end
    ia = Vector{Int32}(undef, nodes + 1)
    ja = Int32[]
    ia[1] = 1
    for g in 1:nodes
        push!(ja, Int32(g))
        append!(ja, sort!(rows[g]))
        ia[g + 1] = length(ja) + 1
    end
    nja = length(ja)
    isym = Vector{Int32}(undef, nja)
    fill_isym!(isym, ia, ja)
    jas = Vector{Int32}(undef, nja)
    njas = fill_jas!(jas, ia, ja, isym)
    position(g, h) = (lo = ia[g] + 1; hi = ia[g + 1] - 1; p = searchsortedfirst(view(ja, lo:hi), h) + lo - 1; (p <= hi && ja[p] == h) ? Int32(p) : error("solution_setup: no entry ($g, $h)"))
    # Model CSR entries in the merged CSR.
    jamaps = Vector{Int32}[]
    for k in 1:nm
        c = conns[k]
        jm = Vector{Int32}(undef, c.nja)
        for n in 1:c.nodes
            g = n + offset[k]
            jm[c.ia[n]] = ia[g]
            for ii in (c.ia[n] + 1):(c.ia[n + 1] - 1)
                jm[ii] = position(g, c.ja[ii] + offset[k])
            end
        end
        push!(jamaps, jm)
    end
    # Per merged symmetric connection: geometry and condsat.
    ihc = zeros(Int32, njas)
    cl1 = zeros(njas)
    cl2 = zeros(njas)
    hwva = zeros(njas)
    anglex = zeros(njas)
    condsat = zeros(njas)
    hyl = zeros(njas)
    hyh = zeros(njas)
    for k in 1:nm
        c = conns[k]
        s = setups[k]
        for n in 1:c.nodes, ii in (c.ia[n] + 1):(c.ia[n + 1] - 1)
            c.ja[ii] > n || continue
            j = c.jas[ii]
            jj = jas[jamaps[k][ii]]
            ihc[jj], cl1[jj], cl2[jj], hwva[jj], anglex[jj] = c.ihc[j], c.cl1[j], c.cl2[j], c.hwva[j], c.anglex[j]
            condsat[jj], hyl[jj], hyh[jj] = s.condsat[j], s.hyl[j], s.hyh[j]
        end
    end
    exgentry = falses(nja)
    xsetups = ExchangeSetup[]
    for (ie, ex) in enumerate(exchanges)
        k1, k2 = byname[ex.model1], byname[ex.model2]
        ianglex = _exg_aux(ex, "ANGLDEGX")
        sel = [e for e in exg if e[1] == ie]
        n1 = Int32[e[3] for e in sel]
        n2 = Int32[e[4] for e in sel]
        g1 = Int32[e[5] for e in sel]
        g2 = Int32[e[6] for e in sel]
        pos = Int32[position(g1[i], g2[i]) for i in eachindex(g1)]
        cs = zeros(length(sel))
        for i in eachindex(sel)
            p = pos[i]
            exgentry[p] = true
            exgentry[isym[p]] = true
            jj = jas[p]
            c, hn, hm = _exchange_condsat(ex, i, setups[k1].properties, conns[k1], n1[i], setups[k2].properties, conns[k2], n2[i])
            cs[i] = c
            condsat[jj] = c
            d = ex.data
            ihc[jj], hwva[jj] = d.ihc[i], d.hwva[i]
            ang = ianglex > 0 ? d.auxvar[ianglex, i] : 0.0
            # GridConnection.f90 fillConnectionDataFromExchanges: cl1 belongs to the
            # lower-numbered cell, ANGLDEGX points from it to the other one.
            if g1[i] < g2[i]
                cl1[jj], cl2[jj], hyl[jj], hyh[jj] = d.cl1[i], d.cl2[i], hn, hm
                anglex[jj] = ang * (pi / 180.0)
            else
                cl1[jj], cl2[jj], hyl[jj], hyh[jj] = d.cl2[i], d.cl1[i], hm, hn
                anglex[jj] = mod(ang + 180.0, 360.0) * (pi / 180.0)
            end
        end
        push!(xsetups, ExchangeSetup(ex, "GWF-GWF_$ie", ie, k1, k2, n1, n2, g1, g2, pos, cs))
    end
    # Nodes: user numbering of each model after those of the models before it.
    nodeuser = reduce(vcat, (c.nodeuser .+ uoffset[k] for (k, c) in enumerate(conns)))
    nodereduced = reduce(vcat, (Int32[r > 0 ? r + offset[k] : r for r in c.nodereduced] for (k, c) in enumerate(conns)))
    cat(f) = reduce(vcat, (getfield(c, f) for c in conns))
    hasxy = all(c -> length(c.xc) == c.nodes, conns)
    xc = hasxy ? cat(:xc) : Float64[]
    yc = hasxy ? cat(:yc) : Float64[]
    conn = Connectivity(
        Int32(nodes), Int32(nodesuser), Int32(nja), Int32(njas), nodeuser, nodereduced, ia, ja, isym, jas, ihc, cl1, cl2,
        hwva, anglex, cat(:top), cat(:bot), cat(:area), xc, yc, :disu,
    )
    props = _merge_properties(NpfProperties[s.properties for s in setups])
    r = reduce(vcat, (s.r for s in setups))
    setup = NpfSetup(props, condsat, r, reference_differences(conn, r), hyl, hyh)
    # Stress packages in model order (each model's in MF6's order).
    parts = ModelPart[]
    ip = 0
    for k in 1:nm
        np = length(models[k].boundaries)
        push!(parts, ModelPart(models[k], conns[k], setups[k], offset[k], uoffset[k], jamaps[k], (ip + 1):(ip + np)))
        ip += np
    end
    xt3d = _solution_xt3d(parts, xsetups, conn, setup)
    ic = IcSpec(; strt = reduce(vcat, (m.ic.strt for m in models)))
    return SolutionSetup(parts, xsetups, conn, setup, Vector{Bool}(exgentry), xt3d, ic, _merge_sto(models))
end

# One StoSpec over the merged user nodes: each model's SS (times the cell thickness when
# STORAGECOEFFICIENT differs between models), SS 0 for a model without STO (steady state
# in every period). Models with STO must agree on the steady-state/transient periods.
function _merge_sto(models::Vector{GwfSpec})
    stos = [m.sto for m in models if !isnothing(m.sto)]
    isempty(stos) && return nothing
    s1 = first(stos)
    for s in stos
        (s.periods.iper == s1.periods.iper && s.periods.data == s1.periods.data) ||
            unsupported(s.file, "STO6 PERIOD blocks (steady state / transient) that differ between the models of a solution")
        isempty(s.tvs_files) || unsupported(s.file, "TVS6 time-varying storage in STO6 (not implemented)")
    end
    sc = s1.storagecoefficient
    same = all(s -> s.storagecoefficient == sc, stos)
    ss = Float64[]
    for m in models
        nu = Int(nodes_user(m.dis))
        if isnothing(m.sto)
            append!(ss, zeros(nu))
        elseif same || m.sto.storagecoefficient
            append!(ss, m.sto.ss)
        else
            # STORAGECOEFFICIENT in another model: SS × thickness here (`SsCapacity`).
            c = build_connectivity(m)
            v = zeros(nu)
            for n in 1:c.nodes
                u = c.nodeuser[n]
                v[u] = m.sto.ss[u] * (c.top[n] - c.bot[n])
            end
            append!(ss, v)
        end
    end
    return StoSpec(;
        iconvert = zeros(Int32, length(ss)), ss, storagecoefficient = same ? sc : true,
        save_flows = any(s -> s.save_flows, stos), periods = s1.periods, file = s1.file,
    )
end

# --- XT3D of a solution

# The connections whose coefficients the interface models compute (the mask of
# GridConnection.f90 `createConnectionMask` and SpatialModelConnection.f90
# `maskOwnerConnections`), as merged symmetric connection numbers: every connection of an
# exchange with XT3D (level 1), and when the owner model has XT3D (`internalStencilDepth`
# 2, GwfGwfConnection.f90 `setGridExtent`) the connections of the owner's exchange cells to
# their neighbours in the owner model (level 2).
function _interface_connections(parts, exchanges, conn::Connectivity)
    owned = falses(conn.njas)
    for ex in exchanges
        ex.spec.xt3d || continue
        for p in ex.pos
            owned[conn.jas[p]] = true
        end
        for (k, cells) in ((ex.m1, ex.g1), (ex.m2, ex.g2))
            part = parts[k]
            part.spec.npf.xt3d || continue
            lo, hi = part.offset + 1, part.offset + part.conn.nodes
            for g in unique(cells), ii in (conn.ia[g] + 1):(conn.ia[g + 1] - 1)
                m = conn.ja[ii]
                (lo <= m <= hi) && (owned[conn.jas[ii]] = true)
            end
        end
    end
    return owned
end

# The interface model's grid (GridConnection.f90 `getDiscretization`): the merged
# connectivity as a DISU grid with the cell centres in global coordinates.
function _interface_grid(parts, conn::Connectivity)
    xc = Float64[]
    yc = Float64[]
    for part in parts
        c = part.conn
        length(c.xc) == c.nodes || error("XT3D on an exchange needs cell coordinates in model $(part.spec.name)")
        xo, yo, ar = _grid_origin(part.spec.dis)
        for n in 1:c.nodes
            x, y = _transform_xy(c.xc[n], c.yc[n], xo, yo, ar)
            push!(xc, x)
            push!(yc, y)
        end
    end
    return Connectivity(
        conn.nodes, conn.nodesuser, conn.nja, conn.njas, conn.nodeuser, conn.nodereduced, conn.ia, conn.ja, conn.isym, conn.jas,
        conn.ihc, conn.cl1, conn.cl2, conn.hwva, conn.anglex, conn.top, conn.bot, conn.area, xc, yc, :disu,
    )
end

# The flux terms (`Xt3dSetup`, flow into the lower-numbered node) of every connection of
# the merged model, then the merged XT3D setup; `nothing` without XT3D in the solution.
function _solution_xt3d(parts, exchanges, conn::Connectivity, setup::NpfSetup)
    anyx = any(p -> p.spec.npf.xt3d, parts) || any(ex -> ex.spec.xt3d, exchanges)
    anyx || return nothing
    rhs = any(p -> p.spec.npf.xt3d && p.spec.npf.xt3d_rhs, parts)
    njas = Int(conn.njas)
    terms = [Tuple{Int32, Int32, Float64}[] for _ in 1:njas]
    # Standard conductance: one term condsat·(h_m − h_n).
    for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
        m = conn.ja[ii]
        m > n || continue
        push!(terms[conn.jas[ii]], (m, Int32(n), setup.condsat[conn.jas[ii]]))
    end
    # XT3D of each model with the XT3D option, on its own grid.
    for part in parts
        part.spec.npf.xt3d || continue
        c = part.conn
        xs = xt3d_setup(c, part.setup.properties, part.setup.r; rhs = part.spec.npf.xt3d_rhs)
        for n in 1:c.nodes, ii in (c.ia[n] + 1):(c.ia[n + 1] - 1)
            c.ja[ii] > n || continue
            j = c.jas[ii]
            jj = conn.jas[part.jamap[ii]]
            empty!(terms[jj])
            for t in xs.tptr[j]:(xs.tptr[j + 1] - 1)
                push!(terms[jj], (xs.ta[t] + part.offset, xs.tb[t] + part.offset, xs.tc[t]))
            end
        end
    end
    # The interface models of the exchanges with XT3D.
    owned = _interface_connections(parts, exchanges, conn)
    any(owned) && _interface_xt3d!(terms, owned, _interface_grid(parts, conn), setup.properties)
    tptr = Vector{Int32}(undef, njas + 1)
    tptr[1] = 1
    ta, tb, tc = Int32[], Int32[], Float64[]
    for j in 1:njas
        for (a, b, c) in terms[j]
            push!(ta, a), push!(tb, b), push!(tc, c)
        end
        tptr[j + 1] = length(tc) + 1
    end
    return xt3d_assemble(conn, tptr, ta, tb, tc, setup.r; rhs)
end

# XT3D on the interface grid `iconn` for the connections in `owned` (Xt3dInterface.f90
# `xt3d_fcpc` of the interface model: ANGLE1–3 enabled, so `nozee` is false; every cell
# active and confined).
function _interface_xt3d!(terms, owned, iconn::Connectivity, p::NpfProperties)
    ibound = ones(Int32, iconn.nodes)
    loads = Dict{Int, Any}()
    load(n) = get!(() -> _xt3d_load(iconn, p, ibound, false, n), loads, n)
    for n in 1:iconn.nodes, il0 in 1:(iconn.ia[n + 1] - iconn.ia[n] - 1)
        ii = iconn.ia[n] + il0
        m = iconn.ja[ii]
        m > n || continue
        jj = iconn.jas[ii]
        owned[jj] || continue
        c0, c1 = load(n), load(m)
        il10 = findfirst(==(n), c1.inbr)
        ar01, ar10 = _xt3d_areas(iconn, n, m, jj)
        chat01, chati0, chat1j = _xt3d_qconds(c0, il0, c1, il10, ar01, ar10, _XT3D_VCTHRESH)
        t = terms[jj]
        empty!(t)
        push!(t, (m, Int32(n), chat01))
        for il in 1:c0.nnbr
            (c0.inbr[il] == 0 || il == il0 || chati0[il] == 0.0) && continue
            push!(t, (c0.inbr[il], Int32(n), chati0[il]))
        end
        for il in 1:c1.nnbr
            (c1.inbr[il] == 0 || il == il10 || chat1j[il] == 0.0) && continue
            push!(t, (c1.inbr[il], m, -chat1j[il]))
        end
    end
    return terms
end

"""
    solution_colors(sol::SolutionSetup) -> Vector{Int32}

The colouring of the merged model for the multicolour preconditioners: each model's own
(`grid_colors`; red–black on a DIS grid), swapped per model where that makes every exchange
connection join two colours (always possible for a periodic exchange between cells of
different colour, or for models connected as a tree); otherwise `greedy_colors` of the
merged connectivity.
"""
function solution_colors(sol::SolutionSetup)
    conn = sol.conn
    cols = [grid_colors(p.conn, p.spec.dis) for p in sol.models]
    if all(c -> maximum(c; init = Int32(1)) <= 2, cols)
        nm = length(sol.models)
        flip = fill(-1, nm)
        ok = true
        for start in 1:nm
            flip[start] >= 0 && continue
            flip[start] = 0
            queue = [start]
            while ok && !isempty(queue)
                k = popfirst!(queue)
                for ex in sol.exchanges, i in eachindex(ex.node1)
                    (ex.m1 == k || ex.m2 == k) || continue
                    a = cols[ex.m1][ex.node1[i]]
                    b = cols[ex.m2][ex.node2[i]]
                    # Colours after flipping: a ⊻ flip1, b ⊻ flip2 must differ.
                    other, need = ex.m1 == k ? (ex.m2, (a == b) ⊻ (flip[k] == 1)) : (ex.m1, (a == b) ⊻ (flip[k] == 1))
                    want = need ? 1 : 0
                    if ex.m1 == ex.m2
                        a == b && (ok = false)
                    elseif flip[other] < 0
                        flip[other] = want
                        push!(queue, other)
                    elseif flip[other] != want
                        ok = false
                    end
                end
            end
        end
        if ok
            c = reduce(vcat, (flip[k] == 1 ? Int32(3) .- cols[k] : cols[k] for k in 1:nm))
            is_valid_coloring(conn.ia, conn.ja, c) && return c
        end
    end
    return greedy_colors(conn.ia, conn.ja)
end

"""
    ModelView{M}

One model of a multi-model solution, seen through the merged model `model` (a `GwfModel`):
its `index` in the solution and its `part` (`ModelPart`). The output, budget and address
code takes a `GwfModel` (a single model is its own view) or a `ModelView`; the accessors
`_vspec`, `_vconn`, `_vrange`, `_voffset`, `_vpackages` and `_vmodel` give the model's spec,
connectivity, node range and offset in the merged model, package indices and the merged model.
"""
struct ModelView{M}
    model::M
    index::Int
    part::ModelPart
end
