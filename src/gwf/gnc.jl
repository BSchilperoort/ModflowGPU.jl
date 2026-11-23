# GNC6: ghost-node correction (T6.13), within a model and on GWF-GWF exchanges.
#
# Ports MF6 6.8.2 GhostNode.f90: `gnc_df`/`read_data` (the records, reduced nodes),
# `gnc_ac` (implicit: the entries n–j and m–j added to the sparse structure), `gnc_mc`
# ("GHOST NODE ERROR ... is not connected to cell ..."), `gnc_fmsav` (the n–m conductance
# taken from the matrix after NPF), `gnc_fc` (implicit: matrix terms; explicit: right-hand
# side with the current heads), `gnc_cq`/`deltaQgnc` (the correction added to FLOW-JA-FACE);
# gwf.f90 (`gwf_fc`: GNC after NPF and HFB), gwf-npf.f90 `npf_df` (XT3D with GNC is an
# error), exg-gwfgwf.f90 `gwf_gwf_ac`/`_mc`/`_fc` (the exchange conductance as `cond`),
# `read_gnc` (the records match the exchange connections; EXPLICIT with NEWTON is an
# error), `gwf_gwf_calc_simvals` (the exchange flow includes `deltaQgnc`) and
# GwfGwfConnection.f90 (GNC with XT3D on an exchange is an error).
#
# The correction. For a record (n, m, j_1..j_k, α_1..α_k) the flow into n from m is
# computed with a ghost node in n, h_g = (1 − Σα)·h_n + Σ α_j·h_j:
#
#     Q = cond·(h_m − h_g) = cond·(h_m − h_n) + Σ_j cond·α_j·(h_n − h_j),
#
# so each contributing cell adds a flux term `c·(h_a − h_b)` with c = cond·α_j to the
# connection n–m. These are the flux terms of the XT3D formulation (`Xt3dSetup`, T6.5):
# the primary term `condsat·(h_m − h_n)` first, the GNC terms after it. A model with GNC
# runs on the XT3D kernels (`xt3d_formulate_kernel!`, `xt3d_flowja_kernel!`): flux-form
# residual in offset heads (`tdr = r_a − r_b`, PLAN.md 3.7 rules 1–2), FLOW-JA-FACE with
# `deltaQgnc` included (`gnc_cq`). Implicit GNC puts the terms into the matrix: the system
# structure is MF6's connectivity plus the entries of `gnc_ac` (`structure = :terms` of
# `xt3d_assemble`, not XT3D's neighbours of neighbours), on which the solver, the
# preconditioners and their colouring (greedy, as for XT3D) work. EXPLICIT GNC is the RHS
# option: only the primary terms in the matrix, the GNC terms on the right-hand side at
# the current heads (`gnc_fc`: `rhs(n) −= c·(x_n − x_j)`, `rhs(m) += ...`); with correction
# solves this is the same iteration as MF6's.
#
# `cond` is the conductance MF6 finds in the matrix at (n, m) after NPF (and HFB, BUY):
# for confined cells the saturated conductance, constant, so the terms are computed once on
# the host in Float64. T6.14: with convertible cells, NEWTON (`gnc_fn`), HFB6 or TVK6 the
# conductance changes, and the model runs on the head-dependent path of
# src/gwf/xt3d_var.jl (`gnc_var_setup`). Not supported (`MF6UnsupportedError`): GNC6 and
# XT3D in one solution. PRINT_INPUT and PRINT_FLOWS (the `gnc_ot` table) write nothing.

"""
    check_gnc(spec::GwfSpec, icelltype)

The checks of a model's GNC6 package: XT3D with GNC is MF6's error (`npf_df`,
`MF6InputError`). T6.14: NEWTON, convertible cells, HFB6 and TVK6 with GNC6 are supported
(src/gwf/xt3d_var.jl).
"""
function check_gnc(spec::GwfSpec, icelltype::AbstractVector{<:Integer})
    g = spec.gnc
    isnothing(g) && return nothing
    file = isempty(g.file) ? "GNC" : g.file
    spec.npf.xt3d && throw(
        MF6InputError(
            isempty(spec.npf.file) ? "NPF" : spec.npf.file, 0,
            "Error in model $(spec.name).  The XT3D option cannot be used with the GNC Package.",
        ),
    )
    return nothing
end

# GhostNode.f90 `nodeu_to_noder`: the records' user nodes as reduced nodes (the loader
# checked IDOMAIN > 0), contributing cells 0 where none (also a contributing cell with
# IDOMAIN 0, reduced node 0, as in MF6).
function _gnc_reduced(g::GncSpec, conn1::Connectivity, conn2::Connectivity)
    n = Int32[conn1.nodereduced[u] for u in g.noden]
    m = Int32[conn2.nodereduced[u] for u in g.nodem]
    j = Int32[u == 0 ? Int32(0) : conn1.nodereduced[u] for u in g.nodesj]
    return n, m, reshape(j, size(g.nodesj))
end

# Standard conductance: one term condsat·(h_m − h_n) per symmetric connection (n < m).
function _conductance_terms(conn::Connectivity, condsat::AbstractVector{Float64})
    terms = [Tuple{Int32, Int32, Float64}[] for _ in 1:conn.njas]
    for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
        m = conn.ja[ii]
        m > n || continue
        push!(terms[conn.jas[ii]], (m, Int32(n), condsat[conn.jas[ii]]))
    end
    return terms
end

# The flux terms per symmetric connection as the `tptr`, `ta`, `tb`, `tc` of `Xt3dSetup`.
function _flatten_terms(terms)
    tptr = Vector{Int32}(undef, length(terms) + 1)
    tptr[1] = 1
    ta, tb, tc = Int32[], Int32[], Float64[]
    for j in eachindex(terms)
        for (a, b, c) in terms[j]
            push!(ta, a), push!(tb, b), push!(tc, c)
        end
        tptr[j + 1] = length(tc) + 1
    end
    return tptr, ta, tb, tc
end

"""
    gnc_terms!(terms, conn, condsat, n, m, nodesj, alphasj; file, model1, model2, user1, user2) -> terms

Add the GNC flux terms of the records (`n[k]`, `m[k]`, contributing cells `nodesj[:, k]`
(0: none) with weights `alphasj[:, k]`; nodes of `conn`) to `terms` (per symmetric
connection of `conn`, flow into the lower-numbered node; `gnc_fc`): `c·(h_n − h_j)` into
`n` with `c = condsat·α_j` of the connection n–m. A record whose cells are not connected is
MF6's error (`gnc_mc`, an `MF6InputError` naming the reduced nodes `user1[k]`, `user2[k]`
of `model1`, `model2`).
"""
function gnc_terms!(
        terms, conn::Connectivity, condsat::AbstractVector{Float64}, n, m, nodesj, alphasj;
        file = "GNC", model1 = "", model2 = model1, user1 = n, user2 = m,
    )
    for k in eachindex(n)
        nn, mm = n[k], m[k]
        ii = getjaindex(conn, nn, mm)
        (ii > 0 && nn != mm) || throw(
            MF6InputError(file, 0, "GHOST NODE ERROR.  Cell $(user1[k]) in model $model1 is not connected to cell $(user2[k]) in model $model2"),
        )
        jj = conn.jas[ii]
        cond = condsat[jj]
        for i in axes(nodesj, 1)
            j = nodesj[i, k]
            alpha = alphasj[i, k]
            (j == 0 || alpha == 0.0) && continue
            c = alpha * cond
            # Flow into n: c·(h_n − h_j); into the other cell its negative.
            push!(terms[jj], nn < mm ? (Int32(nn), Int32(j), c) : (Int32(j), Int32(nn), c))
        end
    end
    return terms
end

"""
    gnc_setup(spec::GwfSpec, conn::Connectivity, setup::NpfSetup) -> Union{Nothing, Xt3dSetup}

The flux terms of a model with GNC6 (`nothing` without): the conductance of every
connection and the GNC terms (`gnc_terms!`), assembled with the GNC structure (implicit)
or as the RHS option (EXPLICIT). Runs `check_gnc`.
"""
function gnc_setup(spec::GwfSpec, conn::Connectivity, setup::NpfSetup)
    g = spec.gnc
    isnothing(g) && return nothing
    check_gnc(spec, setup.properties.icelltype)
    n, m, j = _gnc_reduced(g, conn, conn)
    terms = _conductance_terms(conn, setup.condsat)
    gnc_terms!(terms, conn, setup.condsat, n, m, j, g.alphasj; file = isempty(g.file) ? "GNC" : g.file, model1 = spec.name)
    tptr, ta, tb, tc = _flatten_terms(terms)
    return xt3d_assemble(conn, tptr, ta, tb, tc, setup.r; rhs = g.explicit, structure = :terms)
end
