# Reference heads and per-connection reference differences (PLAN.md section 3.7, rule 1).
#
# The device stores heads as offsets `h'_n = h_n − r_n` from a per-cell reference
# `r_n = clamp(STRT_n, bot_n, top_n)`, computed here on the host in Float64. Connection
# terms then read `C_nm·(h'_m − h'_n + Δr_nm)` with `Δr_nm = r_m − r_n`.
#
# Storage convention (PLAN.md T2.2, like `cl1`/`cl2` in src/discretization/connectivity.jl):
# `Δr` is stored once per symmetric connection (`njas`) and indexed through `jas`;
# `Δr[jas]` is `r[m] − r[n]` with `n < m`. Gather rule for off-diagonal entry `ii` of row
# `n` (neighbour `m = ja[ii]`): `+Δr[jas[ii]]` when `m > n`, `−Δr[jas[ii]]` otherwise
# (`reference_difference`).

"""
    reference_heads(conn::Connectivity, strt) -> Vector{Float64}

Per reduced node, the reference head `r_n = clamp(STRT_n, bot_n, top_n)` (PLAN.md section
3.7, rule 1) in Float64. `strt` is the IC6 starting head by user node (`IcSpec.strt`).
"""
function reference_heads(conn::Connectivity, strt::AbstractVector{<:Real})
    length(strt) == conn.nodesuser ||
        throw(ArgumentError("reference_heads: length(strt) = $(length(strt)), expected $(conn.nodesuser) user nodes"))
    r = Vector{Float64}(undef, conn.nodes)
    for n in 1:conn.nodes
        r[n] = clamp(Float64(strt[conn.nodeuser[n]]), conn.bot[n], conn.top[n])
    end
    return r
end

"""
    reference_differences(conn::Connectivity, r) -> Vector{Float64}

Per symmetric connection (`njas`), `Δr[jas] = r[m] − r[n]` with `n < m`, in Float64. Use
`reference_difference` to read it from a CSR row.
"""
function reference_differences(conn::Connectivity, r::AbstractVector{Float64})
    length(r) == conn.nodes || throw(ArgumentError("reference_differences: length(r) = $(length(r)), expected $(conn.nodes)"))
    Δr = Vector{Float64}(undef, conn.njas)
    for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
        m = conn.ja[ii]
        m > n && (Δr[conn.jas[ii]] = r[m] - r[n])
    end
    return Δr
end

"""
    reference_difference(Δr, conn::Connectivity, n, ii) -> eltype(Δr)

`r[m] − r[n]` for off-diagonal entry `ii` of row `n` (`m = ja[ii]`), gathered from the
per-symmetric-connection array `Δr`: `+Δr[jas[ii]]` when `m > n`, `−Δr[jas[ii]]`
otherwise.
"""
@inline function reference_difference(Δr::AbstractVector, conn::Connectivity, n::Integer, ii::Integer)
    d = Δr[conn.jas[ii]]
    return conn.ja[ii] > n ? d : -d
end

"""
    to_offsets(r, values) -> Vector{Float64}
    to_offsets(r, values, nodes) -> Vector{Float64}

Head-like values as offsets from the cell references (PLAN.md section 3.7, rule 1), in
Float64; cast to the engine precision `T` only afterwards (`T.(to_offsets(r, v))`), so
each value is rounded once.

- `to_offsets(r, values)`: one value per reduced node, `values[n] − r[n]` (top, bottom,
  initial and old heads).
- `to_offsets(r, values, nodes)`: one value per record with reduced node `nodes[i]`,
  `values[i] − r[nodes[i]]` (boundary heads, stages, elevations). Records without a node
  (`nodes[i] <= 0`) keep their value.
"""
function to_offsets(r::AbstractVector{Float64}, values::AbstractVector{<:Real})
    length(values) == length(r) || throw(ArgumentError("to_offsets: length(values) = $(length(values)), expected $(length(r))"))
    return Float64[Float64(values[n]) - r[n] for n in eachindex(r, values)]
end

function to_offsets(r::AbstractVector{Float64}, values::AbstractVector{<:Real}, nodes::AbstractVector{<:Integer})
    length(values) == length(nodes) || throw(ArgumentError("to_offsets: $(length(values)) values for $(length(nodes)) nodes"))
    return Float64[nodes[i] > 0 ? Float64(values[i]) - r[nodes[i]] : Float64(values[i]) for i in eachindex(values, nodes)]
end
