# Initial conditions (gwf-ic.f90, `ic_ar`: x(n) = strt(n) for every reduced node).
#
# The device stores heads as offsets from the per-cell references `r` (PLAN.md section 3.7,
# rule 1), so the starting head is converted to offsets on the host in Float64 and cast to
# the engine precision once.

"""
    initial_heads(ic::IcSpec, conn::Connectivity, r) -> Vector{Float64}
    initial_heads(backend, T, ic::IcSpec, conn::Connectivity, r) -> device vector of T

The starting heads (`ic_ar`) per reduced node as offsets from the cell references `r`
(`reference_heads`), `strt[nodeuser[n]] − r[n]`, computed in Float64 (`to_offsets`). The
second form casts to `T` and copies to a new array on `backend`.
"""
function initial_heads(ic::IcSpec, conn::Connectivity, r::AbstractVector{Float64})
    length(ic.strt) == conn.nodesuser ||
        throw(ArgumentError("initial_heads: length(strt) = $(length(ic.strt)), expected $(conn.nodesuser) user nodes"))
    return to_offsets(r, view(ic.strt, conn.nodeuser))
end

function initial_heads(backend::Backend, ::Type{T}, ic::IcSpec, conn::Connectivity, r::AbstractVector{Float64}) where {T <: AbstractFloat}
    h = KernelAbstractions.allocate(backend, T, conn.nodes)
    copyto!(h, T.(initial_heads(ic, conn, r)))
    return h
end
