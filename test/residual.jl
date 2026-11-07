# Helpers for the residual tests (PLAN.md section 4) until the assembly of T2.6 exists.

"Copy of `x` on TEST_BACKEND."
function to_test_device(x::AbstractArray)
    d = KernelAbstractions.allocate(TEST_BACKEND, eltype(x), size(x))
    copyto!(d, x)
    return d
end

"""
    flux_residual(T, conn, condsat, Δr, h, diag, b, ib) -> (rel, roundoff)

Flux-form residual (PLAN.md section 3.7, rule 2) on the host in `T`: per row `n` with
`ib[n] > 0`,

    res_n = Σ_m condsat_nm·(h'_m − h'_n + Δr_nm) + diag_n·h'_n − b_n

with the boundary and storage terms in `diag` and `b` (host arrays). Returns
`max |res| / max Σ|terms|` and the bound from rounding the offset heads to `T`,
`eps(T)·max_n (Σ C·(|h'_m| + |h'_n| + |Δr|) + |diag_n·h'_n| + |b_n|) / max Σ|terms|`.
"""
function flux_residual(::Type{T}, conn, condsat, Δr, h, diag, b, ib) where {T}
    maxres = zero(T)
    maxscale = zero(T)
    maxround = zero(T)
    for n in 1:conn.nodes
        ib[n] > 0 || continue
        res = zero(T)
        scale = zero(T)
        round = zero(T)
        for ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
            m = conn.ja[ii]
            d = conn.ja[ii] > n ? Δr[conn.jas[ii]] : -Δr[conn.jas[ii]]
            t = condsat[conn.jas[ii]] * (h[m] - h[n] + d)
            res += t
            scale += abs(t)
            round += condsat[conn.jas[ii]] * (abs(h[m]) + abs(h[n]) + abs(d))
        end
        t = diag[n] * h[n] - b[n]
        res += t
        scale += abs(diag[n] * h[n]) + abs(b[n])
        round += abs(diag[n] * h[n]) + abs(b[n])
        maxres = max(maxres, abs(res))
        maxscale = max(maxscale, scale)
        maxround = max(maxround, round)
    end
    return maxres / maxscale, eps(T) * maxround / maxscale
end
