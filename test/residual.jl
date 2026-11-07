# Helpers for the residual tests (PLAN.md section 4). `flux_residual` runs the device
# residual kernel of src/solver/csr.jl (T2.6); `host_flux_residual` is the host loop it
# replaced (T2.3), kept to validate the kernel (test/parity/assembly_test.jl).

"Copy of `x` on TEST_BACKEND."
function to_test_device(x::AbstractArray)
    d = KernelAbstractions.allocate(TEST_BACKEND, eltype(x), size(x))
    copyto!(d, x)
    return d
end

"""
    host_flux_residual(T, conn, condsat, Δr, h, diag, b, ib) -> (rel, roundoff)

Flux-form residual (PLAN.md section 3.7, rule 2) on the host in `T`: per row `n` with
`ib[n] > 0`,

    res_n = Σ_m condsat_nm·(h'_m − h'_n + Δr_nm) + diag_n·h'_n − b_n

with the boundary and storage terms in `diag` and `b` (host arrays). Returns
`max |res| / max Σ|terms|` and the bound from rounding the offset heads to `T`,
`eps(T)·max_n (Σ C·(|h'_m| + |h'_n| + |Δr|) + |diag_n·h'_n| + |b_n|) / max Σ|terms|`.
"""
function host_flux_residual(::Type{T}, conn, condsat, Δr, h, diag, b, ib) where {T}
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

"""
    residual_stats(T, res, scale, bound) -> (rel, roundoff)

`max |res| / max scale` and `eps(T)·max bound / max scale` from the per-row outputs of
`ModflowGPU.flux_residual!` (copied to the host; max norms need no accumulation).
"""
function residual_stats(::Type{T}, res, scale, bound) where {T}
    maxscale = Float64(maximum(Array(scale)))
    return Float64(maximum(abs, Array(res))) / maxscale, eps(T) * Float64(maximum(Array(bound))) / maxscale
end

"""
    flux_residual(T, conn, condsat, Δr, h, diag, b, ib) -> (rel, roundoff)

As `host_flux_residual`, with the device kernel `ModflowGPU.flux_residual!` on
TEST_BACKEND: `condsat`, `Δr` (`T`, per symmetric connection), `h`, `diag`, `b` (`T`) and
`ib` are host or device arrays.
"""
function flux_residual(::Type{T}, conn, condsat, Δr, h, diag, b, ib) where {T}
    csr = ModflowGPU.CsrStructure(TEST_BACKEND, conn)
    npf = ModflowGPU.NpfModel(to_test_device(T.(Array(condsat))), to_test_device(T.(Array(Δr))))
    sys = ModflowGPU.LinearSystem(TEST_BACKEND, T, csr)
    copyto!(sys.bdiag, T.(Array(diag)))
    copyto!(sys.brhs, T.(Array(b)))
    res, scale, bound = (KernelAbstractions.zeros(TEST_BACKEND, T, conn.nodes) for _ in 1:3)
    ModflowGPU.flux_residual!(res, sys, csr, npf, to_test_device(T.(Array(h))), to_test_device(Int32.(Array(ib))); scale, bound)
    KernelAbstractions.synchronize(TEST_BACKEND)
    return residual_stats(T, res, scale, bound)
end
