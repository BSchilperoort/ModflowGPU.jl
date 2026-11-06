# Hardware baseline (T0.4): stream triad, CSR SpMV, dot and atomics on one backend.
#
#     julia --project=benchmark benchmark/hardware.jl rocm
#     JULIA_KA_CPU_THREADS=8 julia -t 8 --project=benchmark benchmark/hardware.jl cpu
#
# The script loads AMDGPU (for `allowscalar` and the ISA check), so AMDGPU must be installed
# even for the `cpu` run; the `benchmark/` workspace project provides it.
#
# Optional second argument `quick` uses small sizes (smoke test). Results are printed as
# Markdown tables and also written to `benchmark/out/hardware_<backend>.md`; the numbers
# worth keeping are copied into `benchmark/results/hardware.md`.
#
# All kernels are generic over the KernelAbstractions backend. Timing: warm-up calls,
# then `nrep` repetitions, each bracketed by `KernelAbstractions.synchronize`; median and
# minimum are reported. Bandwidths use 1 GB = 1e9 bytes.

using AMDGPU
using Atomix: Atomix
using KernelAbstractions
using KernelAbstractions: @atomic
using ModflowGPU
using Printf
using Statistics: median

const KA = KernelAbstractions

# ---------------------------------------------------------------------------------------
# Timing and output helpers

"""
    bench(f, backend; nwarm = 3, nrep = 20)

Time `f()` `nrep` times after `nwarm` warm-up calls, synchronising the backend before
and after each call. Returns `(median, minimum)` in seconds.
"""
function bench(f, backend; nwarm = 3, nrep = 20)
    for _ in 1:nwarm
        f()
    end
    KA.synchronize(backend)
    times = Float64[]
    for _ in 1:nrep
        KA.synchronize(backend)
        t0 = time_ns()
        f()
        KA.synchronize(backend)
        push!(times, (time_ns() - t0) * 1.0e-9)
    end
    return median(times), minimum(times)
end

const OUT = IOBuffer()

# Print to stdout and keep a copy for the output file.
function out(s = "")
    println(s)
    println(OUT, s)
    return flush(stdout)
end

table_header(cols...) = (out("| " * join(cols, " | ") * " |"); out("|" * repeat(" --- |", length(cols))))
table_row(cols...) = out("| " * join(cols, " | ") * " |")

gbs(bytes, t) = @sprintf("%.1f", bytes / t / 1.0e9)
ms(t) = @sprintf("%.3f", t * 1.0e3)
us(t) = @sprintf("%.1f", t * 1.0e6)
mb(bytes) = @sprintf("%.0f", bytes / 2^20)

# ---------------------------------------------------------------------------------------
# 1. Stream triad: a = b + s * c

@kernel function triad_kernel!(a, @Const(b), @Const(c), s)
    i = @index(Global)
    @inbounds a[i] = b[i] + s * c[i]
end

function triad!(a, b, c, s)
    backend = get_backend(a)
    triad_kernel!(backend)(a, b, c, s; ndrange = length(a))
    return nothing
end

function bench_triad(backend, bytes_per_array)
    out("## Stream triad `a = b + s*c`")
    out()
    out("Bytes moved: 3 × array size (no write-allocate). Broadcast is `a .= b .+ s .* c`.")
    out()
    table_header("T", "MB per array", "variant", "median ms", "min ms", "GB/s (median)", "GB/s (best)")
    for T in (Float32, Float64)
        n = bytes_per_array ÷ sizeof(T)
        a = KA.zeros(backend, T, n)
        b = KA.allocate(backend, T, n)
        c = KA.allocate(backend, T, n)
        fill!(b, one(T))
        fill!(c, T(2))
        s = T(3)
        bytes = 3 * n * sizeof(T)
        for (name, f) in (
                "KA kernel" => () -> triad!(a, b, c, s),
                "broadcast" => () -> (a .= b .+ s .* c; nothing),
            )
            tmed, tmin = bench(f, backend)
            table_row(T, mb(n * sizeof(T)), name, ms(tmed), ms(tmin), gbs(bytes, tmed), gbs(bytes, tmin))
        end
        # Check one result on the host (no scalar indexing on the device).
        @assert all(==(T(7)), Array(a))
    end
    return out()
end

# ---------------------------------------------------------------------------------------
# 2. CSR SpMV, scalar CSR (one work-item per row), MF6 layout (diagonal first in each row)

"""
    laplacian7(nx, ny, nz, T)

Host CSR (`ia`, `ja` as `Int32`, values `a::Vector{T}`) of the 3-D 7-point Laplacian on an
`nx × ny × nz` grid in MF6 layout: diagonal first in each row, then the off-diagonal
columns in increasing order, negative diagonal (MF6 sign convention).
"""
function laplacian7(nx, ny, nz, ::Type{T}) where {T}
    n = nx * ny * nz
    lin(i, j, k) = i + (j - 1) * nx + (k - 1) * nx * ny
    ia = Vector{Int32}(undef, n + 1)
    ja = Int32[]
    a = T[]
    sizehint!(ja, 7n)
    sizehint!(a, 7n)
    ia[1] = 1
    for k in 1:nz, j in 1:ny, i in 1:nx
        row = lin(i, j, k)
        push!(ja, row)
        push!(a, zero(T))
        idiag = length(a)
        # Neighbours in increasing column order.
        for (di, dj, dk) in ((0, 0, -1), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
            ii, jj, kk = i + di, j + dj, k + dk
            if 1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz
                push!(ja, lin(ii, jj, kk))
                push!(a, one(T))
                a[idiag] -= one(T)
            end
        end
        a[idiag] -= T(0.01) # strictly diagonally dominant, like a model with storage
        ia[row + 1] = length(ja) + 1
    end
    return ia, ja, a
end

@kernel function spmv_csr_kernel!(y, @Const(ia), @Const(ja), @Const(a), @Const(x))
    i = @index(Global)
    T = eltype(y)
    acc = zero(T)
    @inbounds for k in ia[i]:(ia[i + 1] - one(eltype(ia)))
        acc += a[k] * x[ja[k]]
    end
    @inbounds y[i] = acc
end

"""
    spmv!(y, ia, ja, a, x)

`y = A x` for CSR `A` with one work-item per row (scalar CSR).
"""
function spmv!(y, ia, ja, a, x)
    backend = get_backend(y)
    spmv_csr_kernel!(backend)(y, ia, ja, a, x; ndrange = length(y))
    return nothing
end

# Host reference SpMV in Float64.
function spmv_host(ia, ja, a, x)
    y = zeros(Float64, length(ia) - 1)
    for i in eachindex(y), k in ia[i]:(ia[i + 1] - 1)
        y[i] += Float64(a[k]) * Float64(x[ja[k]])
    end
    return y
end

# Byte models for SpMV with n rows, nnz non-zeros, value type T, Int32 indices:
# - "min": values + col indices + row pointers + x read once + y written once
#   (compulsory traffic, x reuse perfectly cached).
# - "gather": as "min" but x read once per non-zero (no reuse).
spmv_bytes_min(n, nnz, T) = nnz * (sizeof(T) + 4) + (n + 1) * 4 + n * sizeof(T) + n * sizeof(T)
spmv_bytes_gather(n, nnz, T) = nnz * (sizeof(T) + 4) + (n + 1) * 4 + nnz * sizeof(T) + n * sizeof(T)

function bench_spmv(backend, grids)
    out("## CSR SpMV, 3-D 7-point Laplacian, scalar CSR (one work-item per row)")
    out()
    out("MF6 layout (diagonal first), Int32 `ia`/`ja`. Byte models: *min* = values + `ja` + `ia` +")
    out("x once + y once; *gather* = same with x read once per non-zero. GB/s use the median time.")
    out()
    table_header(
        "grid", "rows", "nnz", "T", "median ms", "min ms", "GB/s (min model)",
        "GB/s (gather model)", "max rel err",
    )
    for (nx, ny, nz) in grids
        for T in (Float64, Float32)
            ia_h, ja_h, a_h = laplacian7(nx, ny, nz, T)
            n = length(ia_h) - 1
            nnz = length(ja_h)
            x_h = T[T(1) + T(mod(i, 17)) / T(16) for i in 1:n]
            ia = KA.allocate(backend, Int32, n + 1)
            ja = KA.allocate(backend, Int32, nnz)
            a = KA.allocate(backend, T, nnz)
            x = KA.allocate(backend, T, n)
            y = KA.zeros(backend, T, n)
            copyto!(ia, ia_h)
            copyto!(ja, ja_h)
            copyto!(a, a_h)
            copyto!(x, x_h)
            tmed, tmin = bench(() -> spmv!(y, ia, ja, a, x), backend; nrep = 50)
            yref = spmv_host(ia_h, ja_h, a_h, x_h)
            err = maximum(abs.(Array(y) .- yref)) / maximum(abs, yref)
            table_row(
                "$(nx)×$(ny)×$(nz)", n, nnz, T, ms(tmed), ms(tmin),
                gbs(spmv_bytes_min(n, nnz, T), tmed), gbs(spmv_bytes_gather(n, nnz, T), tmed),
                @sprintf("%.1e", err),
            )
            ia = ja = a = x = y = nothing
            GC.gc()
        end
    end
    return out()
end

# ---------------------------------------------------------------------------------------
# 3. dot via mapreduce

# Product accumulated in Float64 (a named function: a closure capturing a type is not isbits).
mul_f64(p, q) = Float64(p) * Float64(q)

function bench_dot(backend, sizes)
    out("## `dot` via `mapreduce` on device arrays")
    out()
    out("`mapreduce(*, +, x, y)`; the Float32 → Float64 row is")
    out("`mapreduce(mul_f64, +, x, y)` with `mul_f64(a, b) = Float64(a) * Float64(b)`. Time per call includes")
    out("returning the scalar to the host. Bytes = 2 × n × sizeof(T).")
    out()
    table_header("n", "data", "accumulate", "median µs", "min µs", "GB/s (median)", "rel err")
    for n in sizes
        for (T, Tacc) in ((Float32, Float32), (Float32, Float64), (Float64, Float64))
            x_h = T[T(1) + T(mod(i, 7)) / T(8) for i in 1:n]
            y_h = T[T(1) - T(mod(i, 5)) / T(8) for i in 1:n]
            x = KA.allocate(backend, T, n)
            y = KA.allocate(backend, T, n)
            copyto!(x, x_h)
            copyto!(y, y_h)
            f = Tacc === T ? () -> mapreduce(*, +, x, y) : () -> mapreduce(mul_f64, +, x, y)
            r = f()
            exact = sum(Float64(p) * Float64(q) for (p, q) in zip(x_h, y_h))
            tmed, tmin = bench(f, backend; nrep = 50)
            table_row(
                n, T, Tacc, us(tmed), us(tmin), gbs(2 * n * sizeof(T), tmed),
                @sprintf("%.1e", abs(r - exact) / abs(exact)),
            )
        end
    end
    return out()
end

# ---------------------------------------------------------------------------------------
# 4. Atomic add

@kernel function atomic_add_kernel!(y, @Const(idx), @Const(v))
    i = @index(Global)
    @inbounds @atomic y[idx[i]] += v[i]
end

@kernel function plain_add_kernel!(y, @Const(idx), @Const(v))
    i = @index(Global)
    @inbounds y[idx[i]] += v[i]
end

function scatter_add!(y, idx, v; atomic::Bool)
    backend = get_backend(y)
    kernel! = atomic ? atomic_add_kernel!(backend) : plain_add_kernel!(backend)
    kernel!(y, idx, v; ndrange = length(idx))
    return nothing
end

# Index patterns for `n` updates: name => (targets, idx).
function atomic_patterns(n)
    return [
        # Many collisions: every update goes to one of 16 addresses.
        "16 targets (heavy contention)" => (16, Int32[mod(i - 1, 16) + 1 for i in 1:n]),
        # Few collisions: ~8 updates per target, scattered by a multiplicative hash.
        "n/8 targets, hashed (≈8 per target)" =>
            (n ÷ 8, Int32[mod((i - 1) * 2654435761, n ÷ 8) + 1 for i in 1:n]),
        # No collisions, coalesced.
        "n targets, idx = i (none)" => (n, Int32.(1:n)),
    ]
end

function bench_atomics(backend, n)
    out("## Atomic add `Atomix.@atomic y[idx[i]] += v[i]`")
    out()
    out("n = $n updates per launch, v[i] = 0.5 (sums are exact, so the check is exact).")
    out("*plain* = same kernel with a non-atomic `+=` (racy when targets collide; timing only).")
    out()
    table_header("pattern", "T", "kernel", "median ms", "min ms", "G updates/s", "correct")
    for (pname, (m, idx_h)) in atomic_patterns(n)
        idx = KA.allocate(backend, Int32, n)
        copyto!(idx, idx_h)
        expected = zeros(Float64, m)
        for j in idx_h
            expected[j] += 0.5
        end
        for T in (Float32, Float64)
            v = KA.allocate(backend, T, n)
            fill!(v, T(0.5))
            y = KA.zeros(backend, T, m)
            for atomic in (true, false)
                # Correctness: one launch onto zeros.
                fill!(y, zero(T))
                scatter_add!(y, idx, v; atomic)
                KA.synchronize(backend)
                ok = Array(y) == T.(expected)
                tmed, tmin = bench(() -> scatter_add!(y, idx, v; atomic), backend)
                table_row(
                    pname, T, atomic ? "atomic" : "plain", ms(tmed), ms(tmin),
                    @sprintf("%.2f", n / tmed / 1.0e9), ok ? "yes" : "**no**",
                )
            end
        end
    end
    return out()
end

# Instructions used for the Float64 atomic add on AMDGPU (hardware instruction or a
# compare-and-swap loop).
function atomic_isa_note(backend)
    backend isa AMDGPU.ROCBackend || return nothing
    for T in (Float32, Float64)
        y = KA.zeros(backend, T, 16)
        idx = KA.ones(backend, Int32, 64)
        v = KA.ones(backend, T, 64)
        io = IOBuffer()
        AMDGPU.@device_code_gcn io = io scatter_add!(y, idx, v; atomic = true)
        isa = String(take!(io))
        instrs = unique(m.match for m in eachmatch(r"\b(?:global|flat|buffer|ds)_atomic_\w+", isa))
        out("- $T atomic add, gfx ISA atomic instructions: `$(join(instrs, "`, `"))`")
    end
    return out()
end

# ---------------------------------------------------------------------------------------
# Machine info

function cpu_model()
    for l in eachline("/proc/cpuinfo")
        startswith(l, "model name") && return strip(split(l, ':'; limit = 2)[2])
    end
    return "unknown"
end

function machine_info(backend)
    out("## Machine")
    out()
    out("- Date: $(read(`date -I`, String) |> strip)")
    atype = nameof(typeof(KA.allocate(backend, Float32, 1)))
    out("- Backend: `$(ModflowGPU.backend_name(backend))`, arrays from `KA.allocate` are `$atype`")
    out("- Julia $(VERSION); KernelAbstractions $(pkgversion(KernelAbstractions)); AMDGPU $(pkgversion(AMDGPU)); Atomix $(pkgversion(Atomix))")
    out("- CPU: $(cpu_model()); `nproc` = $(Sys.CPU_THREADS); `Threads.nthreads()` = $(Threads.nthreads()); `JULIA_KA_CPU_THREADS` = $(get(ENV, "JULIA_KA_CPU_THREADS", "unset"))")
    if backend isa AMDGPU.ROCBackend
        dev = AMDGPU.device()
        out("- GPU: $(AMDGPU.HIP.name(dev)) ($(AMDGPU.HIP.gcn_arch(dev))), wavefront $(AMDGPU.HIP.wavefrontsize(dev)); ROCm $(AMDGPU.HIP.runtime_version())")
    end
    return out()
end

# ---------------------------------------------------------------------------------------

function main(args)
    name = isempty(args) ? "rocm" : args[1]
    quick = "quick" in args
    backend = ModflowGPU.select_backend(name)
    AMDGPU.allowscalar(false)

    triad_bytes = quick ? 2^24 : 2^28                   # 256 MB per array
    grids = quick ? [(32, 32, 32)] : [(100, 100, 100), (216, 216, 216)]
    dot_sizes = quick ? [10^5] : [10^6, 10^7, 2^25]
    atomic_n = quick ? 2^16 : 2^24

    out("# Hardware baseline: $name$(quick ? " (quick)" : "")")
    out()
    machine_info(backend)
    bench_triad(backend, triad_bytes)
    bench_spmv(backend, grids)
    bench_dot(backend, dot_sizes)
    bench_atomics(backend, atomic_n)
    atomic_isa_note(backend)

    mkpath(joinpath(@__DIR__, "out"))
    file = joinpath(@__DIR__, "out", "hardware_$(name)$(quick ? "_quick" : "").md")
    write(file, take!(OUT))
    println("Written to $file")
    return nothing
end

main(ARGS)
