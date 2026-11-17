# Hardware baseline (T0.4)

Measured 2026-11-06 with `benchmark/hardware.jl` (it loads AMDGPU, so AMDGPU must be
installed even for the `cpu` run; the `benchmark/` project provides it):

```sh
julia --project=benchmark benchmark/hardware.jl rocm
JULIA_KA_CPU_THREADS=8 julia -t 8 --project=benchmark benchmark/hardware.jl cpu
```

Timing: 3 warm-up calls, then 20 repetitions (50 for SpMV and dot), each bracketed by
`KernelAbstractions.synchronize`. Tables give the median and the minimum; GB/s use the
median unless marked "best". 1 GB = 1e9 bytes. `AMDGPU.allowscalar(false)` throughout.
The rocm run was repeated: medians agreed within 1% (SpMV 1 M Float32 within 8%).

## Machine

| | |
| --- | --- |
| GPU | AMD Radeon RX 7800 XT, gfx1101, 16 GB GDDR6, wavefront 32, 64 MB Infinity Cache; spec peak bandwidth **624 GB/s** (19.5 Gbps × 256 bit) |
| ROCm | 7.2.2 (`/opt/rocm`; HIP runtime reports 7.16.26385) |
| CPU | AMD Ryzen 7 7700 (8 cores / 16 threads, `nproc` = 16), 32 MB L3, dual-channel DDR5 (DIMM speed not queried) |
| Julia | 1.12.6 |
| Packages (default resolve) | KernelAbstractions 0.10.5, AMDGPU 2.11.0, Atomix 1.7.0 |
| CPU backend | KA 0.10: `CPU()` = `POCLBackend`, `JULIA_KA_CPU_THREADS=8`, `julia -t 8`; arrays from `KA.allocate` are `Array` |

The script also runs on KA 0.9.45 + AMDGPU 2.8.0 (the mGV combination, see
`docs/dev/compat.md`). The native KA 0.9 CPU backend (`-t 8`) gives the same triad and SpMV
numbers as POCL within 6%; only the broadcast triad is slower (26 GB/s, single-threaded).

## 1. Stream triad `a = b + s*c`, 256 MB per array

Bytes = 3 × array size (no write-allocate counted).

| backend | T | variant | median ms | min ms | GB/s | GB/s (best) |
| --- | --- | --- | --- | --- | --- | --- |
| rocm | Float32 | KA kernel | 1.812 | 1.598 | **445** | 504 |
| rocm | Float32 | broadcast | 1.813 | 1.608 | 444 | 501 |
| rocm | Float64 | KA kernel | 1.727 | 1.523 | **466** | 529 |
| rocm | Float64 | broadcast | 1.725 | 1.512 | 467 | 532 |
| cpu | Float32 | KA kernel | 20.22 | 20.07 | **39.8** | 40.1 |
| cpu | Float32 | broadcast | 23.26 | 22.72 | 34.6 | 35.4 |
| cpu | Float64 | KA kernel | 20.31 | 20.12 | **39.7** | 40.0 |
| cpu | Float64 | broadcast | 22.91 | 22.46 | 35.1 | 35.9 |

GPU: 71–75% of the 624 GB/s spec at the median (the reference; best-of times do not
reproduce well). A KA kernel and the broadcast are equal on the GPU.

## 2. CSR SpMV, 3-D 7-point Laplacian, scalar CSR

One work-item per row, MF6 layout (diagonal first, then columns ascending), Int32 `ia`/`ja`,
nnz ≈ 6.94 n. Results match a host Float64 SpMV (max relative error 0 in Float64,
1.5e-7 in Float32).

Byte models, for n rows, nnz non-zeros, value size `s`:
- **min** (compulsory traffic): `nnz·(s + 4) + (n + 1)·4 + n·s (x once) + n·s (y)`.
  This is the model for the speed-of-light targets below.
- **gather**: as *min* but x read once per non-zero (`nnz·s`), i.e. no cache reuse of x.

| backend | grid | rows | T | median ms | min ms | GB/s (min model) | GB/s (gather) | % of triad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rocm | 100³ | 1.00 M | Float64 | 0.387 | 0.364 | 267 | 390 | 57% |
| rocm | 100³ | 1.00 M | Float32 | 0.139 | 0.137 | 484 | 654 | (cache) |
| rocm | 216³ | 10.08 M | Float64 | **3.14** | 2.74 | **333** | 486 | 71% |
| rocm | 216³ | 10.08 M | Float32 | **1.62** | 1.35 | **423** | 572 | 95% |
| cpu | 100³ | 1.00 M | Float64 | 1.63 | 1.58 | 63 | 92 | (cache) |
| cpu | 100³ | 1.00 M | Float32 | 0.825 | 0.810 | 82 | 111 | (cache) |
| cpu | 216³ | 10.08 M | Float64 | 24.2 | 23.9 | 43 | 63 | 109% |
| cpu | 216³ | 10.08 M | Float32 | 14.2 | 13.9 | 48 | 65 | 121% |

- The 1 M matrices (Float32: 67 MB, Float64: 103 MB) partly fit the GPU's 64 MB Infinity
  Cache and the CPU's caches, so they exceed or distort the DRAM numbers. The 10 M rows are
  the DRAM-bound numbers.
- Float64 scalar CSR reaches only 71% of triad at 10 M rows (57% at 1 M). Float32 is at
  triad. Float64 is 1.95× slower than Float32 for 1.53× the bytes, so a better Float64 SpMV
  layout (e.g. a sub-group per row, SELL-C-σ, T4.5) could gain up to ~1.4×.
- On the CPU the min model exceeds triad because x reuse comes from cache and triad pays
  no write-allocate on `a` in the count; read it as "SpMV runs at memory speed".

## 3. `dot` via `mapreduce`

`mapreduce(*, +, x, y)`; "F32 → F64" is `mapreduce(mul_f64, +, x, y)` with
`mul_f64(a, b) = Float64(a) * Float64(b)` (a closure capturing `Tacc` is not isbits and
fails to compile on the GPU). Time per call includes returning the scalar to the host.
Bytes = 2 n × sizeof(data).

| backend | n | data → accumulate | median µs | min µs | GB/s | rel err |
| --- | --- | --- | --- | --- | --- | --- |
| rocm | 1 M | F32 → F32 | 50.7 | 49.4 | 158 | 4.5e-8 |
| rocm | 1 M | F32 → F64 | 60.6 | 58.5 | 132 | 0 |
| rocm | 1 M | F64 → F64 | 54.7 | 53.4 | 292 | 0 |
| rocm | 10 M | F32 → F32 | 205 | 201 | 390 | 1.7e-8 |
| rocm | 10 M | F32 → F64 | 229 | 226 | 349 | 0 |
| rocm | 10 M | F64 → F64 | 345 | 340 | 464 | 0 |
| rocm | 33.5 M | F32 → F32 | 584 | 575 | 460 | 4.1e-9 |
| rocm | 33.5 M | F32 → F64 | 658 | 645 | 408 | 0 |
| rocm | 33.5 M | F64 → F64 | 1041 | 1032 | 516 | 0 |
| cpu | 1 M | F32 → F32 | 1421 | 266 | 5.6 | 1.5e-8 |
| cpu | 1 M | F32 → F64 | 2682 | 2614 | 3.0 | 0 |
| cpu | 1 M | F64 → F64 | 2836 | 613 | 5.6 | 0 |
| cpu | 10 M | F32 → F32 | 4260 | 4038 | 18.8 | 1.7e-8 |
| cpu | 10 M | F32 → F64 | 6640 | 6257 | 12.0 | 0 |
| cpu | 10 M | F64 → F64 | 34910 | 8280 | 4.6 | 0 |
| cpu | 33.5 M | F32 → F32 | 80711 | 13986 | 3.3 | 4.1e-9 |
| cpu | 33.5 M | F32 → F64 | 134694 | 97500 | 2.0 | 0 |
| cpu | 33.5 M | F64 → F64 | 141831 | 104896 | 3.8 | 0 |

- GPU: a `mapreduce` call costs **≈ 45–50 µs** even for small n (launches + scalar copy to
  host). With 2–3 dots per Krylov iteration that is ~0.15 ms of latency per iteration, which
  matters at 1 M rows (SpMV 0.14–0.39 ms). Phase 4 should fuse dots or keep scalars on the
  device.
- F32 data accumulated in F64 costs 12–20% over plain F32 on the GPU: cheap enough for
  rule 4 of section 3.7.
- CPU: on `Array`, Base's multi-array `mapreduce` materialises `map(*, x, y)` (allocates
  76 MB at 10 M) and runs single-threaded, so it is 2–10× below memory speed. CPU engine
  code needs a KA reduction kernel instead. The CPU dot rows (idle machine) vary strongly
  from run to run because of GC of the `map` temporary (e.g. 10 M F64: 35 ms and 8.4 ms
  medians in two runs; median ≫ min); they are indicative only and not a target.

## 4. Atomic add `Atomix.@atomic y[idx[i]] += v[i]`

16.8 M updates per launch, `v = 0.5` (exact sums, so the check is exact). Patterns: 16
targets (heavy contention), n/8 targets scattered by a multiplicative hash (≈ 8 updates per
target, few collisions), and `idx = i` (no collisions). "plain" is the same kernel with a
non-atomic `+=` (racy; timing reference).

| backend | pattern | T | atomic G upd/s | plain G upd/s | atomic correct |
| --- | --- | --- | --- | --- | --- |
| rocm | 16 targets | Float32 | 22.8 | 65.4 | yes |
| rocm | 16 targets | Float64 | **0.040** (419 ms) | 39.7 | yes |
| rocm | n/8, hashed | Float32 | 9.5 | 8.3 | yes |
| rocm | n/8, hashed | Float64 | 6.0 | 7.0 | yes |
| rocm | none | Float32 | 32.6 | 33.1 | yes |
| rocm | none | Float64 | 18.0 | 18.0 | yes |
| cpu | 16 targets | Float32 | 0.06 | 0.92 | yes |
| cpu | 16 targets | Float64 | 0.06 | 0.45 | yes |
| cpu | n/8, hashed | Float32 | 1.26 | 5.6 | yes |
| cpu | n/8, hashed | Float64 | 0.47 | 3.0 | yes |
| cpu | none | Float32 | 3.2 | 3.4 | yes |
| cpu | none | Float64 | 1.9 | 2.0 | yes |

CPU rows are from a run on the idle machine; between runs the hashed and no-collision
atomic rows vary by up to ~25% (contention, scheduling), so they are indicative only.

**Verdict:** Float64 `Atomix.@atomic` add is **correct** on the RX 7800 XT (and on POCL)
in every pattern. gfx1101 has no Float64 atomic-add instruction: the ISA shows
`global_atomic_cmpswap_b64` (a compare-and-swap loop) for Float64 versus a native
`global_atomic_add_f32` for Float32. With few collisions Float64 atomics cost ~15% over a
plain store (6.0 vs 7.0 G upd/s); under heavy contention they are **~570× slower than
Float32 atomics** (0.04 vs 22.8 G upd/s). This confirms section 3.4: no atomics in assembly
or reductions. Float64 atomics are acceptable only for rare, low-contention updates.

## Speed-of-light targets for Phase 4

Global 5′: n ≈ 6 M active rows, two layers, so nnz ≈ 6 n = 36 M (diagonal, 4 horizontal,
1 vertical neighbour), Int32 indices, min byte model. "Triad" uses the measured triad
bandwidth (466 GB/s Float64, 445 GB/s Float32); "measured" uses the scalar CSR bandwidth at
10 M rows (333 / 423 GB/s). The 216³ 3-D Laplacian is only a proxy: the 2-layer pattern
has fewer non-zeros per row and different x reuse, so its achieved bandwidth may differ.

| kernel (n = 6 M) | precision | bytes | at triad | at measured scalar CSR / dot |
| --- | --- | --- | --- | --- |
| SpMV | Float64 | 552 MB | **1.18 ms** | 1.66 ms |
| SpMV | Float32 | 360 MB | **0.81 ms** | 0.85 ms |
| dot | Float64 | 96 MB | 0.21 ms | 0.21 ms (+ ~50 µs latency) |
| dot | Float32 → Float64 | 48 MB | 0.11 ms | 0.14 ms |
| axpy `y = a x + y` | Float64 | 144 MB | 0.31 ms | |
| axpy | Float32 | 72 MB | 0.16 ms | |

A Jacobi-preconditioned CG iteration (1 SpMV, 2 dots, 3 axpy, 1 diagonal scaling counted
as an axpy; no fusion) on global 5′ is therefore ≈ **3.3 ms in Float64**
(1.66 + 2 × 0.21 + 4 × 0.31) and ≈ **1.8 ms in Float32** with Float64 dots
(0.85 + 2 × 0.14 + 4 × 0.16) at the measured rates; fusing the vector updates lowers this.
The CPU (POCL, 8 threads) is 8–12× slower (40 GB/s triad, 43–48 GB/s SpMV): SpMV
≈ 13 ms (Float64) / 7.5 ms (Float32) at 6 M rows.
