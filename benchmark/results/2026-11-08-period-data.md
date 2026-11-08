# Period-data path (T4.6)

Measured 2026-11-08 on the machine of `2026-11-07.md` (RX 7800 XT, Ryzen 7 7700, Julia
1.12.6 with one thread, ROCm, KernelAbstractions 0.10.5), with T4.2/T4.3 agents active on
the machine but no other benchmark running (GPU runs serialised with a lock).

## What changed

The cost per new stress period on `global5m` (RCH 3.26 M records, RIV 0.76 M records get a
new PERIOD block every period) was host work and uploads in `apply_period!`:

1. **Structure fast path**: a block with the same cells as the block in force (`nodeu`, or
   the layers for array input) keeps the reduced nodes and the node→record map; only
   `bound` is copied. The CHD IBOUND is rebuilt only when a CHD package gets new cells.
2. **One fused pass**: the values are read from the PERIOD block directly (`BoundSource`;
   `bound_user` is filled only when the stepping API asks, `bound_user!`), the multiplier
   and the offsets are applied in Float64 (per-record references `rrec = r[nodelist]`
   instead of a gather), the cast to `T` goes into a staging buffer registered with
   `pin_memory!` (new ROCm method), and one `copyto!` of `ncol·nbound` values. Float64
   columns without multiplier or offset (RCH in `:double`) are copied without a pass.
   Threaded when Julia has threads (not used below).
3. **Stepping API**: BOUND and NODELIST mirrors are versioned (copied only when the engine
   values change); dirty checks by `memcmp`; no copy before a push.

## Results (global5m, rocm)

| | `:double` | `:single` |
| --- | --- | --- |
| period data per period, `main` (`apply_period!(model, kper)` + sync, median of 12) | 52.2 ms | 52.0 ms |
| period data per period, T4.6 (same script) | 7.0 ms | 9.2 ms |
| period data per daily step, T4.1 baseline (`benchmark/run.jl`, timed) | 64 ms | 61 ms |
| period data per daily step, T4.6 (`benchmark/run.jl rocm global5m steps=8`, timed) | **5.6 ms** | **9.1 ms** |
| daily step wall time (untimed, mean of 8) | 0.563 s (baseline 0.618) | 0.187 s (baseline 0.230) |

Per period in T4.6 (`:double` / `:single`): RCH 2.6 / 5.6 ms (the `:single` cast pass of
3.26 M values is memory bound on one core, ~4 ms), RIV 3.9 ms (riv_ck 0.5 ms, pass 2 ms,
copy 0.65 ms), cell comparison 0.7 ms. With `julia -t 8` the passes run threaded.

## Bitwise check

Heads after every step, iteration counts (outer, inner), hncg, every budget term (rates and
cumulative volumes) and the SIMVALS of every package, `main` vs this branch, compared with
`isequal`: **bitwise identical** for indus (7 steps) and test/models/coupling_rde (10
steps) on CPU and rocm in `:double` and `:single`, for global5m (4 steps) on rocm in both
precisions, and for runs that write RIV stages through a `get_value_ptr` mirror every step
(indus rocm `:double` and CPU `:single`, coupling_rde CPU `:double` and rocm `:single`,
global5m rocm `:single`).
